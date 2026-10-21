import Foundation
import CoreML
import UIKit
import Accelerate

/// On-device visual localization using DINOv2 (Core ML).
///
/// Compares a query photo's embedding against pre-computed reference embeddings
/// to find the closest matching node in the topological map.
/// Typical latency: ~30ms on iPhone Neural Engine.
final class DINOLocalizationService {
    static let shared = DINOLocalizationService()

    struct Match {
        let nodeId: Int
        let slot: String
        let similarity: Float
        let photoFile: String
    }

    private var model: MLModel?
    private var refEmbeddings: [[Float]] = []
    private var refMeta: [RefEntry] = []
    private(set) var isLoaded = false

    private let embDim = 768

    // MARK: - Loading / Unloading

    /// Release the Core ML model and reference data to free memory.
    /// Call when navigation ends. `load()` can be called again later.
    func unload() {
        model = nil
        refEmbeddings = []
        refMeta = []
        isLoaded = false
    }

    /// Load model and reference data. Call once at app launch (background thread).
    func load() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard let modelURL = Bundle.main.url(forResource: "DINOv2", withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: "DINOv2", withExtension: "mlpackage") else {
            throw DINOError.modelNotFound
        }
        model = try MLModel(contentsOf: modelURL, configuration: config)

        guard let embURL = Bundle.main.url(forResource: "dino_ref_embeddings", withExtension: "bin"),
              let metaURL = Bundle.main.url(forResource: "dino_ref_meta", withExtension: "json") else {
            throw DINOError.referenceDataNotFound
        }

        let embData = try Data(contentsOf: embURL)
        let metaData = try Data(contentsOf: metaURL)
        refMeta = try JSONDecoder().decode([RefEntry].self, from: metaData)

        let floatCount = embData.count / MemoryLayout<Float>.size
        guard floatCount == refMeta.count * embDim else {
            throw DINOError.embeddingDimensionMismatch
        }

        refEmbeddings = embData.withUnsafeBytes { ptr in
            let floats = ptr.bindMemory(to: Float.self)
            return stride(from: 0, to: floatCount, by: embDim).map { start in
                Array(floats[start..<start + embDim])
            }
        }

        isLoaded = true
    }

    // MARK: - Localization

    /// Match a query image against reference embeddings.
    /// Returns top-K matches sorted by similarity (highest first), deduplicated by node.
    func localize(image: UIImage, topK: Int = 5) -> [Match] {
        guard isLoaded, let model else { return [] }
        guard let embedding = computeEmbedding(image: image, model: model) else { return [] }
        return matchEmbedding(embedding, topK: topK)
    }

    /// Match from a CGImage (e.g., from camera buffer).
    func localize(cgImage: CGImage, topK: Int = 5) -> [Match] {
        let image = UIImage(cgImage: cgImage)
        return localize(image: image, topK: topK)
    }

    // MARK: - Embedding

    private func computeEmbedding(image: UIImage, model: MLModel) -> [Float]? {
        guard let resized = image.resized(to: CGSize(width: 224, height: 224)),
              let pixelBuffer = resized.toCVPixelBuffer() else {
            return nil
        }

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["image": pixelBuffer]
        ) else { return nil }

        guard let output = try? model.prediction(from: input),
              let embValue = output.featureValue(for: "embedding"),
              let embArray = embValue.multiArrayValue else {
            return nil
        }

        var embedding = [Float](repeating: 0, count: embDim)
        let ptr = embArray.dataPointer.bindMemory(to: Float.self, capacity: embDim)
        for i in 0..<embDim {
            embedding[i] = ptr[i]
        }

        // L2 normalize
        var norm: Float = 0
        vDSP_svesq(embedding, 1, &norm, vDSP_Length(embDim))
        norm = sqrtf(norm)
        if norm > 0 {
            var scale = 1.0 / norm
            vDSP_vsmul(embedding, 1, &scale, &embedding, 1, vDSP_Length(embDim))
        }

        return embedding
    }

    // MARK: - Matching

    private func matchEmbedding(_ query: [Float], topK: Int) -> [Match] {
        var scores: [(Int, Float)] = []

        for (idx, refEmb) in refEmbeddings.enumerated() {
            var sim: Float = 0
            vDSP_dotpr(query, 1, refEmb, 1, &sim, vDSP_Length(embDim))
            scores.append((idx, sim))
        }

        scores.sort { $0.1 > $1.1 }

        // Deduplicate by node — keep best match per node
        var seen = Set<Int>()
        var results: [Match] = []
        for (idx, sim) in scores {
            let meta = refMeta[idx]
            if seen.contains(meta.nid) { continue }
            seen.insert(meta.nid)
            results.append(Match(
                nodeId: meta.nid,
                slot: meta.slot,
                similarity: sim,
                photoFile: meta.file
            ))
            if results.count >= topK { break }
        }

        return results
    }
}

// MARK: - Types

extension DINOLocalizationService {
    enum DINOError: LocalizedError {
        case modelNotFound
        case referenceDataNotFound
        case embeddingDimensionMismatch

        var errorDescription: String? {
            switch self {
            case .modelNotFound: return "DINOv2.mlpackage not found in bundle"
            case .referenceDataNotFound: return "Reference embeddings or metadata not found in bundle"
            case .embeddingDimensionMismatch: return "Embedding file size doesn't match metadata count"
            }
        }
    }

    struct RefEntry: Codable {
        let nid: Int
        let slot: String
        let file: String
    }
}

// MARK: - UIImage helpers

private extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    func toCVPixelBuffer() -> CVPixelBuffer? {
        guard let cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height

        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_32BGRA, attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
