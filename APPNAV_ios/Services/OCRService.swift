import Foundation
import Vision
import UIKit

final class OCRService {
    static let shared = OCRService()

    // iOS built-in OCR using Vision framework
    func recognizeText(from image: UIImage) async throws -> String {
        guard let cgImage = image.cgImage else { throw OCRError.invalidImage }

        return try await withCheckedThrowingContinuation { continuation in
            let request = VNRecognizeTextRequest { request, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                let text = (request.results as? [VNRecognizedTextObservation])?
                    .compactMap { $0.topCandidates(1).first?.string }
                    .joined(separator: "\n") ?? ""
                continuation.resume(returning: text)
            }
            request.recognitionLanguages = ["zh-Hant", "en-US"]
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }

    // PaddleOCR API (optional, matches Android implementation)
    func callPaddleOCR(image: UIImage, apiUrl: String, accessToken: String) async throws -> String {
        guard let jpegData = image.jpegData(compressionQuality: 0.9) else {
            throw OCRError.invalidImage
        }
        let base64Image = jpegData.base64EncodedString()

        let body: [String: Any] = [
            "file": base64Image,
            "fileType": 1,
            "visualize": false
        ]
        let bodyData = try JSONSerialization.data(withJSONObject: body)

        var request = URLRequest(url: URL(string: apiUrl)!)
        request.httpMethod = "POST"
        request.setValue("token \(accessToken)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = bodyData
        request.timeoutInterval = 120

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw OCRError.apiError("PaddleOCR API error")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let result = json["result"] as? [String: Any]
        let ocrResults = result?["ocrResults"] as? [[String: Any]] ?? []
        var allText: [String] = []
        for page in ocrResults {
            if let pruned = page["prunedResult"] as? [String: Any],
               let recTexts = pruned["rec_texts"] as? [String] {
                allText.append(contentsOf: recTexts)
            }
        }
        return allText.joined(separator: "\n")
    }

    enum OCRError: Error {
        case invalidImage
        case apiError(String)
    }
}
