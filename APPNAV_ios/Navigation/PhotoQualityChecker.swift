import CoreGraphics
import Foundation

/// On-device "is this photo worth uploading?" checks, run before the upload.
///
/// Same measures and thresholds as the backend (backend/server/photo_quality.py), so a
/// photo the phone lets through is not rejected by the server for the same reason:
/// - sharpness = variance of a 4-neighbour Laplacian on a ≤512 px grayscale copy (< 150 → blurred)
/// - brightness = mean gray level 0–255 (< 45 → too dark)
/// Thresholds come from the 測試路線1 photos (sharpness 1662–2597, brightness 106–138)
/// vs blurred / darkened copies of them.
///
/// Plus a tilt check from the device-motion gravity vector: reference photos were taken
/// with the phone upright facing down the aisle; a photo of the floor or ceiling
/// matches nothing.
///
/// Pure CoreGraphics / Foundation — unit tested in scripts/swift_logic_tests.
enum PhotoQualityChecker {
    static let minSharpness = 150.0
    static let minBrightness = 45.0
    /// Maximum angle between the phone's long axis and vertical (portrait, upright).
    static let maxTiltDegrees = 25.0

    static let blurHint = "照片太模糊，請拿穩手機、停一下再拍"
    static let darkHint = "照片太暗，請開啟閃光燈（右上角燈泡）再拍"
    static let tiltHint = "請把手機拿直、對準走道前方再拍"

    struct Measurement: Equatable {
        var sharpness: Double
        var brightness: Double
    }

    /// Sharpness and brightness of an image (downscaled to ≤512 px gray).
    static func measure(_ image: CGImage) -> Measurement? {
        let scale = min(1.0, 512.0 / Double(max(image.width, image.height)))
        let w = max(3, Int(Double(image.width) * scale))
        let h = max(3, Int(Double(image.height) * scale))
        var pixels = [UInt8](repeating: 0, count: w * h)
        let ok: Bool = pixels.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            ctx.interpolationQuality = .medium
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }

        var sum = 0.0
        for p in pixels { sum += Double(p) }
        let brightness = sum / Double(pixels.count)

        var lapSum = 0.0, lapSq = 0.0
        var n = 0.0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let c = Double(pixels[y * w + x])
                let lap = Double(pixels[(y - 1) * w + x]) + Double(pixels[(y + 1) * w + x])
                    + Double(pixels[y * w + x - 1]) + Double(pixels[y * w + x + 1]) - 4 * c
                lapSum += lap
                lapSq += lap * lap
                n += 1
            }
        }
        let mean = lapSum / n
        return Measurement(sharpness: lapSq / n - mean * mean, brightness: brightness)
    }

    /// Retake hint for an image, or nil when it is fine.
    static func hint(for m: Measurement) -> String? {
        if m.brightness < minBrightness { return darkHint }
        if m.sharpness < minSharpness { return blurHint }
        return nil
    }

    /// Angle (degrees) between the phone's long axis and vertical, from CMDeviceMotion
    /// gravity (device frame: upright portrait ≈ (0, −1, 0)).
    static func tiltDegrees(gravity: (x: Double, y: Double, z: Double)) -> Double {
        let g = (gravity.x * gravity.x + gravity.y * gravity.y + gravity.z * gravity.z).squareRoot()
        guard g > 0.1 else { return 0 }
        let cosA = min(1, max(-1, -gravity.y / g))
        return acos(cosA) * 180 / .pi
    }

    static func isUpright(gravity: (x: Double, y: Double, z: Double)) -> Bool {
        tiltDegrees(gravity: gravity) <= maxTiltDegrees
    }
}
