import CoreMotion
import Foundation

/// Hard-iron (offset) + soft-iron (per-axis scale) correction derived from a short
/// calibration gesture, per the ellipsoid-fit approach in 南投縣政府 "智慧型手機的磁力計
/// 改正輔助慣性行人室內定位": collect magnetometer samples while rotating the phone
/// through many orientations, then offset = (max+min)/2 and scale = avgRange/axisRange
/// per axis. This is the axis-aligned (min/max) simplification of a full rotated-
/// ellipsoid fit — it corrects the dominant hard-iron bias and per-axis soft-iron
/// scaling without needing an eigen-decomposition, at some cost in accuracy versus the
/// paper's full method.
struct MagnetometerCalibration: Codable {
    var offsetX: Double = 0
    var offsetY: Double = 0
    var offsetZ: Double = 0
    var scaleX: Double = 1
    var scaleY: Double = 1
    var scaleZ: Double = 1

    static let identity = MagnetometerCalibration()

    func apply(x: Double, y: Double, z: Double) -> (x: Double, y: Double, z: Double) {
        ((x - offsetX) * scaleX, (y - offsetY) * scaleY, (z - offsetZ) * scaleZ)
    }
}

/// Runs a short data-collection session (ask the user to rotate the phone through
/// varied orientations, e.g. a slow figure-8) and derives a MagnetometerCalibration
/// from the min/max magnetic field seen on each axis.
final class MagnetometerCalibrator {
    private let motionManager = CMMotionManager()

    private(set) var isCalibrating = false
    var onSampleCount: ((Int) -> Void)?

    func start(duration: TimeInterval = 12, completion: @escaping (MagnetometerCalibration) -> Void) {
        guard !isCalibrating, motionManager.isMagnetometerAvailable else {
            completion(.identity)
            return
        }
        isCalibrating = true

        var minX = Double.greatestFiniteMagnitude, maxX = -Double.greatestFiniteMagnitude
        var minY = Double.greatestFiniteMagnitude, maxY = -Double.greatestFiniteMagnitude
        var minZ = Double.greatestFiniteMagnitude, maxZ = -Double.greatestFiniteMagnitude
        var sampleCount = 0

        motionManager.magnetometerUpdateInterval = 0.05
        motionManager.startMagnetometerUpdates(to: .main) { [weak self] data, _ in
            guard let self, let field = data?.magneticField else { return }
            minX = min(minX, field.x); maxX = max(maxX, field.x)
            minY = min(minY, field.y); maxY = max(maxY, field.y)
            minZ = min(minZ, field.z); maxZ = max(maxZ, field.z)
            sampleCount += 1
            self.onSampleCount?(sampleCount)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + duration) { [weak self] in
            guard let self else { return }
            self.motionManager.stopMagnetometerUpdates()
            self.isCalibrating = false

            guard sampleCount > 20, maxX > minX, maxY > minY, maxZ > minZ else {
                completion(.identity)
                return
            }
            let rangeX = (maxX - minX) / 2
            let rangeY = (maxY - minY) / 2
            let rangeZ = (maxZ - minZ) / 2
            let avgRange = (rangeX + rangeY + rangeZ) / 3

            let calibration = MagnetometerCalibration(
                offsetX: (maxX + minX) / 2,
                offsetY: (maxY + minY) / 2,
                offsetZ: (maxZ + minZ) / 2,
                scaleX: avgRange / rangeX,
                scaleY: avgRange / rangeY,
                scaleZ: avgRange / rangeZ
            )
            completion(calibration)
        }
    }
}
