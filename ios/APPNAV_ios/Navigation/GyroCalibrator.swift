import CoreMotion
import Foundation

/// A gyroscope's static zero-rate offset — unlike the magnetometer/accelerometer, this is a
/// real, measurable hardware bias: held perfectly still, a MEMS gyro's raw output isn't
/// exactly (0, 0, 0), and every hand-rolled heading formula that integrates or projects the
/// raw `rotationRate` (`TiltCompensatedCompass.verticalAxisYawRate`, the Holding/Pocket/Swing
/// eq. 3.20-3.23 formulas, `HeadingEKF`'s gyro-predict step) inherits that offset as a slow,
/// steady phantom rotation — exactly the kind of error the "加速度＋陀螺儀" and "Game Rotation
/// Vector" comparison methods are seen accumulating over a walk (see `SensorComboTracker`'s
/// doc comment). Standard MEMS-IMU practice: average raw readings while stationary, subtract
/// that average from every subsequent reading.
struct GyroBias: Codable {
    var x: Double = 0
    var y: Double = 0
    var z: Double = 0

    static let identity = GyroBias()

    /// Bias-corrected reading — subtract this offset before the value is used in any
    /// downstream heading formula.
    func apply(x: Double, y: Double, z: Double) -> (x: Double, y: Double, z: Double) {
        (x - self.x, y - self.y, z - self.z)
    }
}

/// Runs a short "hold the phone still" data-collection session and derives a `GyroBias` from
/// the average raw gyroscope reading over that window. Mirrors `MagnetometerCalibrator`'s
/// shape (same `start(duration:completion:)`/`onSampleCount` pattern), but the underlying
/// technique is different: this only works because the gesture is "don't move at all" — any
/// real rotation during the window would get averaged into the bias estimate and wrongly
/// subtracted from every future reading.
final class GyroCalibrator {
    private let motionManager = CMMotionManager()

    private(set) var isCalibrating = false
    var onSampleCount: ((Int) -> Void)?

    func start(duration: TimeInterval = 5, completion: @escaping (GyroBias) -> Void) {
        guard !isCalibrating, motionManager.isGyroAvailable else {
            completion(.identity)
            return
        }
        isCalibrating = true

        var sumX = 0.0, sumY = 0.0, sumZ = 0.0
        var sampleCount = 0

        motionManager.gyroUpdateInterval = 0.02
        motionManager.startGyroUpdates(to: .main) { [weak self] data, _ in
            guard let self, let data else { return }
            sumX += data.rotationRate.x
            sumY += data.rotationRate.y
            sumZ += data.rotationRate.z
            sampleCount += 1
            self.onSampleCount?(sampleCount)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + duration) { [weak self] in
            guard let self else { return }
            self.motionManager.stopGyroUpdates()
            self.isCalibrating = false

            guard sampleCount > 20 else {
                completion(.identity)
                return
            }
            let n = Double(sampleCount)
            completion(GyroBias(x: sumX / n, y: sumY / n, z: sumZ / n))
        }
    }
}
