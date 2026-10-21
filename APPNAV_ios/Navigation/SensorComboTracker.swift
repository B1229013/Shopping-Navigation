import CoreMotion
import Foundation

struct PointXY: Codable {
    var x: Double
    var y: Double
}

/// One footstep's result across all five heading methods being compared — see
/// `HeadingMethodID` and `SensorComboTracker`'s doc comment. Dictionaries are keyed by
/// `HeadingMethodID.rawValue` (not the enum itself) purely so this stays trivially
/// `Codable` for JSON/CSV export.
struct MultiHeadingPoint: Codable {
    var timestamp: Double
    var stepCount: Int
    var distanceMeters: Double
    var headings: [String: Double]
    var positions: [String: PointXY]
}

/// Runs all five heading-computation methods simultaneously off ONE shared step-detection
/// stream (reusing `StepPeakDetector` and the Kim step-length model unchanged — this is
/// purely a heading-method comparison, not a re-test of step detection), producing one
/// independent (x, y) trajectory per method so they can be plotted and compared from the
/// same walk instead of needing a separate walk per method.
///
/// Mirrors Android's sensor terminology: `rotationVector`/`gameRotationVector` are the
/// two "already fused by the OS" methods (matching `TYPE_ROTATION_VECTOR`/
/// `TYPE_GAME_ROTATION_VECTOR`); `accelMag`/`accelGyro`/`accelMagGyro` are hand-rolled from
/// raw sensor components via `TiltCompensatedCompass`.
///
/// Needs **two** concurrent `CMMotionManager` instances because iOS ties "does this attitude
/// use the magnetometer" to the reference frame chosen at subscription time, and the two
/// pieces of information (magnetic-referenced vs. magnetometer-free fused attitude) can't
/// both come from one subscription:
/// - `magneticMotionManager` (`.xMagneticNorthZVertical`) supplies gravity/magneticField/
///   rotationRate for `accelMag`/`accelGyro`/`accelMagGyro`, and `motion.heading` directly
///   for `rotationVector`.
/// - `arbitraryMotionManager` (`.xArbitraryZVertical` — deliberately *not*
///   `.xArbitraryCorrectedZVertical`, which does pull in some magnetometer-based drift
///   correction and so isn't a true magnetometer-free match for Android's Game Rotation
///   Vector) supplies `attitude.yaw` for `gameRotationVector`.
///
/// Running two `CMMotionManager` instances concurrently is a documented, supported iOS
/// pattern, but this is the first time this project has relied on it — worth confirming on
/// a real device that both deliver reliably at the same time before trusting the comparison.
final class SensorComboTracker {
    private let stepDetector = StepPeakDetector()
    private let accelMotionManager = CMMotionManager()
    private let magneticMotionManager = CMMotionManager()
    private let arbitraryMotionManager = CMMotionManager()

    private(set) var isTracking = false
    private var gaitProfile: GaitProfile
    private var gyroBias: GyroBias
    private var detectedStepCount = 0
    private var lastStepTimeForFrequency: Date?
    private var cumulativeStepDistance: Double = 0

    private var positions: [HeadingMethodID: (x: Double, y: Double)] = [:]
    private var currentHeadings: [HeadingMethodID: Double] = [:]

    /// Seeded once from the first valid `accelMag` reading, then integrated purely from
    /// gyro yaw rate forever after — deliberately never corrected again, so it shows how
    /// much a magnetometer-free heading drifts over a real walk.
    private var accelGyroHeading: Double?
    private let comboEKF = HeadingEKF()
    private var lastMagneticSampleTime: Date?

    private var latestGravity: (x: Double, y: Double, z: Double) = (0, 0, -1)
    private var latestMagnetic: (x: Double, y: Double, z: Double) = (0, 0, 0)

    var onPathPoint: ((MultiHeadingPoint) -> Void)?
    /// Fires on every magnetic-reference device-motion tick (~10Hz) with the current
    /// heading for all five methods, independent of stepping — mirrors
    /// `MotionPathTracker.onHeadingUpdate` for the same reason (live UI feedback without
    /// waiting for the next step).
    var onHeadingsUpdate: (([HeadingMethodID: Double]) -> Void)?

    static var isAvailable: Bool { CMPedometer.isStepCountingAvailable() }

    init() {
        gaitProfile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
        gyroBias = PersistenceService.shared.load(GyroBias.self, from: "gyro_bias.json") ?? .identity
        stepDetector.onStep = { [weak self] time in
            self?.handleStepDetected(at: time)
        }
    }

    func reloadGaitProfile() {
        gaitProfile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
    }

    /// Call after `GyroCalibrator` finishes (run from `SensorTestView`) so a round started
    /// here right after calibrating picks it up without needing the app relaunched — same
    /// reasoning as `reloadGaitProfile()`.
    func reloadGyroBias() {
        gyroBias = PersistenceService.shared.load(GyroBias.self, from: "gyro_bias.json") ?? .identity
    }

    func start() {
        guard !isTracking, Self.isAvailable else { return }
        isTracking = true
        detectedStepCount = 0
        lastStepTimeForFrequency = nil
        cumulativeStepDistance = 0
        accelGyroHeading = nil
        lastMagneticSampleTime = nil
        latestGravity = (0, 0, -1)
        latestMagnetic = (0, 0, 0)
        stepDetector.reset()
        comboEKF.reset()
        for method in HeadingMethodID.allCases {
            positions[method] = (0, 0)
            currentHeadings[method] = 0
        }

        // Record the true starting point explicitly, same reasoning as
        // MotionPathTracker.start() — every subsequent point already reflects position
        // *after* that step, so without this the plotted origin and the first point would
        // have a visible gap.
        emitPoint(at: Date())

        if accelMotionManager.isAccelerometerAvailable {
            accelMotionManager.accelerometerUpdateInterval = 0.02 // 50Hz — needed to resolve a step's peak
            accelMotionManager.startAccelerometerUpdates(to: .main) { [weak self] data, _ in
                guard let self, let data else { return }
                self.stepDetector.ingest(x: data.acceleration.x, y: data.acceleration.y, z: data.acceleration.z, at: Date())
            }
        }

        if magneticMotionManager.isDeviceMotionAvailable {
            magneticMotionManager.deviceMotionUpdateInterval = 0.1
            magneticMotionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: .main) { [weak self] motion, _ in
                guard let self, let motion else { return }
                self.ingestMagneticMotion(motion)
            }
        }

        if arbitraryMotionManager.isDeviceMotionAvailable {
            arbitraryMotionManager.deviceMotionUpdateInterval = 0.1
            arbitraryMotionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] motion, _ in
                guard let self, let motion else { return }
                self.ingestArbitraryMotion(motion)
            }
        }
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        accelMotionManager.stopAccelerometerUpdates()
        magneticMotionManager.stopDeviceMotionUpdates()
        arbitraryMotionManager.stopDeviceMotionUpdates()
    }

    /// Snapshot of every method's current position — used to stamp a waypoint marker at
    /// the moment the user taps "拍照標記" in `SensorComboTestView`, before the camera
    /// sheet even opens (so the recorded position reflects where they were standing when
    /// they decided to take the photo, not wherever they end up after the camera UI closes).
    func currentPositionsSnapshot() -> [HeadingMethodID: (x: Double, y: Double)] {
        positions
    }

    /// One-off nudge to a single method's running position — used by `TopoMapTestView`'s
    /// online mode when `TopologicalMap` recognizes a revisit of an already-mapped node for
    /// whichever method is currently selected to drive the map. Only that one method's
    /// trajectory is adjusted; the other four keep accumulating independently, since the
    /// whole point of this tracker is comparing their *uncorrected* drift against each other.
    func applyDriftCorrection(method: HeadingMethodID, dx: Double, dy: Double) {
        guard var pos = positions[method] else { return }
        pos.x += dx
        pos.y += dy
        positions[method] = pos
    }

    private func ingestMagneticMotion(_ motion: CMDeviceMotion) {
        latestGravity = (motion.gravity.x, motion.gravity.y, motion.gravity.z)
        latestMagnetic = (motion.magneticField.field.x, motion.magneticField.field.y, motion.magneticField.field.z)

        let now = Date()
        let dt = lastMagneticSampleTime.map { now.timeIntervalSince($0) } ?? 0
        lastMagneticSampleTime = now

        let correctedGyro = gyroBias.apply(x: motion.rotationRate.x, y: motion.rotationRate.y, z: motion.rotationRate.z)
        let yawRateDegPerSec = TiltCompensatedCompass.verticalAxisYawRate(gyro: correctedGyro, gravity: latestGravity) * 180 / .pi

        if dt > 0 {
            if accelGyroHeading != nil {
                accelGyroHeading = Self.normalize(accelGyroHeading! + yawRateDegPerSec * dt)
            }
            comboEKF.predict(gyroYawRateDegPerSec: yawRateDegPerSec, dt: dt)
        }

        if let accelMagHeading = TiltCompensatedCompass.heading(magnetic: latestMagnetic, gravity: latestGravity) {
            currentHeadings[.accelMag] = accelMagHeading
            if accelGyroHeading == nil {
                accelGyroHeading = accelMagHeading
            }
            comboEKF.correct(measuredHeadingDegrees: accelMagHeading)
        }

        currentHeadings[.accelGyro] = accelGyroHeading ?? 0
        currentHeadings[.accelMagGyro] = comboEKF.headingDegrees
        if motion.heading >= 0 {
            currentHeadings[.rotationVector] = motion.heading
        }

        onHeadingsUpdate?(currentHeadings)
    }

    /// `CMAttitude.yaw` follows the right-hand rule about the (up-pointing) z-axis: positive
    /// is counterclockwise viewed from above. Every heading this app uses (compass headings,
    /// and the `x = sin(heading)`/`y = cos(heading)` position formula in `handleStepDetected`)
    /// is clockwise-positive instead — North 0° → East 90° → South 180° → West 270°. Using
    /// `yaw` directly without negating was an actual bug that shipped for a while: it made
    /// Game Rotation Vector turn the opposite way from every other method for the same
    /// physical turn, which (by the same sin/cos sign-flip reasoning as `TiltCompensatedCompass
    /// .heading`'s gravity-sign bug — see that fix's comment) mirrors the whole walked path
    /// left/right around the north–south line. Confirmed on two real indoor walks via
    /// `SensorComboTestView`'s comparison chart, where Game Rotation Vector was the one
    /// consistently mirrored against the actual walking route.
    private func ingestArbitraryMotion(_ motion: CMDeviceMotion) {
        currentHeadings[.gameRotationVector] = Self.normalize(-motion.attitude.yaw * 180 / .pi)
    }

    private static func normalize(_ deg: Double) -> Double {
        var d = deg.truncatingRemainder(dividingBy: 360)
        if d < 0 { d += 360 }
        return d
    }

    /// Same Kim height/frequency step-length model as `MotionPathTracker.handleStepDetected(at:)`
    /// — shared across all five methods so the comparison is only about heading, not step length.
    private func handleStepDetected(at time: Date) {
        detectedStepCount += 1

        let stepLengthCM: Double
        if let last = lastStepTimeForFrequency {
            let interval = time.timeIntervalSince(last)
            let frequencyHz = interval > 0 ? 1.0 / interval : 0
            stepLengthCM = frequencyHz > 0 ? gaitProfile.k * gaitProfile.heightCM * frequencyHz.squareRoot() : gaitProfile.k1 * gaitProfile.heightCM
        } else {
            stepLengthCM = gaitProfile.k1 * gaitProfile.heightCM
        }
        lastStepTimeForFrequency = time
        let stepLength = stepLengthCM / 100
        cumulativeStepDistance += stepLength

        for method in HeadingMethodID.allCases {
            let heading = currentHeadings[method] ?? 0
            let rad = heading * .pi / 180
            var pos = positions[method] ?? (0, 0)
            pos.x += stepLength * sin(rad)
            pos.y += stepLength * cos(rad)
            positions[method] = pos
        }

        emitPoint(at: time)
    }

    private func emitPoint(at time: Date) {
        var headingsSnapshot: [String: Double] = [:]
        var positionsSnapshot: [String: PointXY] = [:]
        for method in HeadingMethodID.allCases {
            headingsSnapshot[method.rawValue] = currentHeadings[method] ?? 0
            let p = positions[method] ?? (0, 0)
            positionsSnapshot[method.rawValue] = PointXY(x: p.x, y: p.y)
        }
        onPathPoint?(MultiHeadingPoint(
            timestamp: time.timeIntervalSince1970 * 1000,
            stepCount: detectedStepCount,
            distanceMeters: cumulativeStepDistance,
            headings: headingsSnapshot,
            positions: positionsSnapshot
        ))
    }
}
