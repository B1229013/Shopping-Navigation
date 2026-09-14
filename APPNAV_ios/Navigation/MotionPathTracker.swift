import CoreMotion
import Foundation

/// A single estimated position sample in the local pedestrian-dead-reckoning
/// coordinate system (meters, origin = wherever tracking started, x = east, y = north).
struct PathPoint: Codable, Equatable {
    var x: Double
    var y: Double
    var headingDegrees: Double
    var timestamp: Double
    var stepCount: Int = 0
    var distanceMeters: Double = 0
}

/// One throttled snapshot of every raw sensor input, independent of step detection —
/// exists so a walk can be logged once and re-analyzed offline against different
/// algorithm parameters/formulas afterward, instead of needing a fresh physical walk
/// for every change (see `MotionPathTracker.onRawSample`). Modeled after the equivalent
/// CSV export in the Android teammate's `SensorLabScreen.kt`.
struct RawSensorSample: Codable {
    var timestamp: Double
    var accelX: Double
    var accelY: Double
    var accelZ: Double
    var gyroX: Double
    var gyroY: Double
    var gyroZ: Double
    var magX: Double
    var magY: Double
    var magZ: Double
    var headingDegrees: Double
    var stepCount: Int
}

/// Per-user gait parameters for the step-length model in 劉嘉心〈基於行人航位推算之室內定位
/// 研究〉(國立臺北科技大學電機工程系碩士論文, 2020), §3.5.6: `k`/`k1` vary by user (the thesis
/// cites 文獻[16]: male k≈0.3139/k1≈0.415, female k≈0.2975/k1≈0.413) so they need calibrating
/// per person via `GaitCalibrator`, not used as universal constants.
struct GaitProfile: Codable {
    var heightCM: Double = 170
    /// Step-length coefficient for steps *after* the first one — see `GaitCalibrator` and
    /// `MotionPathTracker.handleStepDetected(at:)`.
    var k: Double = 0.305
    /// Step-length coefficient for the first step after standing still (thesis default,
    /// averaged from the cited male/female values).
    var k1: Double = 0.414

    static let identity = GaitProfile()
}

/// Solves for `GaitProfile.k`/`k1` per 劉嘉心 (2020) §3.5.7–3.5.8's two calibration scenarios.
/// Both need only a step count and elapsed time from a completed walk — independent of
/// whatever gait profile was in effect *during* that walk, since `PathPoint.stepCount` counts
/// detected steps regardless of the step-length formula used to place them.
enum GaitCalibrator {
    /// Scenario 1 (§3.5.7) — the walked distance is known. Solves both `k` and `k1`:
    /// L = D/steps, k1 = L/height, f = steps/t, k = L/(height·√f).
    static func calibrate(knownDistanceMeters distance: Double, heightCM: Double,
                          stepCount: Int, elapsedSeconds: Double) -> GaitProfile? {
        guard stepCount > 0, elapsedSeconds > 0, heightCM > 0, distance > 0 else { return nil }
        let averageStepLengthCM = (distance * 100) / Double(stepCount)
        let k1 = averageStepLengthCM / heightCM
        let frequencyHz = Double(stepCount) / elapsedSeconds
        guard frequencyHz > 0 else { return nil }
        let k = averageStepLengthCM / (heightCM * frequencyHz.squareRoot())
        return GaitProfile(heightCM: heightCM, k: k, k1: k1)
    }

    /// Scenario 2 (§3.5.8) — distance unknown, only `k` is (re)solved: `k1` keeps its current
    /// value (there's no distance to re-derive it from) and stands in for the average step
    /// length (L ≈ k1·height) to back out `k` from the observed step frequency.
    static func calibrate(heightCM: Double, currentK1: Double,
                          stepCount: Int, elapsedSeconds: Double) -> GaitProfile? {
        guard stepCount > 0, elapsedSeconds > 0, heightCM > 0 else { return nil }
        let frequencyHz = Double(stepCount) / elapsedSeconds
        guard frequencyHz > 0 else { return nil }
        let approxStepLengthCM = currentK1 * heightCM
        let k = approxStepLengthCM / (heightCM * frequencyHz.squareRoot())
        return GaitProfile(heightCM: heightCM, k: k, k1: currentK1)
    }
}

/// Real-time step detection from raw accelerometer samples, since CoreMotion has no
/// per-step callback (unlike Android's `TYPE_STEP_DETECTOR`).
///
/// Simple single-threshold peak detector: low-pass filter the raw acceleration to estimate
/// the gravity vector, subtract it to get the dynamic (user-motion) component, and fire on
/// each rising edge of that component's magnitude crossing `threshold`. `minStepInterval`
/// debounces a single footstep's up/down swing so it isn't counted twice.
///
/// This replaced a more elaborate design that followed 劉嘉心〈基於行人航位推算之室內定位
/// 研究〉(2020) §3.5.2–3.5.5 closely — a movement-gate + peak/trough-pairing + dual
/// interval/amplitude filter. That design was measured, with real ground-truth step counts
/// via the CSV export, to undercount by roughly half on two separate real walks (one
/// Holding, one Swing) — its extra machinery was rejecting far more real footsteps than
/// intended, and re-tuning its thresholds without being able to test on a real device
/// directly wasn't converging. This simpler version is what was in place before that
/// rewrite and measured as more reliable, so it's what's in use again now.
final class StepPeakDetector {
    var threshold: Double = 0.22
    var minStepInterval: TimeInterval = 0.35

    /// Samples averaged to seed the gravity baseline before step detection starts (at
    /// 50Hz this is ~0.3s). Averaging — instead of seeding from a single raw sample —
    /// avoids the baseline landing mid-stride by bad luck, which otherwise made the
    /// first steps of a session unreliable to detect (coordinates would sit frozen at
    /// the origin until the low-pass filter happened to drift back to the true gravity
    /// vector on its own).
    var warmupSampleCount = 15

    private var gravity = (x: 0.0, y: 0.0, z: 0.0)
    private var warmupSamples: [(x: Double, y: Double, z: Double)] = []
    private var isWarmedUp = false
    private var wasAboveThreshold = false
    private var lastStepTime: Date?

    var onStep: ((Date) -> Void)?

    func ingest(x: Double, y: Double, z: Double, at time: Date) {
        guard isWarmedUp else {
            warmupSamples.append((x, y, z))
            guard warmupSamples.count >= warmupSampleCount else { return }
            let n = Double(warmupSamples.count)
            gravity.x = warmupSamples.reduce(0) { $0 + $1.x } / n
            gravity.y = warmupSamples.reduce(0) { $0 + $1.y } / n
            gravity.z = warmupSamples.reduce(0) { $0 + $1.z } / n
            isWarmedUp = true
            warmupSamples.removeAll()
            return
        }
        let alpha = 0.9
        gravity.x = alpha * gravity.x + (1 - alpha) * x
        gravity.y = alpha * gravity.y + (1 - alpha) * y
        gravity.z = alpha * gravity.z + (1 - alpha) * z

        let dx = x - gravity.x, dy = y - gravity.y, dz = z - gravity.z
        let magnitude = (dx * dx + dy * dy + dz * dz).squareRoot()

        if magnitude > threshold {
            if !wasAboveThreshold, lastStepTime == nil || time.timeIntervalSince(lastStepTime!) >= minStepInterval {
                lastStepTime = time
                onStep?(time)
            }
            wasAboveThreshold = true
        } else {
            wasAboveThreshold = false
        }
    }

    func reset() {
        gravity = (0, 0, 0)
        warmupSamples.removeAll()
        isWarmedUp = false
        wasAboveThreshold = false
        lastStepTime = nil
    }
}

/// 2-state Kalman filter — state `[headingDegrees, gyroBiasDegPerSec]` — fusing gyro
/// yaw-rate (predict) with a magnetometer-derived heading (correct), so a constant gyro
/// offset doesn't leak into the heading estimate the way plain integration would.
///
/// Simplified from the full quaternion-state EKF in 王雅娜等〈基於行人航跡推算的室內定位
/// 演算法研究〉(2017), which fuses inertial sensor data on the full attitude quaternion
/// with a nonlinear observation model to improve heading accuracy — that paper's own
/// words: "EKF model is used to fuse data from inertial sensors to improve calculating
/// accuracy of heading direction," with "observation equations show non-linear
/// relationships between state vector and measurement vector."
///
/// The reduction to 2 states here is deliberate: `CMDeviceMotion` already supplies
/// Apple's own tilt-compensated fused heading (`motion.heading`), so there's no need to
/// re-derive attitude from raw magnetometer/accelerometer components the way the paper's
/// quaternion EKF does from scratch — this filter instead fuses at the heading level:
/// predict with the gyro (drifts slowly, robust to magnetic interference), correct with
/// the magnetic heading (noisy/interference-prone but doesn't drift), weighted by each
/// source's estimated reliability (the Kalman gain) — the same core idea, at lower
/// dimensionality. The angle-wrap handling in `correct(measuredHeadingDegrees:)` is the
/// nonlinearity that makes this a filter *for a circular quantity* rather than a plain
/// textbook linear KF.
final class HeadingEKF {
    private(set) var headingDegrees: Double = 0
    private(set) var gyroBiasDegPerSec: Double = 0
    private var hasInitialHeading = false

    // Covariance P = [[p00, p01], [p10, p11]] over [heading, bias].
    private var p00: Double = 100
    private var p01: Double = 0
    private var p10: Double = 0
    private var p11: Double = 10

    /// Process noise: how fast uncertainty grows between magnetometer corrections.
    /// Larger `processNoiseHeading` = trust the gyro-integrated prediction less.
    var processNoiseHeading: Double = 0.5   // deg^2 per second
    var processNoiseBias: Double = 0.002     // deg^2 per second — bias itself barely moves

    /// Measurement noise: how much to trust the magnetometer-derived heading. Turn this
    /// up in an environment with heavier magnetic interference (e.g. metal shelving).
    var measurementNoiseHeading: Double = 25 // deg^2

    /// Advances the filter by `dt` seconds using the gyro's yaw-rate reading (deg/s).
    func predict(gyroYawRateDegPerSec: Double, dt: Double) {
        guard dt > 0 else { return }
        headingDegrees = Self.normalize(headingDegrees + (gyroYawRateDegPerSec - gyroBiasDegPerSec) * dt)

        // P = F P Fᵀ + Q, with F = [[1, -dt], [0, 1]] (Jacobian of the process model).
        let newP00 = p00 - dt * p10 - dt * p01 + dt * dt * p11 + processNoiseHeading * dt
        let newP01 = p01 - dt * p11
        let newP10 = p10 - dt * p11
        let newP11 = p11 + processNoiseBias * dt
        p00 = newP00; p01 = newP01; p10 = newP10; p11 = newP11
    }

    /// Corrects using a magnetometer-derived heading measurement (already
    /// calibration-adjusted upstream by the caller).
    func correct(measuredHeadingDegrees z: Double) {
        guard hasInitialHeading else {
            headingDegrees = Self.normalize(z)
            hasInitialHeading = true
            return
        }
        // Residual wrapped to (-180, 180] — turning 359° → 1° is a 2° turn, not -358°.
        var residual = z - headingDegrees
        while residual > 180 { residual -= 360 }
        while residual <= -180 { residual += 360 }

        let s = p00 + measurementNoiseHeading // innovation covariance (H = [1, 0])
        guard s > 0 else { return }
        let k0 = p00 / s
        let k1 = p10 / s

        headingDegrees = Self.normalize(headingDegrees + k0 * residual)
        gyroBiasDegPerSec += k1 * residual

        // P = (I - K H) P
        let newP00 = (1 - k0) * p00
        let newP01 = (1 - k0) * p01
        let newP10 = p10 - k1 * p00
        let newP11 = p11 - k1 * p01
        p00 = newP00; p01 = newP01; p10 = newP10; p11 = newP11
    }

    /// Bypasses the Kalman blend entirely and snaps straight to `heading` — used when gyro
    /// fusion is turned off (see `MotionPathTracker.useGyroFusion`) so the estimate is pure
    /// magnetometer, for A/B-testing whether the gyro fusion is actually helping in a given
    /// environment.
    func setDirect(_ heading: Double) {
        headingDegrees = Self.normalize(heading)
        hasInitialHeading = true
    }

    func reset() {
        headingDegrees = 0
        gyroBiasDegPerSec = 0
        hasInitialHeading = false
        p00 = 100; p01 = 0; p10 = 0; p11 = 10
    }

    private static func normalize(_ deg: Double) -> Double {
        var d = deg.truncatingRemainder(dividingBy: 360)
        if d < 0 { d += 360 }
        return d
    }
}

/// Estimates relative movement indoors (where GPS is unreliable) by combining real-time
/// step events (from `StepPeakDetector`, reading the raw accelerometer) with a
/// gyro+magnetometer-fused heading (`HeadingEKF`), a per-user calibrated step-length model,
/// and map-based drift correction. Foreground-only by design — background/lock-screen
/// tracking was tried (CLLocationManager) and dropped in favor of accuracy, since real
/// usage always has the phone open anyway.
///
/// **Step length** uses the height/frequency model from 劉嘉心〈基於行人航位推算之室內定位
/// 研究〉(2020) §3.5.6, calibrated per-user via `GaitCalibrator`:
/// - The very first step of a tracking session: `L = k1 × height` (no prior step to time
///   a frequency against).
/// - Every step after that: `L = k × height × √(step frequency)`, frequency = 1 / (time
///   since the previous detected step).
///
/// **Heading**, off by default (`useGyroFusion = false` — see its doc comment for why),
/// fuses up to three ways when turned on for A/B testing:
/// 1. `HeadingEKF` — predicted with a gyro signal chosen by `carryingMode` (manually
///    selected in `SensorTestView`, since the automatic classifier isn't validated yet —
///    see `CarryingModeDetector`) from the same thesis §3.6 (eq. 3.20/3.21/3.23) —
///    corrected with a magnetometer-derived heading.
/// 2. A magnetometer hard-iron/soft-iron calibration (see MagnetometerCalibrator) —
///    **known broken for this app's held-upright posture, see `useCalibrationCorrection`'s
///    doc comment; leave off.**
/// 3. A single-sample outlier gate: a magnetic-heading reading that implies too large a
///    jump is not fed to the EKF's correction step at all that sample (the filter just
///    coasts on the gyro prediction), so one bad reading (e.g. walking past a speaker)
///    can't yank the estimate on its own.
///
/// With gyro fusion off (the default), heading is simply Apple's own `motion.heading` —
/// every hand-rolled attempt to improve on it (gyro formulas, this calibration) has measured
/// worse on real walks so far; see each one's doc comment for specifics.
///
/// Position-side drift (this class doesn't fix it alone) is instead corrected upstream by
/// `NavigationSessionManager`, which snaps the running position back toward an
/// already-mapped node whenever the user revisits one — see `applyDriftCorrection(dx:dy:)`.
final class MotionPathTracker {
    private let motionManager = CMMotionManager()
    private let stepDetector = StepPeakDetector()
    private let headingEKF = HeadingEKF()
    private let carryingModeDetector = CarryingModeDetector()

    private(set) var isTracking = false
    private var currentPosition = (x: 0.0, y: 0.0)
    private var lastMotionSampleTime: Date?
    private var cumulativeStepDistance: Double = 0
    private var detectedStepCount = 0
    private var lastStepTimeForFrequency: Date?
    private var smoothedOmegaX: Double = 0
    private var smoothedOmegaY: Double = 0
    private var smoothedOmegaZ: Double = 0
    private var hasGyroBaseline = false

    /// Manually declared carrying posture (see `CarryingMode`) — picked by the user in
    /// `SensorTestView` rather than relying on `CarryingModeDetector`'s automatic
    /// classification, since that classifier's exact thresholds are this project's own
    /// best-effort derivation (the paper's real cut points aren't recoverable from the
    /// published text) and hasn't been validated as reliable enough to drive the formula
    /// selection on its own yet. Selects which of 劉嘉心 (2020) §3.6's gyro-signal formulas
    /// `ingestMotion` uses (Holding/Pocket/Swing — Calling was dropped, see `CarryingMode`).
    var carryingMode: CarryingMode = .holding

    // Latest raw gyro/magnetometer readings, cached from the 5Hz device-motion callback so
    // `emitRawSampleIfDue` (driven by the 50Hz accelerometer callback) can attach them to a
    // combined sample without waiting on its own device-motion tick.
    private var latestGyroX: Double = 0
    private var latestGyroY: Double = 0
    private var latestGyroZ: Double = 0
    private var latestMagX: Double = 0
    private var latestMagY: Double = 0
    private var latestMagZ: Double = 0
    private var lastRawSampleTime: Date?

    /// How often `onRawSample` fires — throttled well below the 50Hz accelerometer rate
    /// since this is for offline CSV analysis, not real-time control; ~10Hz matches what the
    /// Android teammate's SensorLab logger uses.
    var rawSampleInterval: TimeInterval = 0.1

    /// Smoothing applied to ωy/ωz before combining into the Holding-mode signal (see
    /// `ingestMotion`) — needed because √(ωy² + ωz²) is a *rectified* (always ≥ 0) quantity,
    /// so unlike a signed single-axis rate it does not average toward zero under gyro noise.
    /// Every footstep's impact jitters the phone slightly on both axes; without smoothing,
    /// that jitter gets integrated as a near-step-periodic phantom turn (observed on real
    /// walk tests: 35-45° of heading swing *per step* on a straight track). Higher = more
    /// smoothing/lag; lower = trusts the raw (noisier) reading more.
    ///
    /// 0.75 (≈0.8s time constant at 5Hz) turned out too aggressive in practice — a real turn
    /// while walking usually completes in well under a second, so that much lag flattens the
    /// turn's peak rate and under-rotates the heading estimate, which is why a full lap ended
    /// up not closing back near the start. 0.4 (≈0.33s) still knocks down single-sample
    /// footstep jitter (one 0.2s tick) without eating as much of a real turn.
    var gyroSmoothingAlpha: Double = 0.4

    /// Below this (deg/s), the smoothed Holding-mode signal is treated as "not turning" and
    /// contributes nothing to the heading predict step — otherwise residual sensor noise,
    /// being rectified into an always-positive magnitude, would still slowly accumulate.
    /// Lowered from 12 for the same reason as `gyroSmoothingAlpha`: it was cutting into the
    /// slower tail end of real turns, not just noise.
    var gyroDeadBandDegPerSec: Double = 5

    private var calibration: MagnetometerCalibration
    private var gyroBias: GyroBias
    private(set) var gaitProfile: GaitProfile

    /// Off by default — this needs a manual rotate-the-phone calibration gesture
    /// (MagnetometerCalibrator) that real end users won't run.
    ///
    /// **Known broken, confirmed harmful on real walk tests — do not enable.**
    /// `calibratedHeading(appleHeading:field:)` computes `atan2(field.y, field.x)`, a "flat
    /// compass" formula only valid when the phone lies flat (screen up). This app holds the
    /// phone upright (portrait, camera forward), so that atan2 isn't tracking heading at
    /// all — it's picking up however the phone tilts while walking. The delta it produces
    /// isn't random noise (which the delta design was meant to be safe against); it's a real
    /// signal correlated with walking motion, so adding it to Apple's already-correct
    /// `motion.heading` partially cancels out genuine turns instead of correcting bias — an
    /// on-device test (2 loops walked) came back with almost no turning recorded at all after
    /// enabling this. Fixing it properly needs a tilt-independent formula (e.g. cross product
    /// of the magnetic field and gravity vectors, both already available from
    /// `CMDeviceMotion`), which hasn't been implemented/validated — this is the third
    /// hand-rolled heading formula in this class to turn out wrong on real hardware (see also
    /// the Holding-mode gyro signal's rectification bug and the gyro-fusion tuning that also
    /// underperformed raw `motion.heading` on every test run so far), so it's left disabled
    /// rather than risk a fourth unverified rewrite.
    var useCalibrationCorrection = false

    /// Reject a single-sample heading jump larger than this (degrees) — the EKF's
    /// correction step is skipped for that sample (predict-only) instead of feeding it
    /// a measurement that would yank the estimate off course.
    var maxPlausibleJumpDegrees: Double = 100

    /// Forwarded tuning knobs for `StepPeakDetector` — see its doc comment. Adjust via
    /// the on-device harness in `SensorTestView` if step counting is off for a given
    /// phone/carry position.
    var stepDetectionThreshold: Double {
        get { stepDetector.threshold }
        set { stepDetector.threshold = newValue }
    }
    var minStepInterval: TimeInterval {
        get { stepDetector.minStepInterval }
        set { stepDetector.minStepInterval = newValue }
    }

    /// When false, skips the gyro predict/Kalman-blend entirely and uses the raw
    /// calibration-adjusted magnetometer heading directly every tick — lets you A/B test
    /// whether the gyro fusion is actually net-helping in a given environment (e.g. outdoors
    /// with little magnetic interference) versus just adding tuning risk for no benefit.
    ///
    /// Defaults to `false`: every on-device test so far (outdoor track and two indoor walks)
    /// came out worse with fusion on than with the raw magnetometer alone. Hypothesis: when
    /// indoor interference pushes a reading past `maxPlausibleJumpDegrees`, the filter is left
    /// coasting on the gyro prediction alone for stretches at a time, and that prediction
    /// isn't reliable enough yet to be trusted uncorrected for that long — so it accumulates
    /// more error than just taking the noisy-but-uncorrelated raw magnetometer reading each
    /// tick would. Flip on to re-test if the gyro path gets revisited.
    var useGyroFusion = false

    var onPathPoint: ((PathPoint) -> Void)?
    /// Fires every device-motion tick (~5Hz) with the current heading estimate, independent
    /// of stepping — unlike `onPathPoint`, this updates continuously even while standing
    /// still, so a live UI can show whether heading is actually tracking a turn in real time
    /// instead of only refreshing once the next step is taken.
    var onHeadingUpdate: ((Double) -> Void)?
    /// Fires at `rawSampleInterval` with every raw sensor input combined into one row — see
    /// `RawSensorSample`'s doc comment for why this exists.
    var onRawSample: ((RawSensorSample) -> Void)?
    /// Fires every device-motion tick with the live carrying-mode classification — see
    /// `CarryingModeDetector`. Not yet wired into the step-length/heading formulas; this is
    /// stage-1 validation only (confirm the classifier itself behaves before building on it).
    var onCarryingModeUpdate: ((CarryingMode, _ degTx: Double, _ degP: Double) -> Void)?

    /// A motion coprocessor being present (the same signal `CMPedometer` availability has
    /// always implied) is used here purely as a "real hardware, not Simulator" gate — actual
    /// step/distance data no longer comes from `CMPedometer` (see class doc comment).
    static var isAvailable: Bool {
        CMPedometer.isStepCountingAvailable()
    }

    init() {
        calibration = PersistenceService.shared.load(MagnetometerCalibration.self, from: "magnetometer_calibration.json") ?? .identity
        gyroBias = PersistenceService.shared.load(GyroBias.self, from: "gyro_bias.json") ?? .identity
        gaitProfile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
        stepDetector.onStep = { [weak self] time in
            self?.handleStepDetected(at: time)
        }
    }

    /// Call after MagnetometerCalibrator finishes, to use the new calibration
    /// immediately without needing to restart tracking.
    func reloadCalibration() {
        calibration = PersistenceService.shared.load(MagnetometerCalibration.self, from: "magnetometer_calibration.json") ?? .identity
    }

    /// Call after `GyroCalibrator` finishes, to use the new bias immediately without needing
    /// to restart tracking.
    func reloadGyroBias() {
        gyroBias = PersistenceService.shared.load(GyroBias.self, from: "gyro_bias.json") ?? .identity
    }

    /// Call after `GaitCalibrator` produces a new profile, to use it immediately without
    /// needing to restart tracking.
    func reloadGaitProfile() {
        gaitProfile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
    }

    func start() {
        guard !isTracking, MotionPathTracker.isAvailable else { return }
        isTracking = true
        currentPosition = (0, 0)
        lastMotionSampleTime = nil
        cumulativeStepDistance = 0
        detectedStepCount = 0
        lastStepTimeForFrequency = nil
        smoothedOmegaX = 0
        smoothedOmegaY = 0
        smoothedOmegaZ = 0
        hasGyroBaseline = false
        latestGyroX = 0; latestGyroY = 0; latestGyroZ = 0
        latestMagX = 0; latestMagY = 0; latestMagZ = 0
        lastRawSampleTime = nil
        stepDetector.reset()
        headingEKF.reset()
        carryingModeDetector.reset()

        // Record the true starting point explicitly — every subsequent PathPoint already
        // reflects position *after* that step's displacement, so without this the first
        // recorded point is already one step away from (0,0), leaving a visible gap between
        // the plotted origin and the start of the path.
        onPathPoint?(PathPoint(x: 0, y: 0, headingDegrees: 0, timestamp: Date().timeIntervalSince1970 * 1000,
                               stepCount: 0, distanceMeters: 0))

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.2
            motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: .main) { [weak self] motion, _ in
                guard let self, let motion, motion.heading >= 0 else { return }
                self.ingestMotion(motion)
            }
        }

        if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = 0.02 // 50Hz — needed to resolve a step's peak
            motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, _ in
                guard let self, let data else { return }
                let now = Date()
                self.stepDetector.ingest(x: data.acceleration.x, y: data.acceleration.y, z: data.acceleration.z, at: now)
                self.emitRawSampleIfDue(accel: data.acceleration, at: now)
            }
        }
    }

    private func emitRawSampleIfDue(accel: CMAcceleration, at time: Date) {
        if let last = lastRawSampleTime, time.timeIntervalSince(last) < rawSampleInterval { return }
        lastRawSampleTime = time
        onRawSample?(RawSensorSample(
            timestamp: time.timeIntervalSince1970 * 1000,
            accelX: accel.x, accelY: accel.y, accelZ: accel.z,
            gyroX: latestGyroX, gyroY: latestGyroY, gyroZ: latestGyroZ,
            magX: latestMagX, magY: latestMagY, magZ: latestMagZ,
            headingDegrees: headingEKF.headingDegrees,
            stepCount: detectedStepCount
        ))
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopAccelerometerUpdates()
    }

    /// One-off nudge to the running position estimate — called by NavigationSessionManager
    /// when TopologicalMap recognizes the current point as a revisit of an already-mapped
    /// node, pulling accumulated drift back toward that trusted anchor instead of letting
    /// it grow unbounded for the rest of the session.
    func applyDriftCorrection(dx: Double, dy: Double) {
        currentPosition.x += dx
        currentPosition.y += dy
    }

    /// Predicts the EKF with the Holding-mode gyro signal, then (unless the magnetic
    /// reading looks like an outlier) corrects it with the calibration-adjusted magnetic
    /// heading.
    private func ingestMotion(_ motion: CMDeviceMotion) {
        let now = Date()
        let dt = lastMotionSampleTime.map { now.timeIntervalSince($0) } ?? 0
        lastMotionSampleTime = now

        // Bias-corrected once here so every downstream reader (raw sample log, the
        // Holding/Pocket/Swing formulas below) sees the same corrected values instead of
        // each having to remember to subtract `gyroBias` itself.
        let correctedGyro = gyroBias.apply(x: motion.rotationRate.x, y: motion.rotationRate.y, z: motion.rotationRate.z)
        latestGyroX = correctedGyro.x
        latestGyroY = correctedGyro.y
        latestGyroZ = correctedGyro.z
        latestMagX = motion.magneticField.field.x
        latestMagY = motion.magneticField.field.y
        latestMagZ = motion.magneticField.field.z

        carryingModeDetector.ingest(gravity: (motion.gravity.x, motion.gravity.y, motion.gravity.z))
        onCarryingModeUpdate?(carryingModeDetector.mode, carryingModeDetector.degTx, carryingModeDetector.degP)

        let measured = calibratedHeading(appleHeading: motion.heading, field: motion.magneticField.field)

        if useGyroFusion {
            // Per-carrying-mode gyro signal, 劉嘉心 (2020) §3.6 eq. (3.20)-(3.23) — each
            // combines whichever gyro axes are most sensitive to a turn for that posture
            // (`carryingMode`, manually selected in `SensorTestView` — see its doc comment
            // for why this isn't driven by the automatic classifier).
            //
            // ωx/ωy/ωz are smoothed first because every one of these formulas is a
            // *rectified* (√ of squares, always ≥ 0) quantity — unlike a signed single-axis
            // rate, sensor noise on it doesn't average toward zero when integrated, so raw
            // per-footstep jitter would accumulate into a phantom turn every step (see
            // `gyroSmoothingAlpha`'s doc comment). A low-pass on the inputs suppresses that
            // jitter while still tracking a genuine turn.
            if !hasGyroBaseline {
                smoothedOmegaX = correctedGyro.x
                smoothedOmegaY = correctedGyro.y
                smoothedOmegaZ = correctedGyro.z
                hasGyroBaseline = true
            } else {
                smoothedOmegaX = gyroSmoothingAlpha * smoothedOmegaX + (1 - gyroSmoothingAlpha) * correctedGyro.x
                smoothedOmegaY = gyroSmoothingAlpha * smoothedOmegaY + (1 - gyroSmoothingAlpha) * correctedGyro.y
                smoothedOmegaZ = gyroSmoothingAlpha * smoothedOmegaZ + (1 - gyroSmoothingAlpha) * correctedGyro.z
            }

            let omegaRadPerSec: Double
            switch carryingMode {
            case .holding, .unknown:
                // eq. (3.20): Ω_hold = sgn(ωz)·√(ωy²+ωz²)
                let sign = smoothedOmegaZ >= 0 ? 1.0 : -1.0
                omegaRadPerSec = sign * (smoothedOmegaY * smoothedOmegaY + smoothedOmegaZ * smoothedOmegaZ).squareRoot()
            case .pocket:
                // eq. (3.21): Ω_pocket = sgn(ωy)·√(ωy²+ωz²)
                let sign = smoothedOmegaY >= 0 ? 1.0 : -1.0
                omegaRadPerSec = sign * (smoothedOmegaY * smoothedOmegaY + smoothedOmegaZ * smoothedOmegaZ).squareRoot()
            case .swingRight, .swingLeft:
                // eq. (3.23): Ω_swing = sgn(ωx)·√(ωx²+ωy²) — same formula for both sides.
                let sign = smoothedOmegaX >= 0 ? 1.0 : -1.0
                omegaRadPerSec = sign * (smoothedOmegaX * smoothedOmegaX + smoothedOmegaY * smoothedOmegaY).squareRoot()
            }
            var gyroYawRateDegPerSec = omegaRadPerSec * 180 / .pi
            // Dead-band: residual noise is still a rectified (always-positive-magnitude)
            // signal, so even after smoothing it won't average to zero on its own — below
            // this threshold treat it as "not turning" rather than letting it accumulate.
            if abs(gyroYawRateDegPerSec) < gyroDeadBandDegPerSec {
                gyroYawRateDegPerSec = 0
            }

            if dt > 0 {
                headingEKF.predict(gyroYawRateDegPerSec: gyroYawRateDegPerSec, dt: dt)
            }

            var residual = measured - headingEKF.headingDegrees
            while residual > 180 { residual -= 360 }
            while residual <= -180 { residual += 360 }

            if abs(residual) <= maxPlausibleJumpDegrees {
                headingEKF.correct(measuredHeadingDegrees: measured)
            }
            // Else: treat as a magnetic-interference spike — skip correction this sample and
            // let the filter coast on the gyro prediction; a genuine fast turn will just get
            // accepted on the next sample or two.
        } else {
            // Gyro fusion disabled — pure magnetometer heading, no Kalman blend at all. See
            // `useGyroFusion`'s doc comment.
            headingEKF.setDirect(measured)
        }

        onHeadingUpdate?(headingEKF.headingDegrees)
    }

    private func calibratedHeading(appleHeading: Double, field: CMMagneticField) -> Double {
        guard useCalibrationCorrection, calibration.scaleX != 1 || calibration.scaleY != 1 || calibration.scaleZ != 1 else {
            return appleHeading
        }
        let raw = atan2(field.y, field.x) * 180 / .pi
        let cal = calibration.apply(x: field.x, y: field.y, z: field.z)
        let calibrated = atan2(cal.y, cal.x) * 180 / .pi
        var delta = calibrated - raw
        // Normalize the delta into (-180, 180] before applying it.
        while delta > 180 { delta -= 360 }
        while delta <= -180 { delta += 360 }
        return appleHeading + delta
    }

    /// §3.5.6: `L = k1×height` for the first step after standing still, `L = k×height×√f`
    /// for every step after that (see `GaitProfile`/`GaitCalibrator`).
    /// §3.5.6: `L = k1×height` for the very first step of a session (no prior step to time
    /// against), `L = k×height×√f` for every step after that, `f` being 1/(time since the
    /// previous detected step).
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

        let heading = headingEKF.headingDegrees
        let headingRadians = heading * .pi / 180
        currentPosition.x += stepLength * sin(headingRadians)
        currentPosition.y += stepLength * cos(headingRadians)
        cumulativeStepDistance += stepLength

        onPathPoint?(PathPoint(
            x: currentPosition.x,
            y: currentPosition.y,
            headingDegrees: heading,
            timestamp: time.timeIntervalSince1970 * 1000,
            stepCount: detectedStepCount,
            distanceMeters: cumulativeStepDistance
        ))
    }
}
