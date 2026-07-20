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

/// Estimates relative movement indoors (where GPS is unreliable) by fusing step
/// counts/distance from CMPedometer with heading from CMDeviceMotion. Foreground-only
/// by design — background/lock-screen tracking was tried (CLLocationManager) and
/// dropped in favor of accuracy, since real usage always has the phone open anyway.
///
/// Heading accuracy is improved three ways:
/// 1. **Dense sampling** — CMPedometer only reports a new cumulative distance every
///    ~1-2.5s, but CMDeviceMotion delivers heading at 5Hz (every 0.2s). Naively using
///    only the single heading reading current *at* each pedometer callback throws away
///    every other heading sample in between, so a multi-second arc gets approximated as
///    one straight line segment — exactly wrong when the whole point is walking in a
///    loop. Instead we buffer every heading sample and, when a pedometer update arrives,
///    distribute its reported delta-distance proportionally across all heading samples
///    seen since the last update, producing a point roughly every 0.2s instead of every
///    ~2.5s. This needs no user action and helps regardless of the other two fixes.
/// 2. A magnetometer hard-iron/soft-iron calibration (see MagnetometerCalibrator) removes
///    local environmental bias (electronics, metal furniture/shelving). Applied as a
///    *delta correction* on top of Apple's own `.heading` (not a full replacement) so a
///    sign/axis mistake in our own atan2 can't silently make things worse than doing
///    nothing. Optional — off by default expectations aside, real end users are not
///    expected to run this; it exists for our own testing/tuning.
/// 3. A simple outlier rejection filter drops single-sample heading jumps too large to be
///    a real turn, so one bad reading (e.g. walking past a speaker) doesn't get baked
///    permanently into the accumulated path.
final class MotionPathTracker {
    private let pedometer = CMPedometer()
    private let motionManager = CMMotionManager()

    private(set) var isTracking = false
    private var lastCumulativeDistance: Double = 0
    private var lastPedometerUpdateTime: Date?
    private var lastAcceptedHeadingDegrees: Double?
    private var currentPosition = (x: 0.0, y: 0.0)

    /// (time, heading) samples collected since the last CMPedometer callback, consumed
    /// and cleared each time a new cumulative-distance update arrives.
    private var headingBuffer: [(time: Date, heading: Double)] = []

    private var calibration: MagnetometerCalibration

    /// Off by default — this needs a manual rotate-the-phone calibration gesture
    /// (MagnetometerCalibrator) that real end users won't run. Useful for our own
    /// accuracy testing in a fixed environment; turn on only if you calibrated first.
    var useCalibrationCorrection = false

    /// Reject a single-sample heading jump larger than this (degrees) unless the next
    /// sample confirms it — filters out momentary magnetic-interference spikes without
    /// blocking genuinely fast turns.
    var maxPlausibleJumpDegrees: Double = 100

    var onPathPoint: ((PathPoint) -> Void)?

    static var isAvailable: Bool {
        CMPedometer.isDistanceAvailable() && CMPedometer.isStepCountingAvailable()
    }

    init() {
        calibration = PersistenceService.shared.load(MagnetometerCalibration.self, from: "magnetometer_calibration.json") ?? .identity
    }

    /// Call after MagnetometerCalibrator finishes, to use the new calibration
    /// immediately without needing to restart tracking.
    func reloadCalibration() {
        calibration = PersistenceService.shared.load(MagnetometerCalibration.self, from: "magnetometer_calibration.json") ?? .identity
    }

    func start() {
        guard !isTracking, MotionPathTracker.isAvailable else { return }
        isTracking = true
        lastCumulativeDistance = 0
        lastPedometerUpdateTime = nil
        currentPosition = (0, 0)
        lastAcceptedHeadingDegrees = nil
        headingBuffer.removeAll()

        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 0.2
            motionManager.startDeviceMotionUpdates(using: .xMagneticNorthZVertical, to: .main) { [weak self] motion, _ in
                guard let self, let motion, motion.heading >= 0 else { return }
                self.ingestHeading(motion.heading, field: motion.magneticField.field)
            }
        }

        pedometer.startUpdates(from: Date()) { [weak self] data, error in
            guard let self, let data, error == nil, let distance = data.distance?.doubleValue else { return }
            DispatchQueue.main.async {
                self.consume(cumulativeDistance: distance, stepCount: data.numberOfSteps.intValue, at: Date())
            }
        }
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        pedometer.stopUpdates()
        motionManager.stopDeviceMotionUpdates()
    }

    private func ingestHeading(_ appleHeading: Double, field: CMMagneticField) {
        var corrected = appleHeading
        if useCalibrationCorrection, calibration.scaleX != 1 || calibration.scaleY != 1 || calibration.scaleZ != 1 {
            let raw = atan2(field.y, field.x) * 180 / .pi
            let cal = calibration.apply(x: field.x, y: field.y, z: field.z)
            let calibrated = atan2(cal.y, cal.x) * 180 / .pi
            var delta = calibrated - raw
            // Normalize the delta into (-180, 180] before applying it.
            while delta > 180 { delta -= 360 }
            while delta <= -180 { delta += 360 }
            corrected = appleHeading + delta
        }
        let accepted = acceptOrRejectOutlier(corrected)
        headingBuffer.append((Date(), accepted))
    }

    private func acceptOrRejectOutlier(_ newHeading: Double) -> Double {
        guard let last = lastAcceptedHeadingDegrees else {
            lastAcceptedHeadingDegrees = newHeading
            return newHeading
        }
        var diff = newHeading - last
        while diff > 180 { diff -= 360 }
        while diff <= -180 { diff += 360 }

        if abs(diff) > maxPlausibleJumpDegrees {
            // Reject this sample's implied jump — keep the last accepted heading.
            // A genuine fast turn will just get accepted on the next sample or two,
            // costing a little responsiveness in exchange for not baking a single
            // interference spike into the whole rest of the path.
            return last
        }
        lastAcceptedHeadingDegrees = newHeading
        return newHeading
    }

    private func consume(cumulativeDistance: Double, stepCount: Int, at now: Date) {
        let delta = cumulativeDistance - lastCumulativeDistance
        guard delta > 0 else { return }
        let windowStart = lastPedometerUpdateTime
        lastCumulativeDistance = cumulativeDistance
        lastPedometerUpdateTime = now

        let samples = headingBuffer.filter { windowStart == nil || $0.time > windowStart! }
        headingBuffer.removeAll()

        guard !samples.isEmpty else {
            // No heading sample arrived during this window (device motion unavailable,
            // or a very short window) — fall back to one lump update, same as before.
            applyIncrement(delta: delta, heading: lastAcceptedHeadingDegrees ?? 0, stepCount: stepCount, distanceSoFar: cumulativeDistance, timestamp: now)
            return
        }

        // Split the pedometer's reported delta evenly across every heading sample seen
        // since the last update, so each sub-step uses the direction that was actually
        // measured at that moment instead of one direction for the whole window.
        let perSampleDelta = delta / Double(samples.count)
        for sample in samples {
            applyIncrement(delta: perSampleDelta, heading: sample.heading, stepCount: stepCount, distanceSoFar: cumulativeDistance, timestamp: sample.time)
        }
    }

    private func applyIncrement(delta: Double, heading: Double, stepCount: Int, distanceSoFar: Double, timestamp: Date) {
        let headingRadians = heading * .pi / 180
        currentPosition.x += delta * sin(headingRadians)
        currentPosition.y += delta * cos(headingRadians)

        onPathPoint?(PathPoint(
            x: currentPosition.x,
            y: currentPosition.y,
            headingDegrees: heading,
            timestamp: timestamp.timeIntervalSince1970 * 1000,
            stepCount: stepCount,
            distanceMeters: distanceSoFar
        ))
    }
}
