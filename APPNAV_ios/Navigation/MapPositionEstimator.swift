import Foundation

/// The user's position on the store map, kept by dead reckoning and corrected by photo fixes.
///
/// One instance per navigation session (owned by `NavigationSessionManager`) is the single
/// source of truth for "where am I on the map" — the route follower, the AR arrow, the
/// on-device DINOv2 fusion and the server upload all read it.
///
/// Frames: the raw PDR from `MotionPathTracker` starts at (0,0) wherever tracking began;
/// the map frame is the reference map's node coordinates (metres, +y = heading 0°).
/// Until the first confident photo fix the map position is unknown (`mapPosition == nil`).
///
/// What a photo fix corrects:
/// - **position** — snapped to the localized node (cancels accumulated drift);
/// - **heading offset** — between two fixes ≥ `minCalibrationDistanceM` apart, the
///   direction actually walked (node A → node B) vs the direction PDR thought it walked;
/// - **step scale** — map distance A→B vs PDR distance over the same stretch, clamped
///   to `stepScaleRange` so one bad fix cannot wreck the step model.
/// Both are smoothed with `learningRate` and reset only with the session.
///
/// Fixes below `minConfidence` (`minFirstFixConfidence` before any position), or implausible jumps (> `maxJumpM` from the dead-reckoned
/// position while confidence < `jumpOverrideConfidence`), are rejected.
///
/// Pure logic, no sensors / UI — unit tested in scripts/swift_logic_tests.
final class MapPositionEstimator {
    struct Config {
        var minConfidence = 0.6
        /// The very first fix has no sensor position to agree with, so the server's
        /// confidence is image-only and lower (backend loc_fusion.fuse_v2 posterior).
        var minFirstFixConfidence = 0.3
        var maxJumpM = 15.0
        var jumpOverrideConfidence = 0.85
        var minCalibrationDistanceM = 8.0
        var learningRate = 0.3
        var maxHeadingCorrectionDeg = 45.0
        var stepScaleRange: ClosedRange<Double> = 0.7...1.3
    }

    enum FixResult: Equatable {
        case accepted
        case rejectedLowConfidence
        case rejectedJump(distanceM: Double)
    }

    let config: Config

    private(set) var mapPosition: (x: Double, y: Double)?
    /// Added to the raw PDR heading (degrees, clockwise).
    private(set) var headingOffsetDeg = 0.0
    /// Multiplies raw PDR distances.
    private(set) var stepScale = 1.0
    /// Distance walked (map metres) since the last accepted fix — drives the server's
    /// sensor weight (backend loc_fusion: σ = σ0 + drift_rate · distance).
    private(set) var distanceSinceFixM = 0.0
    private(set) var acceptedFixes = 0

    private var lastRaw: (x: Double, y: Double)?
    /// Raw PDR distance walked since the last fix (for the step-scale ratio).
    private var rawDistanceSinceFix = 0.0
    private var lastFix: (map: (x: Double, y: Double), raw: (x: Double, y: Double))?

    init(config: Config = Config()) {
        self.config = config
    }

    var isCalibrated: Bool { mapPosition != nil }

    func reset() {
        mapPosition = nil
        headingOffsetDeg = 0
        stepScale = 1
        distanceSinceFixM = 0
        acceptedFixes = 0
        lastRaw = nil
        rawDistanceSinceFix = 0
        lastFix = nil
    }

    /// A new raw PDR sample. Moves the map position by the corrected step.
    func ingest(raw: (x: Double, y: Double)) {
        defer { lastRaw = raw }
        guard let prev = lastRaw else { return }
        let dx = raw.x - prev.x, dy = raw.y - prev.y
        let d = (dx * dx + dy * dy).squareRoot()
        guard d > 0 else { return }
        rawDistanceSinceFix += d
        guard let pos = mapPosition else { return }
        let step = Self.rotate((dx, dy), byDegrees: headingOffsetDeg)
        mapPosition = (pos.x + step.x * stepScale, pos.y + step.y * stepScale)
        distanceSinceFixM += d * stepScale
    }

    /// The raw PDR frame jumped without the user moving (e.g. the tracker's own
    /// revisit drift correction): continue from `raw` without integrating the jump.
    func rebase(raw: (x: Double, y: Double)) {
        lastRaw = raw
    }

    /// The server localized a photo to `node` (map frame) with `confidence`.
    /// `raw` is the PDR reading when the photo was taken.
    @discardableResult
    func applyFix(node: (x: Double, y: Double), confidence: Double,
                  raw: (x: Double, y: Double)) -> FixResult {
        let needed = mapPosition == nil ? config.minFirstFixConfidence : config.minConfidence
        guard confidence >= needed else { return .rejectedLowConfidence }
        if let pos = mapPosition {
            let jump = Self.distance(pos, node)
            if jump > config.maxJumpM && confidence < config.jumpOverrideConfidence {
                return .rejectedJump(distanceM: jump)
            }
        }

        if let last = lastFix {
            let mapVec = (x: node.x - last.map.x, y: node.y - last.map.y)
            let rawVec = (x: raw.x - last.raw.x, y: raw.y - last.raw.y)
            let mapDist = Self.length(mapVec), rawDist = Self.length(rawVec)
            // Only a long, fairly straight stretch says anything about heading / step
            // length: the node snap error (~1–2 m) must be small next to the distance,
            // and the PDR path must not have wandered (raw path length ≈ straight distance).
            let straight = rawDistanceSinceFix > 0 && rawDist / rawDistanceSinceFix > 0.8
            if mapDist >= config.minCalibrationDistanceM && rawDist >= config.minCalibrationDistanceM
                && straight {
                let observed = Self.wrap(Self.bearing(mapVec) - Self.bearing(rawVec))
                if abs(Self.wrap(observed - headingOffsetDeg)) <= config.maxHeadingCorrectionDeg {
                    headingOffsetDeg = Self.wrap(headingOffsetDeg
                        + config.learningRate * Self.wrap(observed - headingOffsetDeg))
                }
                let ratio = min(max(mapDist / rawDist, config.stepScaleRange.lowerBound),
                                config.stepScaleRange.upperBound)
                stepScale += config.learningRate * (ratio - stepScale)
                stepScale = min(max(stepScale, config.stepScaleRange.lowerBound),
                                config.stepScaleRange.upperBound)
            }
        }

        mapPosition = node
        lastFix = (node, raw)
        lastRaw = raw
        distanceSinceFixM = 0
        rawDistanceSinceFix = 0
        acceptedFixes += 1
        return .accepted
    }

    /// Raw PDR heading → map heading.
    func mapHeading(fromRaw rawHeadingDeg: Double) -> Double {
        var h = (rawHeadingDeg + headingOffsetDeg).truncatingRemainder(dividingBy: 360)
        if h < 0 { h += 360 }
        return h
    }

    // MARK: - Geometry (heading 0° = +y, clockwise, like MotionPathTracker)

    static func bearing(_ v: (x: Double, y: Double)) -> Double {
        atan2(v.x, v.y) * 180 / .pi
    }

    /// Rotate a vector clockwise by `deg` (so a heading h becomes h + deg).
    static func rotate(_ v: (x: Double, y: Double), byDegrees deg: Double) -> (x: Double, y: Double) {
        let r = deg * .pi / 180
        return (v.x * cos(r) + v.y * sin(r), -v.x * sin(r) + v.y * cos(r))
    }

    static func wrap(_ deg: Double) -> Double {
        var d = deg.truncatingRemainder(dividingBy: 360)
        if d > 180 { d -= 360 }
        if d <= -180 { d += 360 }
        return d
    }

    static func length(_ v: (x: Double, y: Double)) -> Double { (v.x * v.x + v.y * v.y).squareRoot() }

    static func distance(_ a: (x: Double, y: Double), _ b: (x: Double, y: Double)) -> Double {
        length((b.x - a.x, b.y - a.y))
    }
}
