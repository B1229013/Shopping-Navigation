import CoreGraphics
import Foundation

/// Walks a planned route on the store map using the phone's dead-reckoned position.
///
/// The route is a polyline in map metres (origin = entrance, +y = heading 0°, the same
/// frame `MotionPathTracker` produces). Dead reckoning only gives *relative* motion, so
/// the follower keeps an anchor: the map point the user was last known to be at (route
/// start, or the waypoint a photo was localized to) paired with the PDR reading at that
/// moment. Current map position = anchor + (PDR now − PDR at anchor). Each photo the
/// server localizes re-anchors and cancels accumulated drift.
///
/// Pure logic, no UI or sensors: `update(pdr:)` returns everything the screen needs.
struct FollowState: Equatable {
    var position: CGPoint            // map metres
    var headingDegrees: Double
    var progressM: Double            // distance walked along the route
    var remainingM: Double           // distance still to go
    var offRouteM: Double            // perpendicular distance from the route
    var nextTurnDirection: String?   // e.g. "right"; nil when only straight remains
    var nextTurnInM: Double?         // distance until that turn
    var prompt: String               // what to tell the user right now
    var needsReplan: Bool            // drifted too far from the route
    var arrived: Bool                // within reach of the target
}

final class PathFollower {
    /// Beyond this cross-track distance the route no longer describes where the user is.
    static let offRouteThresholdM = 6.0
    /// Within this distance of the end the target counts as reached.
    static let arrivalRadiusM = 3.0
    /// A turn is "now" inside this distance, "soon" up to `announceTurnM`.
    static let turnNowM = 3.0
    static let announceTurnM = 12.0

    private(set) var polyline: [CGPoint] = []
    private var cumulative: [Double] = []          // route distance at each vertex
    private var turnsAlongRoute: [(atM: Double, direction: String)] = []
    private var segmentIndex = 0

    private var anchorMap = CGPoint.zero
    private var anchorPDR = CGPoint.zero
    private(set) var hasRoute = false

    var totalLengthM: Double { cumulative.last ?? 0 }

    // MARK: - Setup

    /// Install a route. `turns` are (map point where the step starts, direction).
    /// The user is assumed to be at the first vertex unless `keepAnchor` is set
    /// (a mid-route re-plan from the phone's own position keeps the current anchor).
    func setRoute(polyline: [CGPoint], turns: [(at: CGPoint, direction: String)],
                  pdr: CGPoint, keepAnchor: Bool = false) {
        self.polyline = polyline
        cumulative = []
        var acc = 0.0
        for (i, p) in polyline.enumerated() {
            if i > 0 { acc += Self.distance(polyline[i - 1], p) }
            cumulative.append(acc)
        }
        turnsAlongRoute = turns
            .filter { $0.direction != "straight" }
            .map { turn -> (atM: Double, direction: String) in
                // a turn starts at a vertex; find that vertex's route distance
                let idx = Self.nearestVertexIndex(to: turn.at, in: polyline)
                return (cumulative[idx], turn.direction)
            }
            .sorted { $0.atM < $1.atM }
        segmentIndex = 0
        hasRoute = polyline.count >= 1
        if !keepAnchor, let first = polyline.first {
            anchor(mapPoint: first, pdr: pdr)
        }
    }

    /// Pin the user's map position (e.g. the waypoint a photo was localized to) to the
    /// PDR reading at this moment.
    func anchor(mapPoint: CGPoint, pdr: CGPoint) {
        anchorMap = mapPoint
        anchorPDR = pdr
        segmentIndex = 0
    }

    // MARK: - Per-step update

    func update(pdr: CGPoint, headingDegrees: Double) -> FollowState? {
        guard hasRoute else { return nil }
        let pos = CGPoint(x: anchorMap.x + (pdr.x - anchorPDR.x),
                          y: anchorMap.y + (pdr.y - anchorPDR.y))

        guard polyline.count >= 2 else {
            let d = Self.distance(pos, polyline[0])
            return FollowState(position: pos, headingDegrees: headingDegrees, progressM: 0,
                               remainingM: d, offRouteM: 0, nextTurnDirection: nil, nextTurnInM: nil,
                               prompt: Self.arrivalPrompt(remaining: d), needsReplan: false,
                               arrived: d <= Self.arrivalRadiusM)
        }

        // Project onto the route. Only look a few segments ahead of the last one so a
        // route that doubles back on itself can't teleport the marker backwards.
        let lo = max(0, segmentIndex - 1)
        let hi = min(polyline.count - 2, segmentIndex + 3)
        var best = (seg: lo, t: 0.0, cross: Double.infinity)
        for seg in lo...hi {
            let (t, cross) = Self.project(pos, onto: polyline[seg], polyline[seg + 1])
            if cross < best.cross { best = (seg, t, cross) }
        }
        segmentIndex = best.seg
        let segLen = cumulative[best.seg + 1] - cumulative[best.seg]
        let progress = cumulative[best.seg] + best.t * segLen
        // Along-route distance clamps to 0 once the projection hits the last vertex,
        // so the straight-line distance to the target is the floor.
        let toTarget = Self.distance(pos, polyline[polyline.count - 1])
        let remaining = max(totalLengthM - progress, toTarget)

        let next = turnsAlongRoute.first { $0.atM > progress - 1.0 }
        let turnIn = next.map { max(0, $0.atM - progress) }

        let arrived = toTarget <= Self.arrivalRadiusM
        let offRoute = best.cross > Self.offRouteThresholdM && !arrived
        let prompt: String
        if arrived {
            prompt = Self.arrivalPrompt(remaining: toTarget)
        } else if offRoute {
            prompt = "似乎偏離路線（偏差約 \(Int(best.cross.rounded())) 公尺），重新規劃中…"
        } else if let next, let turnIn {
            let verb = Self.zh(next.direction)
            if turnIn <= Self.turnNowM {
                prompt = "現在\(verb)"
            } else if turnIn <= Self.announceTurnM {
                prompt = "\(Int(turnIn.rounded())) 公尺後\(verb)"
            } else {
                prompt = "直走約 \(Int(turnIn.rounded())) 公尺後\(verb)"
            }
        } else {
            prompt = "直走約 \(Int(remaining.rounded())) 公尺就到"
        }

        return FollowState(position: pos, headingDegrees: headingDegrees, progressM: progress,
                           remainingM: remaining, offRouteM: best.cross,
                           nextTurnDirection: next?.direction, nextTurnInM: turnIn,
                           prompt: prompt, needsReplan: offRoute, arrived: arrived)
    }

    // MARK: - Helpers

    static func zh(_ direction: String) -> String {
        switch direction {
        case "right": return "右轉"
        case "left": return "左轉"
        case "slight_right": return "稍微右轉"
        case "slight_left": return "稍微左轉"
        case "sharp_right": return "大幅右轉"
        case "sharp_left": return "大幅左轉"
        case "behind": return "迴轉"
        default: return "直走"
        }
    }

    private static func arrivalPrompt(remaining: Double) -> String {
        "目標就在附近（約 \(max(1, Int(remaining.rounded()))) 公尺），請環顧四周"
    }

    static func distance(_ a: CGPoint, _ b: CGPoint) -> Double {
        hypot(b.x - a.x, b.y - a.y)
    }

    /// Projection of `p` onto segment a–b: (t clamped to 0...1, perpendicular distance).
    static func project(_ p: CGPoint, onto a: CGPoint, _ b: CGPoint) -> (t: Double, cross: Double) {
        let vx = b.x - a.x, vy = b.y - a.y
        let len2 = vx * vx + vy * vy
        guard len2 > 0 else { return (0, distance(p, a)) }
        let t = max(0, min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2))
        let q = CGPoint(x: a.x + t * vx, y: a.y + t * vy)
        return (t, distance(p, q))
    }

    static func nearestVertexIndex(to p: CGPoint, in line: [CGPoint]) -> Int {
        var bestI = 0
        var bestD = Double.infinity
        for (i, v) in line.enumerated() {
            let d = distance(p, v)
            if d < bestD { bestD = d; bestI = i }
        }
        return bestI
    }
}
