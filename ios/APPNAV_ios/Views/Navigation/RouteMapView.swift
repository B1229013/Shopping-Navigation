import Combine
import SwiftUI

/// Owns the editor-map route for a navigation session and drives `PathFollower`
/// with the PDR points `NavigationSessionManager` publishes.
@MainActor
final class RouteFollowerModel: ObservableObject {
    @Published private(set) var route: PathResponse?
    @Published private(set) var state: FollowState?
    @Published private(set) var statusMessage: String?

    private let follower = PathFollower()
    private var lastPDR: PathPoint?
    private var lastReplanAt: Date = .distantPast
    private var isLoading = false

    /// Fetch the route. `fromPhone` re-plans from the dead-reckoned position and keeps
    /// the current anchor; otherwise the server starts from the last photo localization
    /// (or the entrance) and the user is re-anchored at that start.
    func load(sessionId: String, fromPhone: Bool = false) async {
        guard !isLoading else { return }
        isLoading = true
        defer { isLoading = false }
        let pdr = lastPDR
        do {
            let response: PathResponse
            if fromPhone, let s = state {
                response = try await NavigationAPI.shared.getPath(
                    sessionId: sessionId, x: s.position.x, y: s.position.y, heading: s.headingDegrees)
            } else {
                response = try await NavigationAPI.shared.getPath(sessionId: sessionId,
                                                                 heading: pdr?.headingDegrees)
            }
            route = response
            let polyline = response.polyline.map { CGPoint(x: $0[0], y: $0[1]) }
            let turns = response.turns.map { (at: CGPoint(x: $0.at[0], y: $0.at[1]), direction: $0.direction) }
            let pdrPoint = pdr.map { CGPoint(x: $0.x, y: $0.y) } ?? .zero
            follower.setRoute(polyline: polyline, turns: turns, pdr: pdrPoint, keepAnchor: fromPhone)
            statusMessage = nil
            if let pdr { ingest(pdr) } else {
                state = follower.update(pdr: pdrPoint, headingDegrees: 0)
            }
        } catch {
            // Most likely: no editor map for this place, or the goal isn't on it.
            let text = "\(error)"
            if text.contains("no_editor_map") {
                statusMessage = "這個賣場沒有航點地圖，無法顯示路徑"
            } else if text.contains("goal_not_on_map") {
                statusMessage = "地圖上找不到「\(route?.goalItem ?? "目標")」的位置"
            } else {
                statusMessage = "路徑載入失敗：\(error.localizedDescription)"
            }
        }
    }

    /// A new PDR sample from the sensors.
    func ingest(_ point: PathPoint) {
        lastPDR = point
        guard follower.hasRoute else { return }
        state = follower.update(pdr: CGPoint(x: point.x, y: point.y), headingDegrees: point.headingDegrees)
    }

    /// True when the follower wants a re-plan and we haven't asked too recently.
    func shouldReplan() -> Bool {
        guard let state, state.needsReplan else { return false }
        guard Date().timeIntervalSince(lastReplanAt) > 8 else { return false }
        lastReplanAt = Date()
        return true
    }
}

/// Top-down map of the store's walkable waypoints with the planned route, the user's
/// dead-reckoned position and the target. Map metres → screen points, y flipped
/// (map +y is "north" / heading 0°).
struct RouteMapView: View {
    let route: PathResponse
    let state: FollowState?

    var body: some View {
        Canvas { context, size in
            let pts = route.nodes.map { CGPoint(x: $0.x, y: $0.y) }
            guard let bounds = Self.bounds(of: pts) else { return }
            let pad: CGFloat = 12
            let sx = (size.width - 2 * pad) / max(bounds.width, 1)
            let sy = (size.height - 2 * pad) / max(bounds.height, 1)
            let scale = min(sx, sy)
            func toScreen(_ p: CGPoint) -> CGPoint {
                CGPoint(x: pad + (p.x - bounds.minX) * scale,
                        y: size.height - pad - (p.y - bounds.minY) * scale)
            }

            // walkable edges + waypoints
            let byId = Dictionary(uniqueKeysWithValues: route.nodes.map { ($0.id, CGPoint(x: $0.x, y: $0.y)) })
            var edges = Path()
            for e in route.edges {
                guard let a = byId[e.from], let b = byId[e.to] else { continue }
                edges.move(to: toScreen(a))
                edges.addLine(to: toScreen(b))
            }
            context.stroke(edges, with: .color(.white.opacity(0.25)), lineWidth: 1)
            for p in pts {
                let s = toScreen(p)
                context.fill(Path(ellipseIn: CGRect(x: s.x - 1.5, y: s.y - 1.5, width: 3, height: 3)),
                             with: .color(.white.opacity(0.5)))
            }

            // planned route
            var line = Path()
            for (i, v) in route.polyline.enumerated() {
                let s = toScreen(CGPoint(x: v[0], y: v[1]))
                if i == 0 { line.move(to: s) } else { line.addLine(to: s) }
            }
            context.stroke(line, with: .color(.cyan), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))

            // target
            let t = toScreen(CGPoint(x: route.target.x, y: route.target.y))
            context.fill(Path(ellipseIn: CGRect(x: t.x - 6, y: t.y - 6, width: 12, height: 12)), with: .color(.orange))
            context.stroke(Path(ellipseIn: CGRect(x: t.x - 6, y: t.y - 6, width: 12, height: 12)),
                           with: .color(.white), lineWidth: 1.5)

            // user
            if let state {
                let u = toScreen(state.position)
                let color: Color = state.needsReplan ? .red : .green
                context.fill(Path(ellipseIn: CGRect(x: u.x - 6, y: u.y - 6, width: 12, height: 12)), with: .color(color))
                context.stroke(Path(ellipseIn: CGRect(x: u.x - 6, y: u.y - 6, width: 12, height: 12)),
                               with: .color(.white), lineWidth: 1.5)
                // heading arrow: map heading 0° = +y = screen up, clockwise
                let rad = state.headingDegrees * .pi / 180
                let tip = CGPoint(x: u.x + 14 * sin(rad), y: u.y - 14 * cos(rad))
                var arrow = Path()
                arrow.move(to: u)
                arrow.addLine(to: tip)
                context.stroke(arrow, with: .color(color), style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
            }
        }
        .background(Color.black.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private static func bounds(of pts: [CGPoint]) -> CGRect? {
        guard let first = pts.first else { return nil }
        var minX = first.x, maxX = first.x, minY = first.y, maxY = first.y
        for p in pts {
            minX = min(minX, p.x); maxX = max(maxX, p.x)
            minY = min(minY, p.y); maxY = max(maxY, p.y)
        }
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}
