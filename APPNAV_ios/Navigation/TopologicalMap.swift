import Foundation

/// A visited area, not an exact coordinate — nearby PathPoints get merged into
/// the same node so the map stays a small topological graph instead of a raw point cloud.
struct PathNode: Codable, Identifiable, Equatable {
    var id: String = UUID().uuidString
    var x: Double
    var y: Double
    var visitCount: Int = 1
    var firstVisitedAt: Double
    var lastVisitedAt: Double
}

struct PathEdge: Codable, Identifiable, Equatable {
    var id: String = UUID().uuidString
    var fromNodeID: String
    var toNodeID: String
    var traversalCount: Int = 1
}

/// Exportable snapshot of a built map — the intended shape for a future cloud upload,
/// not wired to any network call yet.
struct TopologicalMapSnapshot: Codable {
    let nodes: [PathNode]
    let edges: [PathEdge]
    let recordedAt: Double
}

/// Builds a topological (graph, not metric) map from a stream of PathPoints.
/// Points within `mergeRadius` of an existing node are folded into it instead of
/// creating a new node, which is what lets loop detection work off node revisits.
final class TopologicalMap {
    private(set) var nodes: [PathNode] = []
    private(set) var edges: [PathEdge] = []
    private var lastNodeID: String?

    let mergeRadius: Double

    init(mergeRadius: Double = 3.0) {
        self.mergeRadius = mergeRadius
    }

    /// A revisited node's own (x, y) is intentionally left untouched here — dragging it
    /// toward each new merged point (the old behavior) let a drifted revisit slowly
    /// corrupt the one thing meant to be trustworthy. Instead the node stays anchored at
    /// wherever it was first visited, and the caller (NavigationSessionManager) uses the
    /// returned `driftCorrection` to pull the *live* dead-reckoning position back toward
    /// that anchor — the causality that actually fixes drift instead of spreading it.
    ///
    /// `driftCorrection` only fires on the tick that *transitions into* a node (this merge's
    /// node differs from `lastNodeID`) — not on every subsequent tick spent merged into that
    /// same node. Firing on every tick was a real bug: a single footstep (~0.6–0.9m) is
    /// always well inside `mergeRadius` (3m), so once *any* point merged into a node, the
    /// correction snapped the caller's position to that node's exact (x, y) — and the very
    /// next step, still within `mergeRadius` of that same fixed point, would merge and snap
    /// again, forever. That pins the tracked position to the first node created for the rest
    /// of the session: no second node can ever form, because the corrected position can
    /// never actually get `mergeRadius` away from the first one. Confirmed by replaying a
    /// real recorded walk (324 points) through this exact logic: it collapsed to 1 node with
    /// 323 corrections — a correction on very nearly every single point. Gating on "this is a
    /// genuine transition, not a continued stay" is what the doc comment's "revisit" framing
    /// already implied but the code never actually checked.
    @discardableResult
    func ingest(_ point: PathPoint) -> (node: PathNode, driftCorrection: (dx: Double, dy: Double)?) {
        let node: PathNode
        var driftCorrection: (dx: Double, dy: Double)?

        if let index = nearestNodeIndex(to: point), distance(nodes[index], point) <= mergeRadius {
            let isTransitioningIn = nodes[index].id != lastNodeID
            nodes[index].visitCount += 1
            nodes[index].lastVisitedAt = point.timestamp
            if isTransitioningIn {
                driftCorrection = (nodes[index].x - point.x, nodes[index].y - point.y)
            }
            node = nodes[index]
        } else {
            node = PathNode(x: point.x, y: point.y, firstVisitedAt: point.timestamp, lastVisitedAt: point.timestamp)
            nodes.append(node)
        }

        if let lastID = lastNodeID, lastID != node.id {
            addOrIncrementEdge(from: lastID, to: node.id)
        }
        lastNodeID = node.id
        return (node, driftCorrection)
    }

    func reset() {
        nodes.removeAll()
        edges.removeAll()
        lastNodeID = nil
    }

    func snapshot() -> TopologicalMapSnapshot {
        TopologicalMapSnapshot(nodes: nodes, edges: edges, recordedAt: Date().timeIntervalSince1970 * 1000)
    }

    private func nearestNodeIndex(to point: PathPoint) -> Int? {
        guard !nodes.isEmpty else { return nil }
        var bestIndex = 0
        var bestDistance = distance(nodes[0], point)
        for i in 1..<nodes.count {
            let d = distance(nodes[i], point)
            if d < bestDistance {
                bestDistance = d
                bestIndex = i
            }
        }
        return bestIndex
    }

    private func distance(_ node: PathNode, _ point: PathPoint) -> Double {
        let dx = node.x - point.x
        let dy = node.y - point.y
        return (dx * dx + dy * dy).squareRoot()
    }

    private func addOrIncrementEdge(from: String, to: String) {
        if let index = edges.firstIndex(where: {
            ($0.fromNodeID == from && $0.toNodeID == to) || ($0.fromNodeID == to && $0.toNodeID == from)
        }) {
            edges[index].traversalCount += 1
        } else {
            edges.append(PathEdge(fromNodeID: from, toNodeID: to))
        }
    }
}
