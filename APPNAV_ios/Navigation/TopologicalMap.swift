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

    @discardableResult
    func ingest(_ point: PathPoint) -> PathNode {
        let node: PathNode
        if let index = nearestNodeIndex(to: point), distance(nodes[index], point) <= mergeRadius {
            nodes[index].visitCount += 1
            nodes[index].lastVisitedAt = point.timestamp
            let n = Double(nodes[index].visitCount)
            nodes[index].x += (point.x - nodes[index].x) / n
            nodes[index].y += (point.y - nodes[index].y) / n
            node = nodes[index]
        } else {
            node = PathNode(x: point.x, y: point.y, firstVisitedAt: point.timestamp, lastVisitedAt: point.timestamp)
            nodes.append(node)
        }

        if let lastID = lastNodeID, lastID != node.id {
            addOrIncrementEdge(from: lastID, to: node.id)
        }
        lastNodeID = node.id
        return node
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
