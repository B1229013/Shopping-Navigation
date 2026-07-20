import Foundation

struct LoopEvent: Equatable {
    let nodeID: String
    let repeatCount: Int
    let distinctNodesInWindow: Int
    let detectedAt: Double
}

/// Flags "circling" — the same map node being revisited several times within a short
/// window while the surrounding area stays small (distinguishes actual circling from
/// a long route that happens to pass a hub node twice).
final class LoopDetector {
    private struct Visit {
        let nodeID: String
        let timestamp: Double
    }

    var repeatThreshold: Int
    var windowSeconds: Double
    var cooldownSeconds: Double

    private var recentVisits: [Visit] = []
    private var lastFiredNodeID: String?
    private var lastFiredAt: Double = 0

    var onLoopDetected: ((LoopEvent) -> Void)?

    init(repeatThreshold: Int = 3, windowSeconds: Double = 90, cooldownSeconds: Double = 45) {
        self.repeatThreshold = repeatThreshold
        self.windowSeconds = windowSeconds
        self.cooldownSeconds = cooldownSeconds
    }

    func recordVisit(nodeID: String, at timestamp: Double) {
        recentVisits.append(Visit(nodeID: nodeID, timestamp: timestamp))
        let cutoff = timestamp - windowSeconds * 1000
        recentVisits.removeAll { $0.timestamp < cutoff }

        let matches = recentVisits.filter { $0.nodeID == nodeID }
        guard matches.count >= repeatThreshold else { return }

        if nodeID == lastFiredNodeID, (timestamp - lastFiredAt) < cooldownSeconds * 1000 {
            return
        }

        lastFiredNodeID = nodeID
        lastFiredAt = timestamp

        onLoopDetected?(LoopEvent(
            nodeID: nodeID,
            repeatCount: matches.count,
            distinctNodesInWindow: Set(recentVisits.map(\.nodeID)).count,
            detectedAt: timestamp
        ))
    }

    func reset() {
        recentVisits.removeAll()
        lastFiredNodeID = nil
        lastFiredAt = 0
    }
}
