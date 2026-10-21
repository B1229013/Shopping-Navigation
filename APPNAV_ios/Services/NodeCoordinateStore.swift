import Foundation

struct NodeCoordinate: Codable {
    let pdrX: Double
    let pdrY: Double

    enum CodingKeys: String, CodingKey {
        case pdrX = "pdr_x"
        case pdrY = "pdr_y"
    }
}

final class NodeCoordinateStore {
    static let shared = NodeCoordinateStore()

    /// The store node_coordinates.json — and the on-device DINOv2 references
    /// (dino_ref_embeddings.bin / dino_ref_meta.json) — were exported from. Other stores on
    /// the shared backend reuse the same node numbers (系辦 has 28 of its 36 in common with
    /// A7), so this data names the wrong nodes anywhere else: use it only in this store.
    static let place = "A7家樂福（9/16)"

    private var coords: [Int: NodeCoordinate] = [:]
    private(set) var isLoaded = false

    func unload() {
        coords = [:]
        isLoaded = false
    }

    func load() {
        guard let url = Bundle.main.url(forResource: "node_coordinates", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let dict = try? JSONDecoder().decode([String: NodeCoordinate].self, from: data) else {
            return
        }
        for (key, value) in dict {
            if let nodeId = Int(key) {
                coords[nodeId] = value
            }
        }
        isLoaded = true
    }

    func coordinate(for nodeId: Int) -> (x: Double, y: Double)? {
        guard let c = coords[nodeId] else { return nil }
        return (c.pdrX, c.pdrY)
    }

    func nearestNode(to position: (x: Double, y: Double)) -> (nodeId: Int, distance: Double)? {
        var best: (nodeId: Int, distance: Double)?
        for (nodeId, c) in coords {
            let dx = c.pdrX - position.x
            let dy = c.pdrY - position.y
            let dist = (dx * dx + dy * dy).squareRoot()
            if best == nil || dist < best!.distance {
                best = (nodeId, dist)
            }
        }
        return best
    }

    /// Given DINOv2 top-K candidates and a PDR-estimated position, pick the best match
    /// using a fusion score that combines visual similarity and spatial proximity.
    func fuseDINOWithPDR(
        candidates: [(nodeId: Int, similarity: Float)],
        pdrPosition: (x: Double, y: Double)
    ) -> Int? {
        guard !candidates.isEmpty else { return nil }

        var bestNodeId: Int?
        var bestScore: Double = -.infinity

        for (nodeId, similarity) in candidates {
            guard let coord = coordinate(for: nodeId) else { continue }
            let dx = coord.x - pdrPosition.x
            let dy = coord.y - pdrPosition.y
            let distance = (dx * dx + dy * dy).squareRoot()

            // Normalize distance: closer = higher score. Use exponential decay
            // so nodes within ~5m are strongly preferred but visual similarity
            // still matters at larger distances.
            let proxScore = exp(-distance / 8.0)
            let visScore = Double(similarity)

            // Weight: 40% proximity, 60% visual similarity.
            // When PDR is well-calibrated, proximity strongly disambiguates;
            // visual similarity remains the primary signal.
            let fusedScore = 0.6 * visScore + 0.4 * proxScore
            if fusedScore > bestScore {
                bestScore = fusedScore
                bestNodeId = nodeId
            }
        }

        return bestNodeId
    }

    /// Bearing in degrees (0=north, 90=east) from `from` to `to` in the PDR coordinate system.
    func bearing(from: (x: Double, y: Double), to: (x: Double, y: Double)) -> Double {
        let dx = to.x - from.x
        let dy = to.y - from.y
        var angle = atan2(dx, dy) * 180.0 / .pi
        if angle < 0 { angle += 360 }
        return angle
    }
}
