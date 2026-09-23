import Foundation

// ── Request / Response models (mirrors backend/server + Android NavigationApi.kt) ──

struct StartSessionRequest: Codable {
    let goal: String
    let place: String?
}

struct StartSessionResponse: Codable {
    let sessionId: String
    let guidance: String
    let action: String
    let goalObjects: [String]
    let place: String?
}

struct PlaceInfo: Codable, Identifiable, Hashable {
    let name: String
    let photoCount: Int

    var id: String { name }
}

struct PlacesResponse: Codable {
    let places: [PlaceInfo]
}

struct TurnResponse: Codable {
    let action: String   // "ARRIVED", "MOVE", "ASK"
    let guidance: String
    let question: String?
    let nodeId: Int
    let annotatedPhotoUrl: String?
    // Neo4j visual localization (position correction from reference map)
    let correctedNodeId: Int?
    let correctedConfidence: Double?
    let correctedLocation: String?
    // Map-derived turn-by-turn line (path with distance / "you're at the target" /
    // "photo could not be localized"); shown verbatim so the map's verdict is visible.
    let nextInstruction: String?
    // Navigation phase
    let phase: String?   // "shopping", "checkout", "exit", "done"
}

/// Route on the hand-corrected editor map (GET /session/{id}/path): the polyline the
/// phone's PathFollower walks, its turn list, and the whole map for the mini-map.
struct PathResponse: Codable {
    struct Start: Codable {
        let wp: Int
        let x: Double
        let y: Double
        let source: String          // "phone" | "photo" | "entrance"
    }
    struct Target: Codable {
        let wp: Int
        let x: Double
        let y: Double
        let products: [String]
        let neoNid: Int?
    }
    struct Turn: Codable {
        let direction: String       // straight / right / left / slight_* / sharp_* / behind
        let textZh: String
        let at: [Double]            // [x, y] where this step starts
        let to: [Double]
        let distanceM: Double
        let passedNodes: Int
    }
    struct Node: Codable {
        let id: Int
        let x: Double
        let y: Double
    }
    struct Edge: Codable {
        let from: Int
        let to: Int
        let length: Double
    }

    let place: String
    let goalItem: String
    let start: Start
    let target: Target
    let path: [Int]
    let polyline: [[Double]]        // metres; origin = entrance, +y = heading 0°
    let turns: [Turn]
    let distanceM: Double
    let nodes: [Node]
    let edges: [Edge]
}

/// Response from the standalone /localize endpoint.
struct LocalizationResponse: Codable {
    let matchedNid: Int?
    let confidence: Double
    let method: String
    let reasoning: String
    let detectedObjects: [String]?
    let ocrTexts: [String]?
    let refLocation: RefLocation?
    let runnerUp: RunnerUp?

    struct RefLocation: Codable {
        let nid: Int
        let photoFile: String
        let pdrX: Double
        let pdrY: Double
        let headingDeg: Double
        let session: String
    }

    struct RunnerUp: Codable {
        let nid: Int
        let score: Double
    }
}

struct AnswerRequest: Codable {
    let answer: String
}

struct ConfirmArrivalRequest: Codable {
    let kind: String   // "confirmed", "false_positive", "wrong_instance"
}

struct SessionState: Codable {
    let id: String
    let goal: String
    let goalObjects: [String]
    let pendingQuestion: String?
    let arrived: Bool
    let lastNodeId: Int?
    let goalNode: Int?
    let createdAt: String
}

struct MapNode: Codable {
    let id: Int
    let photo: String
    let detected: [String]
    let summary: String
    let timestamp: String
}

struct MapEdge: Codable {
    let from: Int
    let to: Int
    let action: String
}

struct MapResponse: Codable {
    let nodes: [MapNode]
    let edges: [MapEdge]
    let currentNode: Int?
    let goalNode: Int?
}

struct HealthResponse: Codable {
    let status: String
}

struct StatusResponse: Codable {
    let status: String
}

struct SensorTestLap: Codable {
    let lapNumber: Int
    let x: Double
    let y: Double
    let timestamp: Double
    let distanceFromOrigin: Double
    let distanceFromPreviousLap: Double
}

/// Snapshot of the heading-fusion A/B toggle and gait profile in effect for one
/// `SensorTestView` walk, so a later look at the uploaded record.json can tell which
/// configuration actually produced it instead of guessing.
struct SensorTestConfig: Codable {
    var useGyroFusion: Bool
    var useCalibrationCorrection: Bool
    var heightCM: Double
    var gaitK: Double
    var gaitK1: Double
}

struct SensorTestRecord: Codable {
    let points: [PathPoint]
    let laps: [SensorTestLap]
    let recordedAt: Double
    let config: SensorTestConfig?
}

struct SensorTestUploadResponse: Codable {
    let testId: String
    let status: String
}

final class NavigationAPI {
    static let shared = NavigationAPI()

    private var baseURL: URL? {
        // Runtime override (editable in Settings) takes priority over the Info.plist
        // default, so switching networks/environments never requires a rebuild.
        var raw = UserDefaults.standard.string(forKey: "navigationBackendURL") ?? AppConfig.navigationBackendURL
        guard !raw.isEmpty else { return nil }
        if raw.hasSuffix("/") { raw.removeLast() }
        return URL(string: raw)
    }

    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    private let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()

    private func request(path: String, method: String) throws -> URLRequest {
        guard let baseURL else { throw NavigationAPIError.missingBackendURL }
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = 30
        return request
    }

    private func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw NavigationAPIError.serverError(body)
        }
        return try decoder.decode(T.self, from: data)
    }

    /// Route on the editor map. Pass the phone's dead-reckoned position to re-plan from
    /// where the user actually is; omit it to start from the last photo localization
    /// (or the entrance).
    func getPath(sessionId: String, x: Double? = nil, y: Double? = nil,
                 heading: Double? = nil) async throws -> PathResponse {
        guard let baseURL else { throw NavigationAPIError.missingBackendURL }
        var components = URLComponents(url: baseURL.appendingPathComponent("session/\(sessionId)/path"),
                                       resolvingAgainstBaseURL: false)!
        var items: [URLQueryItem] = []
        if let x, let y {
            items.append(URLQueryItem(name: "x", value: String(format: "%.2f", x)))
            items.append(URLQueryItem(name: "y", value: String(format: "%.2f", y)))
        }
        if let heading {
            items.append(URLQueryItem(name: "heading", value: String(format: "%.1f", heading)))
        }
        if !items.isEmpty { components.queryItems = items }
        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.timeoutInterval = 30
        return try await send(request)
    }

    func getPlaces() async throws -> [PlaceInfo] {
        let request = try request(path: "places", method: "GET")
        let response: PlacesResponse = try await send(request)
        return response.places
    }

    func startSession(goal: String, place: String? = nil) async throws -> StartSessionResponse {
        var request = try request(path: "session", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(StartSessionRequest(goal: goal, place: place))
        return try await send(request)
    }

    /// `heading` is the phone's own compass/gyro heading (degrees, 0° = map +y). The
    /// server prefers it over the heading guessed from reference photos.
    func uploadPhoto(sessionId: String, imageData: Data, heading: Double? = nil) async throws -> TurnResponse {
        var request = try request(path: "session/\(sessionId)/photo", method: "POST")
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 300

        var body = Data()
        if let heading {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"heading\"\r\n\r\n".data(using: .utf8)!)
            body.append(String(format: "%.1f", heading).data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"photo\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        return try await send(request)
    }

    func postAnswer(sessionId: String, answer: String) async throws -> TurnResponse {
        var request = try request(path: "session/\(sessionId)/answer", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(AnswerRequest(answer: answer))
        return try await send(request)
    }

    func confirmArrival(sessionId: String, kind: String) async throws -> TurnResponse {
        var request = try request(path: "session/\(sessionId)/confirm", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(ConfirmArrivalRequest(kind: kind))
        return try await send(request)
    }

    func getSession(sessionId: String) async throws -> SessionState {
        let request = try request(path: "session/\(sessionId)", method: "GET")
        return try await send(request)
    }

    func getMap(sessionId: String) async throws -> MapResponse {
        let request = try request(path: "session/\(sessionId)/map?format=json", method: "GET")
        return try await send(request)
    }

    /// Stores the on-device PDR sensor map next to this session's VLM output on the
    /// backend machine, purely so it can be inspected — not read back or used by
    /// the VLM navigation logic.
    func uploadSensorMap(sessionId: String, snapshot: TopologicalMapSnapshot) async throws {
        var request = try request(path: "session/\(sessionId)/sensor-map", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(snapshot)
        let _: StatusResponse = try await send(request)
    }

    /// Uploads a standalone PDR sensor-accuracy test (SensorTestView.swift) — no
    /// navigation session involved. Backend stores the raw path + laps and renders
    /// a PNG plot; returns the test_id used to fetch that plot back.
    func uploadSensorTest(points: [PathPoint], laps: [SensorTestLap], config: SensorTestConfig?) async throws -> String {
        var request = try request(path: "sensor-test", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = SensorTestRecord(points: points, laps: laps, recordedAt: Date().timeIntervalSince1970 * 1000, config: config)
        request.httpBody = try encoder.encode(body)
        let response: SensorTestUploadResponse = try await send(request)
        return response.testId
    }

    /// Standalone localization: upload a photo and get the matching reference
    /// node from the Neo4j topological map, without starting a navigation session.
    func localize(imageData: Data) async throws -> LocalizationResponse {
        var request = try request(path: "localize", method: "POST")
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 120

        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"photo\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body

        return try await send(request)
    }

    func health() async throws -> HealthResponse {
        let request = try request(path: "health", method: "GET")
        return try await send(request)
    }

    enum NavigationAPIError: Error {
        case missingBackendURL
        case serverError(String)
    }
}
