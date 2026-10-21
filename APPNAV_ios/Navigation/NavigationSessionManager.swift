import Combine
import Foundation

/// Orchestrates MotionPathTracker → TopologicalMap → LoopDetector for one navigation
/// session. This is the entry point the rest of the app (or a future NavigationView)
/// should use — it does not touch UI and does not upload anything anywhere.
final class NavigationSessionManager: ObservableObject {
    static let shared = NavigationSessionManager()

    @Published private(set) var isTracking = false
    @Published private(set) var currentPoint: PathPoint?
    @Published private(set) var nodeCount = 0
    @Published private(set) var lastLoopEvent: LoopEvent?

    /// Live heading from the magnetometer, updated continuously (~5Hz) even while
    /// standing still — used by the AR arrow overlay in NavigationCameraView.
    @Published private(set) var currentHeading: Double = 0

    /// Position on the store map (reference-node frame), from `MapPositionEstimator`:
    /// nil until the first confident photo fix, then dead reckoning corrected by every
    /// later fix. The single position the route map, AR arrow and uploads all use.
    @Published private(set) var mapPosition: (x: Double, y: Double)?
    /// Metres walked since the last accepted photo fix (sent to the server, which trusts
    /// the sensor position less the further the user walked — backend loc_fusion).
    @Published private(set) var distanceSinceFix: Double = 0
    /// Latest gravity vector (device frame) — for the upright-photo check.
    @Published private(set) var gravity: (x: Double, y: Double, z: Double) = (0, -1, 0)
    /// Reference node of the last accepted photo fix — the server only considers nodes
    /// walkable from it in the distance walked since (backend server._reachable_nodes).
    @Published private(set) var lastFixNodeId: Int?

    let estimator = MapPositionEstimator()

    /// Heading in the map frame (raw compass + heading offset learned from photo fixes).
    var mapHeading: Double { estimator.mapHeading(fromRaw: currentHeading) }

    /// Fires alongside `lastLoopEvent` for callers that want a callback instead of
    /// observing @Published state (e.g. a navigation coordinator deciding to reroute).
    var onLoopDetected: ((LoopEvent) -> Void)?

    var isAvailable: Bool { MotionPathTracker.isAvailable }

    /// The store/place this session's calibration was loaded for. Nil = generic.
    private(set) var currentPlace: String?

    private let tracker = MotionPathTracker()
    private let map: TopologicalMap
    private let loopDetector: LoopDetector

    /// Current PDR-estimated position (meters, origin = tracking start).
    var pdrPosition: (x: Double, y: Double) {
        guard let p = currentPoint else { return (0, 0) }
        return (p.x, p.y)
    }

    init(mergeRadius: Double = 3.0, loopDetector: LoopDetector = LoopDetector()) {
        self.map = TopologicalMap(mergeRadius: mergeRadius)
        self.loopDetector = loopDetector

        tracker.onPathPoint = { [weak self] point in
            self?.handle(point)
        }
        tracker.onHeadingUpdate = { [weak self] heading in
            self?.currentHeading = heading
        }
        tracker.onGravityUpdate = { [weak self] g in
            self?.gravity = g
        }
        self.loopDetector.onLoopDetected = { [weak self] event in
            self?.lastLoopEvent = event
            self?.onLoopDetected?(event)
        }
    }

    func start(place: String? = nil) {
        guard !isTracking else { return }
        map.reset()
        loopDetector.reset()
        nodeCount = 0
        currentPoint = nil
        lastLoopEvent = nil
        currentHeading = 0
        currentPlace = place
        estimator.reset()
        mapPosition = nil
        distanceSinceFix = 0
        lastFixNodeId = nil

        if let place, !place.isEmpty {
            tracker.loadStoreCalibration(place)
        }

        tracker.start()
        isTracking = tracker.isTracking
    }

    func stop() {
        if let place = currentPlace, !place.isEmpty {
            tracker.saveStoreCalibration(place)
        }
        tracker.stop()
        isTracking = false
        // The manager is a shared singleton: a later session's view must not see this
        // session's last position before its own start() runs.
        estimator.reset()
        mapPosition = nil
        distanceSinceFix = 0
        currentPoint = nil
        lastFixNodeId = nil
    }

    func exportSnapshot() -> TopologicalMapSnapshot {
        map.snapshot()
    }

    /// Call after MagnetometerCalibrator saves a new calibration so this session's
    /// tracker (already constructed, possibly before the calibration existed) picks
    /// it up without needing an app relaunch.
    func reloadCalibration() {
        tracker.reloadCalibration()
    }

    /// Call after `GyroCalibrator` saves a new bias so this session's tracker picks it up
    /// without needing an app relaunch — same reasoning as `reloadCalibration()`.
    func reloadGyroBias() {
        tracker.reloadGyroBias()
    }

    /// The server localized a photo to a reference node at `node` (map frame).
    /// `raw` is the PDR reading at the moment the photo was taken.
    @discardableResult
    func applyPhotoFix(node: (x: Double, y: Double), nodeId: Int?, confidence: Double,
                       raw: (x: Double, y: Double)) -> MapPositionEstimator.FixResult {
        let result = estimator.applyFix(node: node, confidence: confidence, raw: raw)
        if result == .accepted { lastFixNodeId = nodeId }
        // The photo was taken a moment ago; replay the steps walked since then so the
        // position is "now", not "when the shutter fired".
        if result == .accepted, let p = currentPoint {
            estimator.ingest(raw: (p.x, p.y))
        }
        publishEstimate()
        return result
    }

    /// The user says they are at the store entrance: the map origin (editor map frame:
    /// origin = entrance, +y = heading 0°, same as the PDR frame), so the sensors are on
    /// the map from the first step, before any photo.
    func startAtEntrance(nodeId: Int?) {
        let raw = currentPoint.map { ($0.x, $0.y) } ?? (0, 0)
        estimator.applyFix(node: (0, 0), confidence: 1, raw: raw)
        lastFixNodeId = nodeId
        publishEstimate()
    }

    private func publishEstimate() {
        mapPosition = estimator.mapPosition
        distanceSinceFix = estimator.distanceSinceFixM
    }

    private func handle(_ point: PathPoint) {
        let (node, driftCorrection) = map.ingest(point)
        estimator.ingest(raw: (point.x, point.y))

        var correctedPoint = point
        if let driftCorrection {
            // Revisiting an already-mapped node — pull the live dead-reckoning estimate
            // back toward that anchor so drift doesn't keep growing for the rest of the
            // session (see TopologicalMap.ingest and MotionPathTracker.applyDriftCorrection).
            tracker.applyDriftCorrection(dx: driftCorrection.dx, dy: driftCorrection.dy)
            correctedPoint.x += driftCorrection.dx
            correctedPoint.y += driftCorrection.dy
            // the raw frame jumped, the user didn't move — don't integrate the jump
            estimator.rebase(raw: (correctedPoint.x, correctedPoint.y))
        }
        publishEstimate()

        currentPoint = correctedPoint
        nodeCount = map.nodes.count
        loopDetector.recordVisit(nodeID: node.id, at: point.timestamp)
    }
}
