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

    /// Fires alongside `lastLoopEvent` for callers that want a callback instead of
    /// observing @Published state (e.g. a navigation coordinator deciding to reroute).
    var onLoopDetected: ((LoopEvent) -> Void)?

    var isAvailable: Bool { MotionPathTracker.isAvailable }

    private let tracker = MotionPathTracker()
    private let map: TopologicalMap
    private let loopDetector: LoopDetector

    init(mergeRadius: Double = 3.0, loopDetector: LoopDetector = LoopDetector()) {
        self.map = TopologicalMap(mergeRadius: mergeRadius)
        self.loopDetector = loopDetector

        tracker.onPathPoint = { [weak self] point in
            self?.handle(point)
        }
        self.loopDetector.onLoopDetected = { [weak self] event in
            self?.lastLoopEvent = event
            self?.onLoopDetected?(event)
        }
    }

    func start() {
        guard !isTracking else { return }
        map.reset()
        loopDetector.reset()
        nodeCount = 0
        currentPoint = nil
        lastLoopEvent = nil
        tracker.start()
        isTracking = tracker.isTracking
    }

    func stop() {
        tracker.stop()
        isTracking = false
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

    private func handle(_ point: PathPoint) {
        let (node, driftCorrection) = map.ingest(point)

        var correctedPoint = point
        if let driftCorrection {
            // Revisiting an already-mapped node — pull the live dead-reckoning estimate
            // back toward that anchor so drift doesn't keep growing for the rest of the
            // session (see TopologicalMap.ingest and MotionPathTracker.applyDriftCorrection).
            tracker.applyDriftCorrection(dx: driftCorrection.dx, dy: driftCorrection.dy)
            correctedPoint.x += driftCorrection.dx
            correctedPoint.y += driftCorrection.dy
        }

        currentPoint = correctedPoint
        nodeCount = map.nodes.count
        loopDetector.recordVisit(nodeID: node.id, at: point.timestamp)
    }
}
