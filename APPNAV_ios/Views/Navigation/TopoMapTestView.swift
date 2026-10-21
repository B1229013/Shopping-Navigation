import Combine
import SwiftUI

enum TopoMapMode: String, CaseIterable {
    case online = "線上"
    case offline = "離線"
}

/// Debug tool that exercises `TopologicalMap → LoopDetector` together with the five
/// heading methods from `SensorComboTracker` — the same map-building pipeline
/// `NavigationSessionManager` runs during a real navigation session — but standalone, with
/// no backend/photo/VLM involved, so node merging, loop detection, and drift correction can
/// be inspected against any of the five heading sources.
///
/// Two modes:
/// - **線上 (online)**: walk live, right now, with one chosen heading method feeding the
///   map in real time (same as this tool always did, just with a method picker added).
/// - **離線 (offline)**: pick a walk already recorded in `SensorComboTestView` (which
///   records all five methods plus waypoint photos in one pass) and replay any one of its
///   methods through a fresh map instantly, no walking required — this is how the user
///   plans to primarily use this tool: record once outdoors with photos at each stop, then
///   compare which heading method's map looks most correct back at a desk.
///
/// Deliberately a separate screen from `SensorTestView`/`SensorComboTestView` rather than
/// folding this in: those stay clean, map-free readouts of the raw sensor pipeline for
/// isolating sensor-accuracy questions; this one is for topology-layer questions (does a
/// revisit correctly merge into an existing node, does drift correction actually pull the
/// position back, does the loop-closure trigger) without the map obscuring which one is at
/// fault.
struct TopoMapTestView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = TopoMapTestModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    modeCard
                    if model.mode == .online {
                        onlineControlsCard
                        lapsCard
                    } else {
                        offlineControlsCard
                    }
                    statusCard
                    canvasCard
                }
                .padding(20)
                .padding(.top, 4)
            }
            .scrollDismissesKeyboard(.interactively)
            .background(Color.appBg)
            .navigationTitle("拓樸地圖測試")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .onDisappear { model.stop() }
    }

    private var modeCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("模式", selection: $model.mode) {
                ForEach(TopoMapMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(model.isTracking)

            Text(model.mode == .online
                 ? "現在馬上走，即時建圖，用下面選的方向法即時驅動地圖。"
                 : "從「感測器組合測試」錄好的紀錄裡挑一筆＋一種方向法，一次重建地圖，不用重新走。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var onlineControlsCard: some View {
        VStack(spacing: 12) {
            Picker("驅動地圖的方向法", selection: $model.onlineMethod) {
                ForEach(HeadingMethodID.allCases, id: \.self) { method in
                    Text(method.rawValue).tag(method)
                }
            }
            .pickerStyle(.menu)
            .disabled(model.isTracking)

            Button(model.isTracking ? "停止測試" : "開始測試") {
                if model.isTracking {
                    model.stop()
                } else {
                    model.start()
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 54)
            .background(model.isTracking ? Color.red : Color.appAccent)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .disabled(!SensorComboTracker.isAvailable)

            Button("標記回到原點（第 \(model.laps.count + 1) 圈）") {
                model.markLap()
            }
            .frame(maxWidth: .infinity)
            .frame(height: 46)
            .background(Color.orange)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .disabled(!model.isTracking)

            if !SensorComboTracker.isAvailable {
                Text("這台裝置不支援計步器（模擬器一定會顯示這個）")
                    .font(.caption)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
            }
        }
        .padding(20)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    /// Mirrors SensorTestView's lap list — same idea (走一圈回原地按一下，不用停止測試),
    /// applied here so a loop-closure/drift-correction test can cover several laps in one
    /// continuous walk instead of stopping and restarting (which wipes the map) after every
    /// single loop.
    @ViewBuilder
    private var lapsCard: some View {
        if !model.laps.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("這輪已標記的圈數")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.appTextPrimary)
                ForEach(model.laps, id: \.lapNumber) { lap in
                    HStack {
                        Text("第 \(lap.lapNumber) 圈")
                            .foregroundColor(.appTextSecondary)
                        Spacer()
                        VStack(alignment: .trailing, spacing: 2) {
                            Text("跟原點差 \(String(format: "%.2f", lap.distanceFromOrigin)) 公尺")
                            Text("跟上一圈差 \(String(format: "%.2f", lap.distanceFromPreviousLap)) 公尺")
                                .foregroundColor(.appTextTertiary)
                            Text("當時節點數 \(lap.nodeCountAtLap)")
                                .foregroundColor(.appTextTertiary)
                        }
                        .font(.caption)
                    }
                }
            }
            .padding(16)
            .background(Color.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }

    private var offlineControlsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Button("重新整理紀錄清單") {
                model.refreshAvailableBatches()
            }
            .font(.caption.weight(.semibold))
            .foregroundColor(.blue)

            if model.availableBatches.isEmpty {
                Text("還沒有「感測器組合測試」的紀錄，先去那個畫面錄一筆。")
                    .font(.caption)
                    .foregroundColor(.appTextTertiary)
            } else {
                Picker("選一筆紀錄", selection: $model.selectedBatchId) {
                    Text("尚未選擇").tag(String?.none)
                    ForEach(model.availableBatches) { batch in
                        Text("\(batch.startedAtDisplay)（\(batch.points.count) 點，\(batch.waypoints.count) 標記）").tag(String?.some(batch.id))
                    }
                }
                .pickerStyle(.menu)

                Picker("選一種方向法", selection: $model.selectedOfflineMethod) {
                    ForEach(HeadingMethodID.allCases, id: \.self) { method in
                        Text(method.rawValue).tag(method)
                    }
                }
                .pickerStyle(.menu)

                Button("重建地圖") {
                    model.buildOfflineMap()
                }
                .frame(maxWidth: .infinity)
                .frame(height: 46)
                .background(Color.appAccent)
                .foregroundColor(.white)
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .disabled(model.selectedBatchId == nil)
            }
        }
        .padding(20)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .onAppear { model.refreshAvailableBatches() }
    }

    private var statusCard: some View {
        VStack(spacing: 12) {
            metricRow(label: "狀態", value: model.isTracking ? "追蹤中" : (model.mode == .offline ? "離線重建" : "已停止"))
            metricRow(label: "節點數", value: "\(model.nodeCount)")
            metricRow(label: "邊數", value: "\(model.edgeCount)")
            Divider()
            metricRow(label: "步數", value: "\(model.stepCount)")
            metricRow(label: "方向角", value: model.headingDegrees >= 0 ? String(format: "%.0f°", model.headingDegrees) : "—")
            metricRow(label: "累積漂移修正次數", value: "\(model.driftCorrectionCount)")
            if let event = model.lastLoopEvent {
                metricRow(label: "偵測到繞圈", value: "同一節點第 \(event.repeatCount) 次經過")
            }
            if !model.waypointMarkers.isEmpty {
                metricRow(label: "標記點數", value: "\(model.waypointMarkers.count)")
            }
        }
        .padding(20)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var canvasCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("拓樸地圖")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.appTextPrimary)
            Text("藍點＝已建立的節點，紅點＝目前所在節點，灰線＝走過的邊，橘色數字＝拍照標記點。同一個位置附近的點會合併成同一顆節點（合併半徑 3 公尺），走回舊節點時會觸發漂移修正。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)

            Canvas { context, size in
                drawGraph(context: context, size: size)
            }
            .frame(height: 340)
            .background(Color.black.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.appTextSecondary)
            Spacer()
            Text(value)
                .font(.headline)
                .foregroundColor(.appTextPrimary)
        }
    }

    private func drawGraph(context: GraphicsContext, size: CGSize) {
        let nodes = model.nodes
        guard !nodes.isEmpty else { return }

        var allX = nodes.map(\.x), allY = nodes.map(\.y)
        allX.append(contentsOf: model.waypointMarkers.map(\.x))
        allY.append(contentsOf: model.waypointMarkers.map(\.y))

        let minX = allX.min() ?? 0, maxX = allX.max() ?? 0
        let minY = allY.min() ?? 0, maxY = allY.max() ?? 0
        let spanX = max(maxX - minX, 1)
        let spanY = max(maxY - minY, 1)

        let padding: CGFloat = 24
        let usableW = size.width - padding * 2
        let usableH = size.height - padding * 2
        let scale = min(usableW / CGFloat(spanX), usableH / CGFloat(spanY))
        let drawnW = CGFloat(spanX) * scale
        let drawnH = CGFloat(spanY) * scale
        let offsetX = padding + (usableW - drawnW) / 2
        let offsetY = padding + (usableH - drawnH) / 2

        // y is flipped so north (larger y) draws toward the top of the canvas, matching the
        // backend's plot.png convention instead of SwiftUI's default y-down.
        func canvasPoint(_ x: Double, _ y: Double) -> CGPoint {
            CGPoint(
                x: offsetX + CGFloat(x - minX) * scale,
                y: size.height - (offsetY + CGFloat(y - minY) * scale)
            )
        }

        for edge in model.edges {
            guard let a = nodes.first(where: { $0.id == edge.fromNodeID }),
                  let b = nodes.first(where: { $0.id == edge.toNodeID }) else { continue }
            var path = Path()
            path.move(to: canvasPoint(a.x, a.y))
            path.addLine(to: canvasPoint(b.x, b.y))
            context.stroke(path, with: .color(.gray.opacity(0.6)), lineWidth: 1.5)
        }

        for node in nodes {
            let p = canvasPoint(node.x, node.y)
            let isCurrent = node.id == model.currentNodeID
            let r: CGFloat = isCurrent ? 7 : 5
            let rect = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
            context.fill(Path(ellipseIn: rect), with: .color(isCurrent ? .red : .blue))
        }

        for marker in model.waypointMarkers {
            let p = canvasPoint(marker.x, marker.y)
            let r: CGFloat = 9
            let rect = CGRect(x: p.x - r, y: p.y - r - 14, width: r * 2, height: r * 2)
            context.fill(Path(ellipseIn: rect), with: .color(.orange))
            context.draw(Text("\(marker.waypoint.index)").font(.caption2.bold()).foregroundColor(.white), at: CGPoint(x: rect.midX, y: rect.midY))
        }
    }
}

/// One "標記回到原點" press — a diagnostic marker only, same as `SensorTestLap` in
/// `SensorTestView`: it records how far off (x, y) currently is from true (0, 0), it does
/// NOT itself correct anything. Lets a walk cover several loops (out and back to the start)
/// in one continuous online-tracking session instead of needing to stop (which resets the
/// whole map) after every single loop.
struct TopoMapLap {
    let lapNumber: Int
    let x: Double
    let y: Double
    let timestamp: Double
    let distanceFromOrigin: Double
    let distanceFromPreviousLap: Double
    let nodeCountAtLap: Int
    let driftCorrectionCountAtLap: Int
}

@MainActor
final class TopoMapTestModel: ObservableObject {
    @Published var mode: TopoMapMode = .online
    @Published var onlineMethod: HeadingMethodID = .rotationVector

    @Published private(set) var isTracking = false
    @Published private(set) var nodeCount = 0
    @Published private(set) var edgeCount = 0
    @Published private(set) var stepCount = 0
    @Published private(set) var headingDegrees: Double = -1
    @Published private(set) var driftCorrectionCount = 0
    @Published private(set) var lastLoopEvent: LoopEvent?
    @Published private(set) var nodes: [PathNode] = []
    @Published private(set) var edges: [PathEdge] = []
    @Published private(set) var currentNodeID: String?
    @Published private(set) var waypointMarkers: [(waypoint: ComboWaypoint, x: Double, y: Double)] = []
    @Published private(set) var laps: [TopoMapLap] = []

    @Published private(set) var availableBatches: [SensorComboBatch] = []
    @Published var selectedBatchId: String?
    @Published var selectedOfflineMethod: HeadingMethodID = .rotationVector

    private let tracker = SensorComboTracker()
    private let map = TopologicalMap(mergeRadius: 3.0)
    private let loopDetector = LoopDetector()

    init() {
        tracker.onPathPoint = { [weak self] multiPoint in
            self?.handleOnline(multiPoint)
        }
        loopDetector.onLoopDetected = { [weak self] event in
            self?.lastLoopEvent = event
        }
        refreshAvailableBatches()
    }

    func refreshAvailableBatches() {
        availableBatches = PersistenceService.shared.load([SensorComboBatch].self, from: "sensor_combo_batches.json") ?? []
    }

    // MARK: - Online

    func start() {
        guard !isTracking, SensorComboTracker.isAvailable else { return }
        isTracking = true
        resetMapState()
        tracker.start()
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        tracker.stop()
    }

    private func resetMapState() {
        map.reset()
        loopDetector.reset()
        nodeCount = 0
        edgeCount = 0
        stepCount = 0
        headingDegrees = -1
        driftCorrectionCount = 0
        lastLoopEvent = nil
        nodes = []
        edges = []
        currentNodeID = nil
        waypointMarkers = []
        laps = []
    }

    /// See `TopoMapLap`'s doc comment — purely a diagnostic mark, applies no correction.
    /// Uses the currently-selected online method's own (drift-corrected-so-far) running
    /// position, same source `handleOnline` reads from, so the reported distance matches
    /// what's actually driving the map on screen.
    func markLap() {
        guard isTracking else { return }
        let pos = tracker.currentPositionsSnapshot()[onlineMethod] ?? (0, 0)
        let previous = laps.last.map { (x: $0.x, y: $0.y) } ?? (x: 0, y: 0)
        let dx = pos.x - previous.x
        let dy = pos.y - previous.y
        laps.append(TopoMapLap(
            lapNumber: laps.count + 1,
            x: pos.x,
            y: pos.y,
            timestamp: Date().timeIntervalSince1970 * 1000,
            distanceFromOrigin: (pos.x * pos.x + pos.y * pos.y).squareRoot(),
            distanceFromPreviousLap: (dx * dx + dy * dy).squareRoot(),
            nodeCountAtLap: nodeCount,
            driftCorrectionCountAtLap: driftCorrectionCount
        ))
    }

    /// Mirrors `NavigationSessionManager.handle(_:)` — same drift-correction wiring, just
    /// exposing more of the intermediate state (node/edge lists, correction count) for the
    /// debug UI, and pulling the (x, y) for whichever heading method is currently selected
    /// instead of always using one fixed tracker.
    private func handleOnline(_ multiPoint: MultiHeadingPoint) {
        guard let raw = multiPoint.positions[onlineMethod.rawValue] else { return }
        let heading = multiPoint.headings[onlineMethod.rawValue] ?? 0
        let point = PathPoint(x: raw.x, y: raw.y, headingDegrees: heading, timestamp: multiPoint.timestamp,
                               stepCount: multiPoint.stepCount, distanceMeters: multiPoint.distanceMeters)
        let (node, driftCorrection) = map.ingest(point)

        if let driftCorrection {
            tracker.applyDriftCorrection(method: onlineMethod, dx: driftCorrection.dx, dy: driftCorrection.dy)
            driftCorrectionCount += 1
        }

        currentNodeID = node.id
        stepCount = point.stepCount
        headingDegrees = point.headingDegrees
        nodes = map.nodes
        edges = map.edges
        nodeCount = nodes.count
        edgeCount = edges.count
        loopDetector.recordVisit(nodeID: node.id, at: point.timestamp)
    }

    // MARK: - Offline replay

    /// Replays a previously-recorded `SensorComboBatch` through a fresh `TopologicalMap` in
    /// one pass — no live sensors involved. Drift correction is applied to a local running
    /// offset (mirroring what would have happened live) so the replayed map matches what
    /// walking it live with the same method would have produced. Waypoints are matched to
    /// the replayed point with the closest timestamp to place their markers on the map.
    func buildOfflineMap() {
        guard let batchId = selectedBatchId, let batch = availableBatches.first(where: { $0.id == batchId }) else { return }
        let method = selectedOfflineMethod
        resetMapState()

        let replayMap = TopologicalMap(mergeRadius: 3.0)
        var offsetX = 0.0, offsetY = 0.0
        var correctedByTimestamp: [(t: Double, x: Double, y: Double)] = []
        var driftCorrections = 0

        for multiPoint in batch.points {
            guard let raw = multiPoint.positions[method.rawValue] else { continue }
            var x = raw.x + offsetX
            var y = raw.y + offsetY
            let heading = multiPoint.headings[method.rawValue] ?? 0
            let point = PathPoint(x: x, y: y, headingDegrees: heading, timestamp: multiPoint.timestamp,
                                   stepCount: multiPoint.stepCount, distanceMeters: multiPoint.distanceMeters)
            let (_, driftCorrection) = replayMap.ingest(point)
            if let driftCorrection {
                offsetX += driftCorrection.dx
                offsetY += driftCorrection.dy
                x += driftCorrection.dx
                y += driftCorrection.dy
                driftCorrections += 1
            }
            correctedByTimestamp.append((multiPoint.timestamp, x, y))
        }

        nodes = replayMap.nodes
        edges = replayMap.edges
        nodeCount = nodes.count
        edgeCount = edges.count
        currentNodeID = nodes.last?.id
        stepCount = batch.points.last?.stepCount ?? 0
        headingDegrees = batch.points.last?.headings[method.rawValue] ?? -1
        driftCorrectionCount = driftCorrections

        waypointMarkers = batch.waypoints.compactMap { waypoint in
            guard let nearest = correctedByTimestamp.min(by: { abs($0.t - waypoint.timestamp) < abs($1.t - waypoint.timestamp) }) else { return nil }
            return (waypoint, nearest.x, nearest.y)
        }
    }
}
