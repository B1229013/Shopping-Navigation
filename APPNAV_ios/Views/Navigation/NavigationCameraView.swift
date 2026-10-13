import AVFoundation
import PhotosUI
import SwiftUI

/// Full-screen camera + guidance UI. Mirrors Android's `NavigationScreen` (photo capture,
/// gallery pick, ASK/ARRIVED flows) — the location/AR-store-marker overlay from the Android
/// version is not ported since it belongs to the out-of-scope "附近商店" Places feature.
struct NavigationCameraView: View {
    let sessionId: String
    let goal: String
    let place: String?
    let onExit: () -> Void

    @StateObject private var camera = CameraController()
    @StateObject private var routeModel = RouteFollowerModel()
    @ObservedObject private var sensorSession = NavigationSessionManager.shared
    @State private var showFullMap = false

    @State private var hasCameraPermission = false
    @State private var guidance: String
    @State private var pendingQuestion: String?
    @State private var answerText = ""
    @State private var isUploading = false
    @State private var isAnswering = false
    @State private var isConfirming = false
    @State private var hasArrived = false
    @State private var pendingArrival = false
    @State private var errorMessage: String?
    @State private var galleryItem: PhotosPickerItem?
    @State private var loopBanner: String?
    @State private var lastCapturedImage: UIImage?
    @State private var navPhase: String = "shopping"
    @State private var goalItems: [String] = []
    @State private var visitedItems: Set<String> = []
    @State private var currentTarget: String?

    // AR arrow: target node — set from route plan at session start, updated per photo
    @State private var targetNodeId: Int?
    @State private var dinoReady = false
    // PDR ↔ reference frame calibration: recorded when DINOv2 localizes
    @State private var pdrOffset: (dx: Double, dy: Double)?

    init(sessionId: String, goal: String, place: String? = nil,
         initialGuidance: String, initialTargetNodeId: Int? = nil,
         onExit: @escaping () -> Void) {
        self.sessionId = sessionId
        self.goal = goal
        self.place = place
        self.onExit = onExit
        _guidance = State(initialValue: initialGuidance)
        _targetNodeId = State(initialValue: initialTargetNodeId)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if hasCameraPermission {
                CameraPreviewView(controller: camera)
                    .ignoresSafeArea()
            }

            // AR directional arrow — top-right, always visible during navigation
            if !hasArrived, !pendingArrival {
                DirectionArrowView(
                    relativeBearingDegrees: arrowBearing,
                    isLocalized: pdrOffset != nil
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                .padding(.trailing, 12)
                .padding(.top, 130)
                .allowsHitTesting(false)
            }

            VStack(spacing: 12) {
                HStack(alignment: .top, spacing: 10) {
                    if let lastCapturedImage {
                        lastPhotoThumbnail(lastCapturedImage)
                    }
                    statusCard
                }
                if !goalItems.isEmpty {
                    shoppingChecklist
                }
                if camera.isTorchAvailable && !camera.isTorchOn {
                    Text("光線不足嗎？點擊右上角燈泡圖示開啟閃光燈輔助拍照")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.75))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                Spacer()
                if let loopBanner {
                    Text(loopBanner)
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.orange.opacity(0.85))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                routePanel
                guidancePanel
                actionButtons
            }
            .padding(.horizontal, 16)
            .padding(.top, 8)
            .padding(.bottom, 12)
        }
        .onAppear {
            requestCameraAccessIfNeeded()
            camera.configure()
            camera.start()
            loopBanner = nil
            sensorSession.onLoopDetected = nil
            sensorSession.start(place: place)
            Task { await routeModel.load(sessionId: sessionId) }

            // Load DINOv2 model + node coordinates for on-device fusion
            Task {
                let dino = DINOLocalizationService.shared
                if !dino.isLoaded { try? dino.load() }
                let coords = NodeCoordinateStore.shared
                if !coords.isLoaded { coords.load() }
                dinoReady = dino.isLoaded && coords.isLoaded
            }
        }
        .onDisappear {
            camera.stop()
            let snapshot = sensorSession.exportSnapshot()
            sensorSession.stop()
            // Free DINOv2 model + embeddings from memory
            DINOLocalizationService.shared.unload()
            NodeCoordinateStore.shared.unload()
            Task {
                try? await NavigationAPI.shared.uploadSensorMap(sessionId: sessionId, snapshot: snapshot)
            }
        }
    }

    // MARK: - Status card

    private var phaseLabel: String {
        switch navPhase {
        case "checkout": return "前往結帳"
        case "exit": return "前往出口"
        case "done": return "導航完成"
        default: return "找商品中"
        }
    }

    private var statusTitle: String {
        if hasArrived { return navPhase == "done" ? "導航完成！" : "已到達！" }
        if pendingArrival { return "請確認是否到達" }
        return "導航中"
    }

    private var statusCard: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(statusTitle)
                        .font(.headline)
                        .foregroundColor(.white)
                    Text(phaseLabel)
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            navPhase == "checkout" ? Color.orange.opacity(0.7) :
                            navPhase == "exit" ? Color.purple.opacity(0.7) :
                            navPhase == "done" ? Color.green.opacity(0.7) :
                            Color.blue.opacity(0.5)
                        )
                        .clipShape(Capsule())
                }
                Text(navPhase == "checkout" ? "前往收銀台" :
                     navPhase == "exit" ? "前往出口" :
                     currentTarget != nil ? "正在找　\(currentTarget!)" : "目標　\(goal)")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.85))
                    .lineLimit(2)
                // The localization result (with match confidence) is only shown here —
                // the guidance text no longer repeats it.
                if let currentLocation {
                    Text("📍 \(currentLocation)")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.7))
                        .lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            flashButton
        }
        .padding(14)
        .background(
            navPhase == "checkout" ? Color.orange.opacity(0.45) :
            navPhase == "exit" ? Color.purple.opacity(0.45) :
            Color.blue.opacity(0.55)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var shoppingChecklist: some View {
        HStack(spacing: 8) {
            ForEach(goalItems, id: \.self) { item in
                let isDone = visitedItems.contains(item)
                let isCurrent = item == currentTarget && !isDone
                HStack(spacing: 4) {
                    Image(systemName: isDone ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(isDone ? .green : isCurrent ? .yellow : .white.opacity(0.5))
                    Text(item)
                        .font(.caption2.weight(isCurrent ? .bold : .regular))
                        .foregroundColor(isDone ? .green.opacity(0.7) : isCurrent ? .yellow : .white.opacity(0.6))
                        .strikethrough(isDone)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    isCurrent ? Color.yellow.opacity(0.15) :
                    isDone ? Color.green.opacity(0.1) :
                    Color.white.opacity(0.05)
                )
                .clipShape(Capsule())
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var flashButton: some View {
        Button(action: { camera.toggleTorch() }) {
            Image(systemName: camera.isTorchOn ? "bolt.fill" : "bolt.slash.fill")
                .font(.system(size: 15, weight: .bold))
                .foregroundColor(.white)
                .frame(width: 34, height: 34)
                .background(camera.isTorchOn ? Color.yellow.opacity(0.85) : Color.white.opacity(0.2))
                .clipShape(Circle())
        }
        .disabled(!camera.isTorchAvailable)
        .opacity(camera.isTorchAvailable ? 1 : 0.35)
    }

    // MARK: - Last photo thumbnail

    private func lastPhotoThumbnail(_ image: UIImage) -> some View {
        Image(uiImage: image)
            .resizable()
            .scaledToFill()
            .frame(width: 56, height: 56)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .stroke(Color.white.opacity(0.5), lineWidth: 1)
            )
            .shadow(radius: 3)
    }

    // MARK: - Route panel (editor-map path following)

    @ViewBuilder
    private var routePanel: some View {
        if let route = routeModel.route {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("🧭 \(routeModel.state?.prompt ?? "定位中…")")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(.white)
                    Spacer()
                    if let st = routeModel.state {
                        Text("剩 \(Int(st.remainingM.rounded())) m")
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                RouteMapView(route: route, state: routeModel.state)
                    .frame(height: 150)
                    .onTapGesture { showFullMap = true }
            }
            .padding(10)
            .background(Color.cyan.opacity(0.18))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        } else if let msg = routeModel.statusMessage {
            Text("🗺 \(msg)")
                .font(.caption)
                .foregroundColor(.white.opacity(0.8))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Guidance panel

    private var guidancePanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Text("導航指引")
                    .font(.caption.weight(.bold))
                    .foregroundColor(.white.opacity(0.7))
                Text(guidance)
                    .font(.subheadline)
                    .foregroundColor(.white)

                if let mapInstruction {
                    Text("🗺 \(mapInstruction)")
                        .font(.subheadline.weight(.medium))
                        .foregroundColor(.cyan)
                }

                if let pendingQuestion, !hasArrived {
                    Text(pendingQuestion)
                        .font(.subheadline.weight(.medium))
                        .foregroundColor(.orange)
                    HStack {
                        TextField("輸入回答...", text: $answerText)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit { submitAnswer() }
                        Button(action: submitAnswer) {
                            if isAnswering {
                                ProgressView().tint(.white)
                            } else {
                                Image(systemName: "paperplane.fill")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                        .disabled(answerText.trimmingCharacters(in: .whitespaces).isEmpty || isAnswering)
                    }
                }

                if let errorMessage {
                    Text(errorMessage)
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
        }
        .frame(maxHeight: 180)
        .background(Color.gray.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Bottom action buttons

    @ViewBuilder
    private var actionButtons: some View {
        if pendingArrival {
            VStack(spacing: 6) {
                Button(action: { confirmArrival(kind: "confirmed") }) {
                    if isConfirming {
                        ProgressView().tint(.white)
                    } else {
                        Text(navPhase == "checkout" ? "確認已結帳" :
                             navPhase == "exit" ? "確認已到出口" :
                             "確認到達").font(.subheadline.weight(.bold))
                    }
                }
                .frame(maxWidth: .infinity).frame(height: 46)
                .buttonStyle(.borderedProminent).tint(.green)
                .disabled(isConfirming)

                Button("不是目標") { confirmArrival(kind: "false_positive") }
                    .frame(maxWidth: .infinity).frame(height: 42)
                    .buttonStyle(.borderedProminent).tint(.gray)
                    .disabled(isConfirming)
            }
        } else {
            HStack(spacing: 8) {
                Button(hasArrived ? "完成" : "結束導航") { onExit() }
                    .frame(maxWidth: .infinity).frame(height: 46)
                    .buttonStyle(.borderedProminent)
                    .tint(hasArrived ? .green : .red)

                if !hasArrived && pendingQuestion == nil {
                    Button(action: captureAndUpload) {
                        if isUploading {
                            ProgressView().tint(.white)
                        } else {
                            Text("拍照分析").font(.subheadline.weight(.bold))
                        }
                    }
                    .frame(maxWidth: .infinity).frame(height: 46)
                    .buttonStyle(.borderedProminent).tint(.green)
                    .disabled(isUploading)

                    PhotosPicker(selection: $galleryItem, matching: .images) {
                        Text("選擇照片").font(.subheadline.weight(.bold))
                    }
                    .frame(maxWidth: .infinity).frame(height: 46)
                    .buttonStyle(.borderedProminent).tint(.blue)
                    .disabled(isUploading)
                    .onChange(of: galleryItem) { newItem in
                        uploadFromGallery(newItem)
                    }
                }
            }
        }
    }

    // MARK: - AR arrow computed properties

    /// Bearing the arrow should display.
    /// Before localization (no target node): points north (compensates for phone heading).
    /// After localization: points toward the target node.
    private var arrowBearing: Double {
        guard let nodeId = targetNodeId,
              let targetCoord = NodeCoordinateStore.shared.coordinate(for: nodeId),
              pdrOffset != nil else {
            return 0 // Not calibrated — fixed pointing up
        }
        let pos = adjustedPosition
        let bearing = NodeCoordinateStore.shared.bearing(from: pos, to: targetCoord)
        var relative = bearing - sensorSession.currentHeading
        while relative > 180 { relative -= 360 }
        while relative < -180 { relative += 360 }
        return relative
    }

    /// Run DINOv2 on-device localization and fuse with PDR position.
    /// Returns the best-match node ID as a hint for the backend.
    private func dinoFusedNodeHint(image: UIImage) -> Int? {
        guard dinoReady else { return nil }
        let matches = DINOLocalizationService.shared.localize(image: image, topK: 5)
        guard !matches.isEmpty else { return nil }
        let candidates = matches.map { (nodeId: $0.nodeId, similarity: $0.similarity) }
        return NodeCoordinateStore.shared.fuseDINOWithPDR(
            candidates: candidates, pdrPosition: adjustedPosition
        )
    }

    /// Current position in the reference coordinate frame.
    private var adjustedPosition: (x: Double, y: Double) {
        let pdr = sensorSession.pdrPosition
        guard let offset = pdrOffset else { return pdr }
        return (x: pdr.x + offset.dx, y: pdr.y + offset.dy)
    }

    // MARK: - Actions

    private func requestCameraAccessIfNeeded() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            hasCameraPermission = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { hasCameraPermission = granted }
            }
        default:
            hasCameraPermission = false
        }
    }

    @State private var currentLocation: String?
    @State private var mapInstruction: String?

    private func applyTurn(_ response: TurnResponse) {
        // Calibrate PDR offset when backend confirms localization
        if let locNodeId = response.correctedNodeId,
           let refCoord = NodeCoordinateStore.shared.coordinate(for: locNodeId) {
            let pdr = sensorSession.pdrPosition
            pdrOffset = (dx: refCoord.x - pdr.x, dy: refCoord.y - pdr.y)
        }
        if let loc = response.correctedLocation {
            if let conf = response.correctedConfidence {
                currentLocation = "\(loc)（比對信心 \(Int((conf * 100).rounded()))%）"
            } else {
                currentLocation = loc
            }
        }
        // The map's own verdict for this photo, independent of the VLM prose
        mapInstruction = response.nextInstruction
        // Navigation phase (shopping → checkout → exit → done) drives the status card
        // and the arrival wording, so apply it before the action is interpreted.
        var phaseChanged = false
        if let phase = response.phase, phase != navPhase {
            navPhase = phase
            phaseChanged = true
        }
        // A confident photo fix re-anchors the dead-reckoned position at that
        // waypoint (cancelling drift). A weak fix (keyword localization can jump
        // between nodes on the same photo) only re-plans from where the phone
        // thinks it is, so the dot isn't teleported to a wrong node.
        if response.correctedNodeId != nil {
            let confident = (response.correctedConfidence ?? 0) >= 0.6
            Task { await routeModel.load(sessionId: sessionId, fromPhone: !confident) }
        } else if phaseChanged {
            // The phase moved on (shopping → checkout → exit) but this photo wasn't
            // localized: the path endpoint now plans a different leg, so re-plan from
            // where the phone thinks it is instead of following the stale leg.
            Task { await routeModel.load(sessionId: sessionId, fromPhone: true) }
        }
        if let items = response.goalItemNames, !items.isEmpty {
            goalItems = items
        }
        if let visited = response.visitedItemNames {
            visitedItems = Set(visited)
        }
        currentTarget = response.currentTargetName
        // Use the backend's goal node for the AR arrow direction
        if let goalNid = response.goalNodeId {
            targetNodeId = goalNid
        }
        guidance = response.guidance
        switch response.action {
        case "ARRIVED":
            if navPhase == "done" {
                hasArrived = true
                pendingArrival = false
            } else {
                pendingArrival = true
            }
            pendingQuestion = nil
            targetNodeId = nil
        case "ASK":
            pendingQuestion = response.question
        case "MOVE":
            pendingQuestion = nil
            pendingArrival = false
        default:
            break
        }
    }

    private func captureAndUpload() {
        guard !isUploading, !hasArrived, !pendingArrival, pendingQuestion == nil else { return }
        isUploading = true
        errorMessage = nil

        camera.capturePhoto { data in
            guard let data else {
                isUploading = false
                errorMessage = "拍照失敗"
                return
            }
            let image = UIImage(data: data)
            lastCapturedImage = image
            let hint = image.flatMap { dinoFusedNodeHint(image: $0) }
            Task { await upload(imageData: data, dinoHint: hint) }
        }
    }

    private func uploadFromGallery(_ item: PhotosPickerItem?) {
        guard let item, !isUploading, !hasArrived, !pendingArrival, pendingQuestion == nil else { return }
        isUploading = true
        errorMessage = nil

        Task {
            guard let data = try? await item.loadTransferable(type: Data.self) else {
                await MainActor.run {
                    isUploading = false
                    errorMessage = "無法讀取照片"
                }
                return
            }
            let image = UIImage(data: data)
            await MainActor.run { lastCapturedImage = image }
            let hint = image.flatMap { dinoFusedNodeHint(image: $0) }
            await upload(imageData: data, dinoHint: hint)
        }
    }

    private func upload(imageData: Data, dinoHint: Int? = nil) async {
        let pdr = sensorSession.pdrPosition
        let heading = sensorSession.currentHeading
        do {
            let response = try await NavigationAPI.shared.uploadPhoto(
                sessionId: sessionId,
                imageData: imageData,
                dinoNodeHint: dinoHint,
                pdrX: pdr.x,
                pdrY: pdr.y,
                heading: heading
            )
            await MainActor.run {
                applyTurn(response)
                isUploading = false
                galleryItem = nil
            }
        } catch {
            await MainActor.run {
                errorMessage = "上傳失敗: \(error.localizedDescription)"
                isUploading = false
                galleryItem = nil
            }
        }
    }

    private func submitAnswer() {
        let answer = answerText.trimmingCharacters(in: .whitespaces)
        guard !answer.isEmpty, !isAnswering else { return }
        isAnswering = true
        errorMessage = nil

        Task {
            do {
                let response = try await NavigationAPI.shared.postAnswer(sessionId: sessionId, answer: answer)
                await MainActor.run {
                    applyTurn(response)
                    answerText = ""
                    isAnswering = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = "回答提交失敗: \(error.localizedDescription)"
                    isAnswering = false
                }
            }
        }
    }

    private func confirmArrival(kind: String) {
        guard !isConfirming else { return }
        isConfirming = true
        errorMessage = nil

        Task {
            do {
                let response = try await NavigationAPI.shared.confirmArrival(sessionId: sessionId, kind: kind)
                await MainActor.run {
                    pendingArrival = false
                    guidance = response.guidance
                    // POST /confirm is the ONLY endpoint that advances the phase, and
                    // reaching "done" also sets s.arrived, which makes a later POST
                    // /photo 409 — so if we drop phase here the client can never see
                    // "done" and origin/main's 導航完成 UI is unreachable.
                    if let phase = response.phase, phase != navPhase {
                        navPhase = phase
                    }
                    if response.action == "ARRIVED" {
                        hasArrived = true
                    } else if response.action == "MOVE" {
                        pendingQuestion = nil
                    }
                    isConfirming = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = "確認失敗: \(error.localizedDescription)"
                    isConfirming = false
                }
            }
        }
    }
}
