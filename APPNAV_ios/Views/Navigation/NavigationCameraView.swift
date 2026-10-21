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
    // Retake request (blurred / dark / tilted / occluded photo) — shown as a banner
    @State private var retakeHint: String?
    // "Several places look alike — turn and take one more" (server second_photo_hint)
    @State private var secondPhotoHint: String?
    /// PDR position of the photo that asked for a second one (the server only combines
    /// the two within 3 m, so the request lapses once the user walks further).
    @State private var secondPhotoAt: (x: Double, y: Double)?
    // First-use photo guide; reopen with the "?" button
    @AppStorage("navPhotoGuideSeen") private var photoGuideSeen = false
    @State private var showPhotoGuide = false

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
                    isLocalized: sensorSession.mapPosition != nil
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
                if let retakeHint {
                    Text("📷 \(retakeHint)")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.orange.opacity(0.9))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else if let secondPhotoHint {
                    Text("📷 \(secondPhotoHint)")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.blue.opacity(0.85))
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else if !hasArrived, !pendingArrival, pendingQuestion == nil, sensorSession.mapPosition == nil {
                    Button(action: startAtEntrance) {
                        Text("📍 我在入口，從這裡開始")
                            .font(.caption.weight(.semibold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Capsule())
                    }
                } else if !hasArrived, !pendingArrival, pendingQuestion == nil,
                          sensorSession.distanceSinceFix >= Self.photoReminderM {
                    Text("📍 已走 \(Int(sensorSession.distanceSinceFix)) m，拍一張照片校正位置")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.55))
                        .clipShape(Capsule())
                } else if !hasArrived, !pendingArrival, pendingQuestion == nil, !isPhoneUpright {
                    Text("📱 手機請拿直、對準走道前方")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.black.opacity(0.55))
                        .clipShape(Capsule())
                }
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
        .onReceive(sensorSession.$currentPoint) { point in
            followSensors(point)
        }
        .onChange(of: isPhoneUpright) { upright in
            if upright, retakeHint == PhotoQualityChecker.tiltHint { retakeHint = nil }
        }
        .sheet(isPresented: $showPhotoGuide) {
            PhotoGuideView { showPhotoGuide = false; photoGuideSeen = true }
        }
        .onAppear {
            if !photoGuideSeen {
                showPhotoGuide = true
                photoGuideSeen = true   // shown once, however it is dismissed
            }
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

            VStack(spacing: 8) {
                flashButton
                Button(action: { showPhotoGuide = true }) {
                    Image(systemName: "questionmark")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.white)
                        .frame(width: 34, height: 34)
                        .background(Color.white.opacity(0.2))
                        .clipShape(Circle())
                }
                .accessibilityLabel("拍照說明")
            }
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
              let pos = sensorSession.mapPosition else {
            return 0 // Not calibrated — fixed pointing up
        }
        let bearing = NodeCoordinateStore.shared.bearing(from: pos, to: targetCoord)
        var relative = bearing - sensorSession.mapHeading
        while relative > 180 { relative -= 360 }
        while relative < -180 { relative += 360 }
        return relative
    }

    /// On-device DINOv2 result for one photo: the top-K nodes the server fuses
    /// with its own grid matcher, plus the phone's DINO+PDR pick as a hint.
    private struct DINOResult {
        var hint: Int?
        var topK: [(nodeId: Int, similarity: Float)] = []
    }

    /// Run DINOv2 on-device localization and fuse with PDR position.
    private func dinoLocalize(image: UIImage) -> DINOResult {
        guard dinoReady else { return DINOResult() }
        let matches = DINOLocalizationService.shared.localize(image: image, topK: 10)
        guard !matches.isEmpty else { return DINOResult() }
        let candidates = matches.map { (nodeId: $0.nodeId, similarity: $0.similarity) }
        let hint = NodeCoordinateStore.shared.fuseDINOWithPDR(
            candidates: Array(candidates.prefix(5)),
            pdrPosition: sensorSession.mapPosition ?? sensorSession.pdrPosition
        )
        return DINOResult(hint: hint, topK: candidates)
    }

    /// Remind the user to take a photo after this much walking: the walk simulation kept
    /// the error near 1 m with a photo every 6–12 m (backend eval/report_figures walk_frequency).
    static let photoReminderM = 12.0
    /// Backend config SECOND_PHOTO_MAX_MOVE_M.
    static let secondPhotoMaxMoveM = 3.0

    private func startAtEntrance() {
        let entrance = NodeCoordinateStore.shared.nearestNode(to: (0, 0))
        sensorSession.startAtEntrance(nodeId: entrance?.nodeId)
    }

    private var isPhoneUpright: Bool {
        PhotoQualityChecker.isUpright(gravity: sensorSession.gravity)
    }

    /// Every sensor step: move the route-map dot, and re-plan when the user has walked
    /// off the planned route (PathFollower.needsReplan).
    private func followSensors(_ point: PathPoint?) {
        guard let point else { return }
        if let at = secondPhotoAt, hypot(point.x - at.x, point.y - at.y) > Self.secondPhotoMaxMoveM {
            secondPhotoHint = nil
            secondPhotoAt = nil
        }
        if let m = sensorSession.mapPosition {
            routeModel.ingest(mapPosition: m, headingDegrees: sensorSession.mapHeading, raw: point)
        } else {
            routeModel.ingest(point)
        }
        if routeModel.shouldReplan() {
            Task { await routeModel.load(sessionId: sessionId, fromPhone: true) }
        }
    }

    /// Blur / darkness check before uploading (same thresholds as the server).
    private func localQualityHint(_ image: UIImage?) -> String? {
        guard let cg = image?.cgImage, let m = PhotoQualityChecker.measure(cg) else { return nil }
        return PhotoQualityChecker.hint(for: m)
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

    private func applyTurn(_ response: TurnResponse, rawAtCapture: (x: Double, y: Double)? = nil) {
        retakeHint = response.retakeHint
        secondPhotoHint = response.secondPhotoHint
        secondPhotoAt = response.secondPhotoHint != nil ? rawAtCapture : nil
        // A localized photo corrects the shared map position (and, between two good
        // fixes, the heading offset and step length) — see MapPositionEstimator.
        // When the server is unsure and asks for a second photo, wait for the combined answer.
        var fixNote = response.secondPhotoHint != nil ? "，等第二張照片再校正" : ""
        var fixAccepted = false
        if response.retakeHint == nil, response.secondPhotoHint == nil,
           let locNodeId = response.correctedNodeId, let raw = rawAtCapture {
            let node: (x: Double, y: Double)?
            if let ex = response.estimatedX, let ey = response.estimatedY {
                node = (ex, ey)
            } else {
                node = NodeCoordinateStore.shared.coordinate(for: locNodeId)
            }
            if let node {
                switch sensorSession.applyPhotoFix(node: node, nodeId: locNodeId,
                                                   confidence: response.correctedConfidence ?? 0, raw: raw) {
                case .accepted:
                    fixAccepted = true
                    // move the route-map dot now, not on the next step
                    if let m = sensorSession.mapPosition {
                        routeModel.ingest(mapPosition: m, headingDegrees: sensorSession.mapHeading, raw: nil)
                    }
                case .rejectedLowConfidence: fixNote = "，信心不足未校正位置"
                case .rejectedJump: fixNote = "，跳太遠未校正位置"
                }
            }
        }
        if let loc = response.correctedLocation {
            if let conf = response.correctedConfidence {
                currentLocation = "\(loc)（比對信心 \(Int((conf * 100).rounded()))%\(fixNote)）"
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
        if response.correctedNodeId != nil {
            // Only a fix the phone actually applied re-anchors the route at the server's
            // node; otherwise (gallery photo, rejected jump, low confidence) re-plan from
            // where the phone thinks it is.
            let replanFromPhone = !fixAccepted
            Task { await routeModel.load(sessionId: sessionId, fromPhone: replanFromPhone) }
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
        guard isPhoneUpright else {
            retakeHint = PhotoQualityChecker.tiltHint
            return
        }
        isUploading = true
        errorMessage = nil
        retakeHint = nil
        // Sensor state at the shutter, not when the server answers seconds later.
        let raw = sensorSession.pdrPosition
        let mapPosition = sensorSession.mapPosition
        let distance = sensorSession.distanceSinceFix
        let fixNode = sensorSession.lastFixNodeId
        let heading = sensorSession.mapHeading   // compass + offset learned from photo fixes

        camera.capturePhoto { data in
            guard let data else {
                isUploading = false
                errorMessage = "拍照失敗"
                return
            }
            let image = UIImage(data: data)
            lastCapturedImage = image
            if let hint = localQualityHint(image) {
                retakeHint = hint
                isUploading = false
                return
            }
            let dino = image.map { dinoLocalize(image: $0) } ?? DINOResult()
            Task {
                await upload(imageData: data, dino: dino, raw: raw, mapPosition: mapPosition,
                             distanceSinceFix: distance, fixNode: fixNode, heading: heading, fromGallery: false)
            }
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
            if let hint = localQualityHint(image) {
                await MainActor.run {
                    retakeHint = hint
                    isUploading = false
                    galleryItem = nil
                }
                return
            }
            let dino = image.map { dinoLocalize(image: $0) } ?? DINOResult()
            // A gallery photo was taken some other time: the live position / heading
            // don't describe it, so none are sent and it can't correct the position.
            await upload(imageData: data, dino: dino, raw: nil, mapPosition: nil,
                         distanceSinceFix: nil, fixNode: nil, heading: nil, fromGallery: true)
        }
    }

    /// `raw` / `mapPosition` / `heading` are the sensor state when the photo was taken
    /// (nil for gallery photos). `mapPosition` is nil until the first photo fix — the raw
    /// PDR frame starts wherever tracking began and means nothing on the map.
    private func upload(imageData: Data, dino: DINOResult, raw: (x: Double, y: Double)?,
                        mapPosition: (x: Double, y: Double)?, distanceSinceFix: Double?,
                        fixNode: Int?, heading: Double?, fromGallery: Bool) async {
        do {
            let response = try await NavigationAPI.shared.uploadPhoto(
                sessionId: sessionId,
                imageData: imageData,
                dinoNodeHint: dino.hint,
                dinoTopK: dino.topK,
                pdrX: raw?.x,
                pdrY: raw?.y,
                mapPosition: mapPosition,
                distanceSinceFix: mapPosition != nil ? distanceSinceFix : nil,
                fixNode: mapPosition != nil ? fixNode : nil,
                heading: heading,
                fromGallery: fromGallery
            )
            await MainActor.run {
                applyTurn(response, rawAtCapture: raw)
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
                    secondPhotoHint = nil
                    secondPhotoAt = nil
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

/// How to take a navigation photo — shown on first use and from the "?" button.
/// The reference map was photographed upright, facing along the aisle, so photos taken
/// the same way match it best (item 8 of the localization review).
private struct PhotoGuideView: View {
    let onDone: () -> Void

    private let tips: [(icon: String, title: String, detail: String)] = [
        ("iphone", "手機拿直", "直立拿在胸前高度，不要朝地板或天花板拍。畫面歪斜時下方會提示。"),
        ("arrow.up.forward", "對準走道前方", "沿著走道往前拍，拍到走道的延伸，而不是貼著貨架拍單一商品。"),
        ("signpost.right", "拍到招牌或吊牌", "走道編號、天花板吊牌、區域招牌是最好的定位線索。"),
        ("person.2.slash", "避開人和推車", "前方有人或推車擋住時，等一下或換個角度；被擋住的照片不會用來校正位置。"),
        ("hand.raised", "停一下再按", "走路中拍容易模糊；停一秒、拿穩再按拍照。太暗時開閃光燈。"),
    ]

    var body: some View {
        NavigationStack {
            List(tips, id: \.title) { tip in
                HStack(alignment: .top, spacing: 14) {
                    Image(systemName: tip.icon)
                        .font(.title3)
                        .foregroundStyle(.blue)
                        .frame(width: 28)
                    VStack(alignment: .leading, spacing: 4) {
                        Text(tip.title).font(.headline)
                        Text(tip.detail).font(.subheadline).foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            }
            .navigationTitle("怎麼拍定位照片")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("知道了", action: onDone)
                }
            }
        }
    }
}
