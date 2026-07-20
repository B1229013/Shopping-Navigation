import AVFoundation
import PhotosUI
import SwiftUI

/// Full-screen camera + guidance UI. Mirrors Android's `NavigationScreen` (photo capture,
/// gallery pick, ASK/ARRIVED flows) — the location/AR-store-marker overlay from the Android
/// version is not ported since it belongs to the out-of-scope "附近商店" Places feature.
struct NavigationCameraView: View {
    let sessionId: String
    let goal: String
    let onExit: () -> Void

    @StateObject private var camera = CameraController()
    @ObservedObject private var sensorSession = NavigationSessionManager.shared

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

    init(sessionId: String, goal: String, initialGuidance: String, onExit: @escaping () -> Void) {
        self.sessionId = sessionId
        self.goal = goal
        self.onExit = onExit
        _guidance = State(initialValue: initialGuidance)
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if hasCameraPermission {
                CameraPreviewView(controller: camera)
                    .ignoresSafeArea()
            }

            VStack(spacing: 12) {
                HStack(alignment: .top, spacing: 10) {
                    if let lastCapturedImage {
                        lastPhotoThumbnail(lastCapturedImage)
                    }
                    statusCard
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
            sensorSession.onLoopDetected = { event in
                loopBanner = "偵測到您可能在原地繞圈（第 \(event.repeatCount) 次經過同一節點）"
            }
            sensorSession.start()
        }
        .onDisappear {
            camera.stop()
            let snapshot = sensorSession.exportSnapshot()
            sensorSession.stop()
            Task {
                try? await NavigationAPI.shared.uploadSensorMap(sessionId: sessionId, snapshot: snapshot)
            }
        }
    }

    // MARK: - Status card

    private var statusCard: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(hasArrived ? "已到達目標！" : (pendingArrival ? "請確認是否到達目標" : "導航中"))
                    .font(.headline)
                    .foregroundColor(.white)
                Text("目標　\(goal)")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.85))
                    .lineLimit(2)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            flashButton
        }
        .padding(14)
        .background(Color.blue.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 12))
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

    // MARK: - Guidance panel

    private var guidancePanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Text("導航指引")
                    .font(.caption.weight(.bold))
                    .foregroundColor(.green.opacity(0.8))
                Text(guidance)
                    .font(.subheadline)
                    .foregroundColor(.white)

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
        .background(Color.green.opacity(0.25))
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
                        Text("確認到達").font(.subheadline.weight(.bold))
                    }
                }
                .frame(maxWidth: .infinity).frame(height: 46)
                .buttonStyle(.borderedProminent).tint(.green)
                .disabled(isConfirming)

                HStack(spacing: 8) {
                    Button("不是目標") { confirmArrival(kind: "false_positive") }
                        .frame(maxWidth: .infinity).frame(height: 42)
                        .buttonStyle(.borderedProminent).tint(.gray)
                        .disabled(isConfirming)
                    Button("同類非目標") { confirmArrival(kind: "wrong_instance") }
                        .frame(maxWidth: .infinity).frame(height: 42)
                        .buttonStyle(.borderedProminent).tint(.purple)
                        .disabled(isConfirming)
                }
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
                    .onChange(of: galleryItem) { _, newItem in
                        uploadFromGallery(newItem)
                    }
                }
            }
        }
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

    private func applyTurn(_ response: TurnResponse) {
        guidance = response.guidance
        switch response.action {
        case "ARRIVED":
            pendingArrival = true
            pendingQuestion = nil
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
            lastCapturedImage = UIImage(data: data)
            Task { await upload(imageData: data) }
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
            await MainActor.run { lastCapturedImage = UIImage(data: data) }
            await upload(imageData: data)
        }
    }

    private func upload(imageData: Data) async {
        do {
            let response = try await NavigationAPI.shared.uploadPhoto(sessionId: sessionId, imageData: imageData)
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
