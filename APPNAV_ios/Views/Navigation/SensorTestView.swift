import Combine
import HealthKit
import SwiftUI

/// Standalone pedometer/PDR readout — deliberately bypasses NavigationSessionManager
/// and the whole backend-connected flow so accuracy can be tested with just the phone
/// (no Mac, no network, no session) while walking. Reuses MotionPathTracker directly
/// so this shows exactly the same dead-reckoning math the real navigation feature uses.
///
/// Every walked point is written to disk immediately (not just held in memory), so a
/// batch is never lost — pressing 開始測試 always starts a new batch on top of whatever
/// was recorded before, it never discards it. 上傳並繪圖 uploads any batch (current or an
/// old one from a previous app launch) to the backend for plotting, and can be retried
/// any time since the batch stays on disk either way.
struct SensorTestView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = SensorTestModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    VStack(spacing: 16) {
                        metricRow(label: "狀態", value: model.isTracking ? "追蹤中" : "已停止")
                        metricRow(label: "步數", value: "\(model.stepCount)")
                        metricRow(label: "累積走過距離", value: String(format: "%.2f 公尺", model.distanceMeters))
                        metricRow(label: "目前朝向", value: model.headingDegrees >= 0 ? String(format: "%.0f°", model.headingDegrees) : "—")
                        Divider()
                        metricRow(label: "目前座標", value: String(format: "(%.2f, %.2f)", model.x, model.y))
                        metricRow(label: "距離起點（誤差）", value: String(format: "%.2f 公尺", model.driftFromOrigin))
                    }
                    .padding(20)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    if let laps = model.currentBatch?.laps, !laps.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("這輪已記錄的圈數")
                                .font(.subheadline.weight(.semibold))
                                .foregroundColor(.appTextPrimary)
                            ForEach(laps, id: \.lapNumber) { lap in
                                HStack {
                                    Text("第 \(lap.lapNumber) 圈")
                                        .foregroundColor(.appTextSecondary)
                                    Spacer()
                                    VStack(alignment: .trailing, spacing: 2) {
                                        Text("跟起點差 \(String(format: "%.2f", lap.distanceFromOrigin)) 公尺")
                                        Text("跟上一圈差 \(String(format: "%.2f", lap.distanceFromPreviousLap)) 公尺")
                                            .foregroundColor(.appTextTertiary)
                                    }
                                    .font(.caption)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(16)
                        .background(Color.appSurface)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    }

                    if !MotionPathTracker.isAvailable {
                        Text("這台裝置不支援計步器（模擬器一定會顯示這個）")
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("磁力計校正（進階／選用，一般不需要）")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.appTextPrimary)
                        Text("方向精準度現在主要靠密集取樣自動處理，不需要使用者做任何動作。這裡的手動校正只是額外的實驗工具，留給我們自己測試/比較用，預設關閉。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)

                        Toggle("套用校正修正方向", isOn: $model.useCalibrationCorrection)
                            .font(.caption)

                        Button {
                            model.calibrateMagnetometer()
                        } label: {
                            if model.isCalibrating {
                                Text("校準中…已收集 \(model.calibrationSampleCount) 筆（12 秒）")
                            } else {
                                Text("開始校準（請拿著手機朝各方向緩慢畫 8 字形轉動）")
                            }
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)
                        .disabled(model.isCalibrating || model.isTracking)

                        Text("如果要測校正效果：先校準，再打開上面的開關才會套用。如果套用後誤差反而變更大，代表方向修正的正負號可能相反，關掉開關，跟我說一聲我再修。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    Button(model.isTracking ? "停止測試" : "開始新的一輪（前面幾輪都還在，不會被清掉）") {
                        if model.isTracking {
                            model.stop()
                        } else {
                            model.startNewBatch()
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(model.isTracking ? Color.red : Color.appAccent)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .disabled(!MotionPathTracker.isAvailable)

                    Button("標記回到起點（第 \((model.currentBatch?.laps.count ?? 0) + 1) 圈）") {
                        model.markLap()
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(Color.orange)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .disabled(!model.isTracking)

                    if let uploadError = model.uploadError {
                        Text(uploadError)
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                    }

                    VStack(alignment: .leading, spacing: 10) {
                        Text("本機所有紀錄（手機上永久保存，不會因為按開始新一輪而消失）")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.appTextPrimary)

                        if model.batches.isEmpty {
                            Text("目前還沒有任何紀錄")
                                .font(.caption)
                                .foregroundColor(.appTextTertiary)
                        }

                        ForEach(model.batches.reversed()) { batch in
                            batchRow(batch)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    VStack(alignment: .leading, spacing: 10) {
                        Text("從 Apple Watch 訓練匯入（GPS 路徑，僅限戶外訓練有收到 GPS 時才有資料）")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.appTextPrimary)

                        Button {
                            model.loadRecentWorkouts()
                        } label: {
                            if model.isLoadingWorkouts {
                                ProgressView()
                            } else {
                                Text("查詢最近 3 天的訓練紀錄")
                            }
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)

                        if let healthKitError = model.healthKitError {
                            Text(healthKitError)
                                .font(.caption)
                                .foregroundColor(.red)
                        }

                        ForEach(model.workouts, id: \.uuid) { workout in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(workoutDisplay(workout))
                                        .font(.caption.weight(.medium))
                                        .foregroundColor(.appTextPrimary)
                                    Text(String(format: "%.0f 分鐘", workout.duration / 60))
                                        .font(.caption2)
                                        .foregroundColor(.appTextSecondary)
                                }
                                Spacer()
                                if model.isImporting(workoutId: workout.uuid) {
                                    ProgressView()
                                } else {
                                    Button("匯入這筆") {
                                        model.importWorkoutRoute(workout)
                                    }
                                    .font(.caption.weight(.semibold))
                                    .foregroundColor(.blue)
                                }
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    Text("測試方法：按開始新的一輪，走一圈回原地按「標記回到起點」，可以重複繞多圈。每一輪的資料一走出來就直接存在手機上，不管有沒有按上傳都不會不見。上傳只是把某一輪送到後端畫圖，可以隨時針對任何一輪重新上傳，需要連得上後端。")
                        .font(.caption)
                        .foregroundColor(.appTextTertiary)
                        .multilineTextAlignment(.center)
                }
                .padding(20)
                .padding(.top, 12)
            }
            .background(Color.appBg)
            .navigationTitle("計步器測試")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .onDisappear { model.stop() }
    }

    private func workoutDisplay(_ workout: HKWorkout) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "M/d HH:mm"
        return "\(workout.workoutActivityType.displayName) · \(formatter.string(from: workout.startDate))"
    }

    private func batchRow(_ batch: SensorTestModel.SensorTestBatch) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(batch.startedAtDisplay + (batch.sourceLabel.map { " · \($0)" } ?? ""))
                    .font(.caption.weight(.medium))
                    .foregroundColor(.appTextPrimary)
                Spacer()
                if model.isUploading(batchId: batch.id) {
                    ProgressView()
                } else {
                    Button(batch.uploadedTestId == nil ? "上傳並繪圖" : "重新上傳") {
                        model.upload(batchId: batch.id)
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.blue)
                }
            }
            Text("\(batch.laps.count) 圈，\(batch.points.count) 個點")
                .font(.caption2)
                .foregroundColor(.appTextSecondary)
            if let testId = batch.uploadedTestId {
                Text("已上傳 test_id: \(testId)")
                    .font(.caption2)
                    .foregroundColor(.green)
                Text("backend/output/sensor_tests/\(testId)/plot.png")
                    .font(.caption2)
                    .foregroundColor(.appTextTertiary)
            }
        }
        .padding(.vertical, 4)
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
}

@MainActor
final class SensorTestModel: ObservableObject {
    struct SensorTestBatch: Codable, Identifiable {
        let id: String
        var startedAt: Double
        var points: [PathPoint] = []
        var laps: [SensorTestLap] = []
        var uploadedTestId: String?
        /// Set when this batch came from HealthKit (Apple Watch workout route) instead
        /// of a live CoreMotion recording done in this screen.
        var sourceLabel: String?

        var startedAtDisplay: String {
            let date = Date(timeIntervalSince1970: startedAt / 1000)
            let formatter = DateFormatter()
            formatter.dateFormat = "M/d HH:mm:ss"
            return formatter.string(from: date)
        }
    }

    @Published private(set) var isTracking = false
    @Published private(set) var stepCount = 0
    @Published private(set) var distanceMeters: Double = 0
    @Published private(set) var headingDegrees: Double = -1
    @Published private(set) var x: Double = 0
    @Published private(set) var y: Double = 0
    @Published private(set) var batches: [SensorTestBatch] = []
    @Published private(set) var uploadError: String?
    @Published private var uploadingBatchIds: Set<String> = []

    @Published private(set) var workouts: [HKWorkout] = []
    @Published private(set) var isLoadingWorkouts = false
    @Published private(set) var healthKitError: String?
    @Published private(set) var importingWorkoutIds: Set<UUID> = []

    @Published private(set) var isCalibrating = false
    @Published private(set) var calibrationSampleCount = 0
    @Published var useCalibrationCorrection = false {
        didSet { tracker.useCalibrationCorrection = useCalibrationCorrection }
    }
    private let calibrator = MagnetometerCalibrator()

    var currentBatch: SensorTestBatch? {
        guard let currentBatchId else { return nil }
        return batches.first { $0.id == currentBatchId }
    }

    var driftFromOrigin: Double { (x * x + y * y).squareRoot() }

    private var currentBatchId: String?
    private let tracker = MotionPathTracker()
    private let storageKey = "sensor_test_batches.json"

    init() {
        batches = PersistenceService.shared.load([SensorTestBatch].self, from: storageKey) ?? []

        tracker.onPathPoint = { [weak self] point in
            guard let self, let index = self.currentBatchIndex else { return }
            self.stepCount = point.stepCount
            self.distanceMeters = point.distanceMeters
            self.headingDegrees = point.headingDegrees
            self.x = point.x
            self.y = point.y
            self.batches[index].points.append(point)
            self.persist()
        }
    }

    private var currentBatchIndex: Int? {
        guard let currentBatchId else { return nil }
        return batches.firstIndex { $0.id == currentBatchId }
    }

    func isUploading(batchId: String) -> Bool { uploadingBatchIds.contains(batchId) }

    /// Runs the ~12s rotate-the-phone calibration gesture and persists the result —
    /// see MagnetometerCalibrator's doc comment for what it corrects and why.
    func calibrateMagnetometer() {
        guard !isCalibrating else { return }
        isCalibrating = true
        calibrationSampleCount = 0
        calibrator.onSampleCount = { [weak self] count in
            self?.calibrationSampleCount = count
        }
        calibrator.start { [weak self] calibration in
            guard let self else { return }
            PersistenceService.shared.save(calibration, to: "magnetometer_calibration.json")
            self.tracker.reloadCalibration()
            NavigationSessionManager.shared.reloadCalibration()
            self.isCalibrating = false
        }
    }

    /// Always appends a brand new batch — never touches or clears any previous one.
    func startNewBatch() {
        guard !isTracking, MotionPathTracker.isAvailable else { return }
        isTracking = true
        stepCount = 0
        distanceMeters = 0
        headingDegrees = -1
        x = 0
        y = 0
        uploadError = nil

        let batch = SensorTestBatch(id: UUID().uuidString, startedAt: Date().timeIntervalSince1970 * 1000)
        batches.append(batch)
        currentBatchId = batch.id
        persist()
        tracker.start()
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        tracker.stop()
    }

    func markLap() {
        guard isTracking, let index = currentBatchIndex else { return }
        let laps = batches[index].laps
        let previous = laps.last.map { (x: $0.x, y: $0.y) } ?? (x: 0, y: 0)
        let dx = x - previous.x
        let dy = y - previous.y
        batches[index].laps.append(SensorTestLap(
            lapNumber: laps.count + 1,
            x: x,
            y: y,
            timestamp: Date().timeIntervalSince1970 * 1000,
            distanceFromOrigin: driftFromOrigin,
            distanceFromPreviousLap: (dx * dx + dy * dy).squareRoot()
        ))
        persist()
    }

    /// Uploads any batch — the current one being tracked, or an old one from a previous
    /// app launch. Safe to retry: the batch stays on disk regardless of outcome.
    func upload(batchId: String) {
        guard let index = batches.firstIndex(where: { $0.id == batchId }), !uploadingBatchIds.contains(batchId) else { return }
        uploadingBatchIds.insert(batchId)
        uploadError = nil

        let points = batches[index].points
        let laps = batches[index].laps
        Task {
            do {
                let testId = try await NavigationAPI.shared.uploadSensorTest(points: points, laps: laps)
                await MainActor.run {
                    if let i = self.batches.firstIndex(where: { $0.id == batchId }) {
                        self.batches[i].uploadedTestId = testId
                        self.persist()
                    }
                    self.uploadingBatchIds.remove(batchId)
                }
            } catch {
                await MainActor.run {
                    self.uploadError = "上傳失敗: \(error.localizedDescription)"
                    self.uploadingBatchIds.remove(batchId)
                }
            }
        }
    }

    private func persist() {
        PersistenceService.shared.save(batches, to: storageKey)
    }

    // MARK: - Apple Watch workout import (HealthKit fallback)

    func loadRecentWorkouts() {
        guard !isLoadingWorkouts else { return }
        isLoadingWorkouts = true
        healthKitError = nil

        Task {
            do {
                try await HealthKitService.shared.requestAuthorization()
                let results = try await HealthKitService.shared.fetchRecentWorkouts(daysBack: 3)
                await MainActor.run {
                    self.workouts = results
                    self.isLoadingWorkouts = false
                }
            } catch {
                await MainActor.run {
                    self.healthKitError = "讀取健身紀錄失敗: \(error.localizedDescription)"
                    self.isLoadingWorkouts = false
                }
            }
        }
    }

    /// Adds the workout's GPS route as a new local batch (if it has one) so it can be
    /// uploaded/plotted through the exact same pipeline as a live phone recording.
    func importWorkoutRoute(_ workout: HKWorkout) {
        guard !importingWorkoutIds.contains(workout.uuid) else { return }
        importingWorkoutIds.insert(workout.uuid)
        healthKitError = nil

        Task {
            do {
                let locations = try await HealthKitService.shared.fetchRoute(for: workout)
                await MainActor.run {
                    self.importingWorkoutIds.remove(workout.uuid)
                    guard !locations.isEmpty else {
                        self.healthKitError = "這筆訓練沒有 GPS 路徑資料（可能是室內訓練，或當時收不到 GPS）"
                        return
                    }
                    let points = HealthKitService.shared.convertToLocalPoints(locations)
                    var batch = SensorTestBatch(
                        id: workout.uuid.uuidString,
                        startedAt: workout.startDate.timeIntervalSince1970 * 1000,
                        points: points
                    )
                    batch.sourceLabel = "Apple Watch 訓練匯入"
                    if let existingIndex = self.batches.firstIndex(where: { $0.id == batch.id }) {
                        self.batches[existingIndex] = batch
                    } else {
                        self.batches.append(batch)
                    }
                    self.persist()
                }
            } catch {
                await MainActor.run {
                    self.importingWorkoutIds.remove(workout.uuid)
                    self.healthKitError = "匯入失敗: \(error.localizedDescription)"
                }
            }
        }
    }

    func isImporting(workoutId: UUID) -> Bool { importingWorkoutIds.contains(workoutId) }
}
