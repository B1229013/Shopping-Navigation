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
    @State private var pendingExportBatchId: String?

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

                    carryingModeCard

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
                        Text("步態參數校正（身高、步長 k / k1）")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.appTextPrimary)
                        Text("依《基於行人航位推算之室內定位研究》的步長模型：剛開始走（或停下後再走）的第一步用 k1×身高估計，之後每一步改用 k×身高×√步頻。k、k1 因人而異，走一輪後可以校正。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)

                        HStack {
                            Text("身高 (cm)")
                                .font(.caption)
                            TextField("170", text: $model.heightInputText)
                                .keyboardType(.decimalPad)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 80)
                            Button("更新") { model.updateHeight() }
                                .font(.caption.weight(.semibold))
                                .foregroundColor(.blue)
                        }

                        HStack {
                            Text("已知距離 (m，選填)")
                                .font(.caption)
                            TextField("留空＝情境二", text: $model.knownDistanceInputText)
                                .keyboardType(.decimalPad)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 110)
                        }

                        Text("目前 k = \(String(format: "%.4f", model.gaitProfile.k))，k1 = \(String(format: "%.4f", model.gaitProfile.k1))")
                            .font(.caption)
                            .foregroundColor(.appTextSecondary)

                        Button("用剛剛停止的這一輪資料校正") {
                            model.calibrateGait()
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)
                        .disabled(model.isTracking || model.currentBatch == nil)

                        Text("校正方式：填了「已知距離」＝情境一（同時校正 k 和 k1，較準）；留空＝情境二（只校正 k，沿用目前 k1）。走的時候要保持連續移動、不要中途停下。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)

                        if let gaitCalibrationMessage = model.gaitCalibrationMessage {
                            Text(gaitCalibrationMessage)
                                .font(.caption2)
                                .foregroundColor(.green)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    VStack(alignment: .leading, spacing: 10) {
                        Text("方向角融合方式（A/B 測試用）")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.appTextPrimary)
                        Text("開＝用陀螺儀＋磁力計的 Kalman Filter 融合方向角；關＝直接用手機自己算的磁力計方向角，不做任何融合。目前戶外/室內測試都是「關」比較準，所以預設改成關閉；如果之後陀螺儀那段有調整，再開回來比較看看有沒有改善。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)
                        Toggle("使用陀螺儀融合（關＝純磁力計）", isOn: $model.useGyroFusion)
                            .font(.caption)
                            .disabled(model.isTracking)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    VStack(alignment: .leading, spacing: 10) {
                        Text("磁力計校正（⚠️ 已知有問題，不要開）")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.red)
                        Text("套用方式對手機直立拿著的姿勢是錯的，會把真正的轉彎訊號一起抵銷掉，實測過會讓誤差更大（曾經把繞兩圈的轉彎全部消掉，路徑變成幾乎一直線）。校準動作本身沒問題，問題出在套用校正結果那一段程式邏輯，還沒修好前，下面這個開關保持關閉。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)

                        Toggle("套用校正修正方向（不要開）", isOn: $model.useCalibrationCorrection)
                            .font(.caption)
                            .tint(.red)

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

                        Text("校準本身可以照常做（用來收集資料），但先不要打開上面的套用開關。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
                    .background(Color.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                    VStack(alignment: .leading, spacing: 10) {
                        Text("陀螺儀零偏校正")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.appTextPrimary)
                        Text("陀螺儀靜止不動時理論上該讀 0，但實際硬體會有一個小小的固定偏移，這個偏移一直積分下去就是「加速度＋陀螺儀」「Game Rotation Vector」這類沒有磁力計修正的方向法會慢慢飄走的原因之一。校正方式：把手機完全靜止放著幾秒，量出這個偏移，之後每次讀陀螺儀都先扣掉它——這裡校正完，「計步器測試」「感測器組合測試」兩邊的陀螺儀公式會一起套用。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)

                        Button {
                            model.calibrateGyro()
                        } label: {
                            if model.isGyroCalibrating {
                                Text("校正中…已收集 \(model.gyroCalibrationSampleCount) 筆（5 秒，請勿移動手機）")
                            } else {
                                Text("開始校正（手機完全靜止放著）")
                            }
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)
                        .disabled(model.isGyroCalibrating || model.isTracking)

                        if let msg = model.gyroCalibrationMessage {
                            Text(msg)
                                .font(.caption2)
                                .foregroundColor(.green)
                        }
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
            .scrollDismissesKeyboard(.interactively)
            .background(Color.appBg)
            .navigationTitle("計步器測試")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("完成") {
                        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                    }
                }
            }
            .sheet(item: Binding(
                get: { pendingExportBatchId.flatMap { model.exportCSV(batchId: $0) }.map { ShareItem(url: $0) } },
                set: { _ in pendingExportBatchId = nil }
            )) { item in
                ActivityView(activityItems: [item.url])
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
            Text("\(batch.laps.count) 圈，\(batch.points.count) 個點，\(batch.rawSamples.count) 筆原始感測器紀錄")
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

            if !batch.rawSamples.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text("實際數值（走完後填，跟感測器讀數比對用）")
                        .font(.caption2.weight(.semibold))
                        .foregroundColor(.appTextTertiary)
                    HStack(spacing: 8) {
                        TextField("實際步數", text: model.groundTruthStepsBinding(batch.id))
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                        TextField("實際距離(m)", text: model.groundTruthDistanceBinding(batch.id))
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.decimalPad)
                        TextField("轉彎次數", text: model.groundTruthTurnsBinding(batch.id))
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                    }
                    .font(.caption)

                    Button {
                        pendingExportBatchId = batch.id
                    } label: {
                        Label("匯出 CSV（\(batch.rawSamples.count) 筆）", systemImage: "square.and.arrow.up")
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.blue)
                }
                .padding(.top, 4)
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

    private static let selectableCarryingModes: [CarryingMode] = [
        .holding, .pocket, .swingRight, .swingLeft,
    ]

    private var carryingModeCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("攜帶姿勢（手動選擇，套用對應公式）")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.appTextPrimary)
            Text("依《基於行人航位推算之室內定位研究》§3.6，四種拿法各有各自的陀螺儀方向角公式。自動判別的準確度還沒驗證過，所以改成你自己選現在是哪種拿法，程式就套用對應公式（僅在上面「使用陀螺儀融合」開著時才有作用）。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                ForEach(Self.selectableCarryingModes, id: \.self) { mode in
                    Button(mode.rawValue) {
                        model.selectedCarryingMode = mode
                    }
                    .font(.caption2.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(model.selectedCarryingMode == mode ? Color.appAccent : Color.appBg)
                    .foregroundColor(model.selectedCarryingMode == mode ? .white : .appTextSecondary)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }

            Divider()
            Text("自動判別（僅供參考，跟你手動選的比對用）")
                .font(.caption2.weight(.semibold))
                .foregroundColor(.appTextTertiary)
            metricRow(label: "自動偵測到的姿勢", value: model.carryingMode.rawValue)
            metricRow(label: "傾角 degTx", value: String(format: "%.0f°", model.degTx))
            metricRow(label: "傾角平面姿態角 degP", value: String(format: "%.0f°", model.degP))
        }
        .padding(20)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
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
        /// Snapshot of the heading-fusion A/B toggle and gait profile in effect when this
        /// batch started, so a later look at the exported record.json can tell which
        /// configuration actually produced it instead of guessing.
        var config: SensorTestConfig?
        /// Throttled (~10Hz) raw sensor log, independent of step detection — lets a walk be
        /// re-analyzed offline against different algorithm parameters without a new physical
        /// walk each time. See `RawSensorSample`.
        var rawSamples: [RawSensorSample] = []
        /// Filled in by hand after stopping, for comparing against what the sensors reported.
        var groundTruthSteps: String = ""
        var groundTruthDistanceM: String = ""
        var groundTruthTurns: String = ""

        var startedAtDisplay: String {
            let date = Date(timeIntervalSince1970: startedAt / 1000)
            let formatter = DateFormatter()
            formatter.dateFormat = "M/d HH:mm:ss"
            return formatter.string(from: date)
        }

        init(id: String, startedAt: Double, points: [PathPoint] = [], config: SensorTestConfig? = nil) {
            self.id = id
            self.startedAt = startedAt
            self.points = points
            self.config = config
        }

        private enum CodingKeys: String, CodingKey {
            case id, startedAt, points, laps, uploadedTestId, sourceLabel, config, rawSamples,
                 groundTruthSteps, groundTruthDistanceM, groundTruthTurns
        }

        /// Every field below `startedAt` has been added after this struct's first shipped
        /// version. A default value on a stored property only fills in the compiler-generated
        /// *memberwise* init — Swift's synthesized `Decodable` still requires each key to be
        /// present, so a batch persisted before a field existed would otherwise fail to decode
        /// entirely; `PersistenceService.load`'s `try?` turns that into a silent nil, which
        /// reads as "no saved batches", and the next `persist()` call then overwrites the file
        /// with an empty array — permanently losing every older batch (see the identical fix
        /// on `SensorComboBatch` in `SensorComboTestView.swift`, which this mirrors). This
        /// custom init defaults each newer field with `decodeIfPresent` instead of requiring it.
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            startedAt = try c.decode(Double.self, forKey: .startedAt)
            points = try c.decodeIfPresent([PathPoint].self, forKey: .points) ?? []
            laps = try c.decodeIfPresent([SensorTestLap].self, forKey: .laps) ?? []
            uploadedTestId = try c.decodeIfPresent(String.self, forKey: .uploadedTestId)
            sourceLabel = try c.decodeIfPresent(String.self, forKey: .sourceLabel)
            config = try c.decodeIfPresent(SensorTestConfig.self, forKey: .config)
            rawSamples = try c.decodeIfPresent([RawSensorSample].self, forKey: .rawSamples) ?? []
            groundTruthSteps = try c.decodeIfPresent(String.self, forKey: .groundTruthSteps) ?? ""
            groundTruthDistanceM = try c.decodeIfPresent(String.self, forKey: .groundTruthDistanceM) ?? ""
            groundTruthTurns = try c.decodeIfPresent(String.self, forKey: .groundTruthTurns) ?? ""
        }
    }

    @Published private(set) var isTracking = false
    @Published private(set) var stepCount = 0
    @Published private(set) var distanceMeters: Double = 0
    @Published private(set) var headingDegrees: Double = -1
    @Published private(set) var carryingMode: CarryingMode = .unknown
    @Published private(set) var degTx: Double = 0
    @Published private(set) var degP: Double = 0
    /// Manually declared posture — drives which of §3.6's four gyro formulas
    /// `MotionPathTracker` actually uses, instead of the (unverified) auto-detected `carryingMode`.
    @Published var selectedCarryingMode: CarryingMode = .holding {
        didSet { tracker.carryingMode = selectedCarryingMode }
    }
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
    @Published var useGyroFusion = false {
        didSet { tracker.useGyroFusion = useGyroFusion }
    }
    private let calibrator = MagnetometerCalibrator()

    @Published private(set) var isGyroCalibrating = false
    @Published private(set) var gyroCalibrationSampleCount = 0
    @Published private(set) var gyroCalibrationMessage: String?
    private let gyroCalibrator = GyroCalibrator()

    @Published private(set) var gaitProfile: GaitProfile
    @Published var heightInputText: String
    @Published var knownDistanceInputText: String = ""
    @Published private(set) var gaitCalibrationMessage: String?

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
        let profile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
        gaitProfile = profile
        heightInputText = String(format: "%.0f", profile.heightCM)

        tracker.onHeadingUpdate = { [weak self] heading in
            self?.headingDegrees = heading
        }

        tracker.onCarryingModeUpdate = { [weak self] mode, degTx, degP in
            self?.carryingMode = mode
            self?.degTx = degTx
            self?.degP = degP
        }

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

        tracker.onRawSample = { [weak self] sample in
            guard let self, let index = self.currentBatchIndex else { return }
            self.batches[index].rawSamples.append(sample)
            // Persisting on every ~10Hz sample would re-encode the whole (growing) array
            // each time — throttle to ~1Hz instead; stop() flushes anything left over.
            self.rawSampleCountSinceLastPersist += 1
            if self.rawSampleCountSinceLastPersist >= 10 {
                self.rawSampleCountSinceLastPersist = 0
                self.persist()
            }
        }
    }

    private var rawSampleCountSinceLastPersist = 0

    private var currentBatchIndex: Int? {
        guard let currentBatchId else { return nil }
        return batches.firstIndex { $0.id == currentBatchId }
    }

    func isUploading(batchId: String) -> Bool { uploadingBatchIds.contains(batchId) }

    // MARK: - Ground truth + CSV export

    func groundTruthStepsBinding(_ batchId: String) -> Binding<String> {
        batchStringBinding(batchId: batchId, keyPath: \.groundTruthSteps)
    }
    func groundTruthDistanceBinding(_ batchId: String) -> Binding<String> {
        batchStringBinding(batchId: batchId, keyPath: \.groundTruthDistanceM)
    }
    func groundTruthTurnsBinding(_ batchId: String) -> Binding<String> {
        batchStringBinding(batchId: batchId, keyPath: \.groundTruthTurns)
    }

    private func batchStringBinding(batchId: String, keyPath: WritableKeyPath<SensorTestBatch, String>) -> Binding<String> {
        Binding(
            get: { self.batches.first(where: { $0.id == batchId })?[keyPath: keyPath] ?? "" },
            set: { newValue in
                guard let idx = self.batches.firstIndex(where: { $0.id == batchId }) else { return }
                self.batches[idx][keyPath: keyPath] = newValue
                self.persist()
            }
        )
    }

    /// Writes this batch's raw sensor log (plus its ground truth) to a CSV in the temp
    /// directory and returns the file URL to share — modeled after the Android teammate's
    /// `exportCsv()` in `SensorLabScreen.kt`, so a walk can be re-analyzed offline against
    /// different algorithm parameters without a new physical walk each time.
    func exportCSV(batchId: String) -> URL? {
        guard let batch = batches.first(where: { $0.id == batchId }), let first = batch.rawSamples.first else { return nil }

        var csv = "# SensorLab Export — \(batch.startedAtDisplay)\n"
        csv += "# ground_truth_steps=\(batch.groundTruthSteps),ground_truth_distance_m=\(batch.groundTruthDistanceM),ground_truth_turns=\(batch.groundTruthTurns)\n"
        csv += "timestamp_ms,elapsed_sec,step_count,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,heading_deg\n"
        for s in batch.rawSamples {
            let elapsed = (s.timestamp - first.timestamp) / 1000
            csv += String(
                format: "%.0f,%.3f,%d,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.1f\n",
                s.timestamp, elapsed, s.stepCount,
                s.accelX, s.accelY, s.accelZ,
                s.gyroX, s.gyroY, s.gyroZ,
                s.magX, s.magY, s.magZ,
                s.headingDegrees
            )
        }

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sensor_log_\(batch.id).csv")
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            return nil
        }
    }

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

    /// Runs the ~5s hold-still calibration gesture and persists the result — see
    /// `GyroCalibrator`'s doc comment for what it corrects and why. Reloads into every
    /// tracker that reads raw gyro (this screen's own, the real-navigation session, and —
    /// next time it starts a round — 「感測器組合測試」's `SensorComboTracker`, which reloads
    /// its own copy from disk when a new round starts).
    func calibrateGyro() {
        guard !isGyroCalibrating else { return }
        isGyroCalibrating = true
        gyroCalibrationSampleCount = 0
        gyroCalibrationMessage = nil
        gyroCalibrator.onSampleCount = { [weak self] count in
            self?.gyroCalibrationSampleCount = count
        }
        gyroCalibrator.start { [weak self] bias in
            guard let self else { return }
            PersistenceService.shared.save(bias, to: "gyro_bias.json")
            self.tracker.reloadGyroBias()
            NavigationSessionManager.shared.reloadGyroBias()
            self.isGyroCalibrating = false
            self.gyroCalibrationMessage = String(
                format: "校正完成：零偏 x=%.4f, y=%.4f, z=%.4f 弧度/秒",
                bias.x, bias.y, bias.z
            )
        }
    }

    /// Persists a manually-entered height immediately (independent of gait calibration,
    /// which only updates k/k1 — height itself has no "measured" source here).
    func updateHeight() {
        guard let height = Double(heightInputText), height > 0 else {
            gaitCalibrationMessage = "身高輸入無效"
            return
        }
        gaitProfile.heightCM = height
        PersistenceService.shared.save(gaitProfile, to: "gait_profile.json")
        tracker.reloadGaitProfile()
        gaitCalibrationMessage = "身高已更新為 \(String(format: "%.0f", height)) cm"
    }

    /// Solves k/k1 (see `GaitCalibrator`) from the just-completed batch's total step count
    /// and elapsed time — both independent of whatever step-length formula was in effect
    /// during the walk, since `PathPoint.stepCount` only counts detected steps.
    func calibrateGait() {
        guard !isTracking, let batch = currentBatch, let last = batch.points.last, last.stepCount > 0 else {
            gaitCalibrationMessage = "請先走一輪並按停止，再校正"
            return
        }
        let elapsedSeconds = (last.timestamp - batch.startedAt) / 1000
        let distance = Double(knownDistanceInputText)
        let newProfile: GaitProfile?
        if let distance, distance > 0 {
            newProfile = GaitCalibrator.calibrate(knownDistanceMeters: distance, heightCM: gaitProfile.heightCM,
                                                  stepCount: last.stepCount, elapsedSeconds: elapsedSeconds)
        } else {
            newProfile = GaitCalibrator.calibrate(heightCM: gaitProfile.heightCM, currentK1: gaitProfile.k1,
                                                  stepCount: last.stepCount, elapsedSeconds: elapsedSeconds)
        }
        guard let newProfile else {
            gaitCalibrationMessage = "校正失敗，請確認這一輪的步數與時間有效"
            return
        }
        gaitProfile = newProfile
        PersistenceService.shared.save(gaitProfile, to: "gait_profile.json")
        tracker.reloadGaitProfile()
        gaitCalibrationMessage = "校正完成：k = \(String(format: "%.4f", newProfile.k))，k1 = \(String(format: "%.4f", newProfile.k1))"
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

        let config = SensorTestConfig(useGyroFusion: useGyroFusion, useCalibrationCorrection: useCalibrationCorrection,
                                       heightCM: gaitProfile.heightCM, gaitK: gaitProfile.k, gaitK1: gaitProfile.k1)
        let batch = SensorTestBatch(id: UUID().uuidString, startedAt: Date().timeIntervalSince1970 * 1000, config: config)
        batches.append(batch)
        currentBatchId = batch.id
        persist()
        tracker.start()
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        tracker.stop()
        rawSampleCountSinceLastPersist = 0
        persist() // flush any raw samples not yet written by the ~1Hz throttle above
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
        let config = batches[index].config
        Task {
            do {
                let testId = try await NavigationAPI.shared.uploadSensorTest(points: points, laps: laps, config: config)
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

private struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.path }
}

private struct ActivityView: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
