import Combine
import HealthKit
import SwiftUI

/// A single "拍照標記" waypoint — records where every heading method thought the user was
/// standing at the moment they tapped the button (before the camera even opens), plus the
/// photo they took there. Meant to be cross-referenced later (e.g. in `TopoMapTestView`'s
/// offline mode) against whichever heading method's trajectory turns out most trustworthy.
struct ComboWaypoint: Codable, Identifiable {
    var id: String
    var index: Int
    var timestamp: Double
    var photoFileName: String?
    var positions: [String: PointXY]
}

/// One "標記回到原點" press — same idea as `SensorTestLap` in `SensorTestView`, but keeps
/// all five methods' positions (not just one) since this screen's whole point is comparing
/// them. A diagnostic marker only: recording how far off each method's own trajectory
/// currently is from its own (0, 0), not correcting anything. Lets one continuous walk cover
/// several loops (out and back to the start) instead of having to stop — which would end the
/// batch — after every single loop.
struct ComboLap: Codable, Identifiable {
    var id: String
    var lapNumber: Int
    var timestamp: Double
    var positions: [String: PointXY]
}

/// One manually-confirmed checkpoint — recorded either by tapping "記錄檢查點" directly, or
/// automatically alongside a "拍照標記" waypoint (see `SensorComboTestModel.markCheckpoint()`
/// and `finishWaypointCapture(image:)`), so a photo mark doubles as an accuracy checkpoint
/// without a second button press. All five methods' current positions get snapshotted
/// together with the step count / cumulative distance / timestamp. Unlike `ComboLap` (a
/// manual "I'm back at the start" press meant for whole-loop drift), this is meant to build a
/// per-segment accuracy trail along the walk. Ground truth per checkpoint isn't stored here:
/// it's looked up lazily against the batch's own live GPS track (nearest-timestamp match, see
/// `SensorComboTestModel.gpsGroundTruth(atTimestamp:in:)`) so it always reflects the
/// *complete* GPS history, not just whatever had arrived by that instant while walking.
struct ComboCheckpoint: Codable, Identifiable {
    var id: String
    var stepCount: Int
    var timestamp: Double
    var distanceMeters: Double
    var positions: [String: PointXY]
    /// Set when this checkpoint came from a "拍照標記" capture instead of a bare "記錄檢查點"
    /// press — the matching `ComboWaypoint.index`, so the UI can show which photo it is.
    var waypointIndex: Int?
}

/// One full test walk: every heading method's step-by-step trajectory, any waypoint photos
/// taken along the way, and an optional GPS reference track imported from an Apple Watch
/// workout for comparison.
struct SensorComboBatch: Codable, Identifiable {
    let id: String
    var startedAt: Double
    var points: [MultiHeadingPoint] = []
    var waypoints: [ComboWaypoint] = []
    /// Imported *after the fact* via `HealthKitService` — already converted to the same
    /// local (x,y) meter frame the PDR trajectories use (see
    /// `HealthKitService.convertToLocalPoints`), so it can be overlaid directly without a
    /// separate alignment step.
    var gpsOverlayPoints: [PathPoint]?
    /// Recorded *live, in this same screen, at the same time* as the PDR trajectories —
    /// via the embedded `GPSLogger` (see `SensorComboTestModel.startGPS()`), independent of
    /// the Watch/HealthKit import above. Raw lat/lon; converted to local meters on demand
    /// for display/export (see `SensorComboTestModel.convertGPSSamplesToLocal(_:)`).
    var gpsSamples: [GPSSample] = []
    var laps: [ComboLap] = []
    var checkpoints: [ComboCheckpoint] = []
    var groundTruthSteps: String = ""
    var groundTruthDistanceM: String = ""

    var startedAtDisplay: String {
        let date = Date(timeIntervalSince1970: startedAt / 1000)
        let formatter = DateFormatter()
        formatter.dateFormat = "M/d HH:mm:ss"
        return formatter.string(from: date)
    }

    init(id: String, startedAt: Double) {
        self.id = id
        self.startedAt = startedAt
    }

    private enum CodingKeys: String, CodingKey {
        case id, startedAt, points, waypoints, gpsOverlayPoints, gpsSamples, laps, checkpoints, groundTruthSteps, groundTruthDistanceM
    }

    /// Every field below `startedAt` has been added after this struct's first shipped
    /// version. A default value on a stored property only fills in the compiler-generated
    /// *memberwise* initializer — Swift's synthesized `Decodable` still requires each key to
    /// be present, so a batch persisted before a field existed would otherwise fail to decode
    /// at all. `PersistenceService.load` swallows that failure with `try?` and returns nil,
    /// which reads as "no saved batches" and — worse — the next `persist()` call would then
    /// overwrite the file with an empty array, permanently losing every older batch. This
    /// custom init defaults each newer field with `decodeIfPresent` instead of requiring it.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        startedAt = try c.decode(Double.self, forKey: .startedAt)
        points = try c.decodeIfPresent([MultiHeadingPoint].self, forKey: .points) ?? []
        waypoints = try c.decodeIfPresent([ComboWaypoint].self, forKey: .waypoints) ?? []
        gpsOverlayPoints = try c.decodeIfPresent([PathPoint].self, forKey: .gpsOverlayPoints)
        gpsSamples = try c.decodeIfPresent([GPSSample].self, forKey: .gpsSamples) ?? []
        laps = try c.decodeIfPresent([ComboLap].self, forKey: .laps) ?? []
        checkpoints = try c.decodeIfPresent([ComboCheckpoint].self, forKey: .checkpoints) ?? []
        groundTruthSteps = try c.decodeIfPresent(String.self, forKey: .groundTruthSteps) ?? ""
        groundTruthDistanceM = try c.decodeIfPresent(String.self, forKey: .groundTruthDistanceM) ?? ""
    }
}

/// Compares all five heading methods (see `HeadingMethodID`) side by side on the same walk.
/// A sibling to `SensorTestView`, not a replacement — that screen stays the place for
/// single-method sensor/step-length tuning; this one is purely for the "which heading
/// source drifts least" comparison, and shares the same `GaitProfile` (calibrated in
/// `SensorTestView`) rather than duplicating that UI.
struct SensorComboTestView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = SensorComboTestModel()
    @State private var showCamera = false

    private static let methodColors: [HeadingMethodID: Color] = [
        .accelMag: .orange,
        .accelGyro: .purple,
        .accelMagGyro: .red,
        .rotationVector: .blue,
        .gameRotationVector: .green,
    ]

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    statusCard
                    canvasCard
                    controlsCard
                    gpsLiveCard
                    waypointsCard
                    lapsCard
                    checkpointsCard
                    gpsImportCard
                    historyCard
                }
                .padding(20)
                .padding(.top, 4)
            }
            .scrollDismissesKeyboard(.interactively)
            .background(Color.appBg)
            .navigationTitle("感測器組合測試")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
            .sheet(isPresented: $showCamera) {
                CameraPickerView { image in
                    model.finishWaypointCapture(image: image)
                }
            }
            // Shared across `waypointsCard` (current round) and `historyCard` (any past
            // round) — both just set `model.pendingSharePhotoURLs`, so this one sheet
            // definition covers either trigger regardless of which round it came from.
            .sheet(item: Binding(
                get: { model.pendingSharePhotoURLs.map { SharePhotosItem(urls: $0) } },
                set: { _ in model.pendingSharePhotoURLs = nil }
            )) { item in
                ActivityViewCombo(activityItems: item.urls)
            }
        }
        .onDisappear { model.stopAll() }
    }

    private var statusCard: some View {
        VStack(spacing: 12) {
            metricRow(label: "狀態", value: model.isTracking ? "追蹤中" : "已停止")
            metricRow(label: "步數", value: "\(model.stepCount)")
            metricRow(label: "累積距離", value: String(format: "%.2f 公尺", model.distanceMeters))
            Divider()
            ForEach(HeadingMethodID.allCases, id: \.self) { method in
                HStack {
                    Circle()
                        .fill(Self.methodColors[method] ?? .gray)
                        .frame(width: 10, height: 10)
                    Text(method.rawValue)
                        .font(.caption)
                        .foregroundColor(.appTextSecondary)
                    Spacer()
                    Text(String(format: "%.0f°", model.currentHeadings[method] ?? 0))
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.appTextPrimary)
                }
            }
            Divider()
            Text("身高/步態參數（k=\(String(format: "%.4f", model.gaitProfile.k))，k1=\(String(format: "%.4f", model.gaitProfile.k1))，身高\(String(format: "%.0f", model.gaitProfile.heightCM))cm）沿用「計步器測試」畫面校正的結果，這裡不用重複校正。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)
        }
        .padding(20)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var canvasCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("五種方向法軌跡比較")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.appTextPrimary)
            Canvas { context, size in
                drawTrajectories(context: context, size: size)
            }
            .frame(height: 340)
            .background(Color.black.opacity(0.04))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            legend
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(HeadingMethodID.allCases, id: \.self) { method in
                HStack(spacing: 6) {
                    Rectangle().fill(Self.methodColors[method] ?? .gray).frame(width: 14, height: 3)
                    Text(method.rawValue).font(.caption2).foregroundColor(.appTextSecondary)
                }
            }
            if model.currentBatch?.gpsOverlayPoints?.isEmpty == false {
                HStack(spacing: 6) {
                    Rectangle().fill(Color.black).frame(width: 14, height: 3)
                    Text("GPS（Apple Watch 事後匯入）").font(.caption2).foregroundColor(.appTextSecondary)
                }
            }
            if model.currentBatch?.gpsSamples.isEmpty == false {
                HStack(spacing: 6) {
                    Rectangle().fill(Color.gray).frame(width: 14, height: 3)
                    Text("GPS（即時同步記錄）").font(.caption2).foregroundColor(.appTextSecondary)
                }
            }
        }
    }

    private var controlsCard: some View {
        VStack(spacing: 10) {
            Button(model.isTracking ? "停止測試" : "開始新的一輪") {
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
            .disabled(!SensorComboTracker.isAvailable)

            Button {
                model.beginWaypointCapture()
                showCamera = true
            } label: {
                Label("拍照標記點（第 \((model.currentBatch?.waypoints.count ?? 0) + 1) 個）", systemImage: "camera.fill")
            }
            .frame(maxWidth: .infinity)
            .frame(height: 46)
            .background(Color.orange)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .disabled(!model.isTracking)

            Button("記錄檢查點（第 \((model.currentBatch?.checkpoints.count ?? 0) + 1) 個）") {
                model.markCheckpoint()
            }
            .frame(maxWidth: .infinity)
            .frame(height: 46)
            .background(Color.indigo)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .disabled(!model.isTracking)

            Text("走到你知道確切位置的點（例如量好距離的走道標線）就按一下，記下這一刻五種方向法各自的位置；「拍照標記點」也會自動順便記一個檢查點，不用兩個都按。事後對照同一輪的 GPS，就能看到每個檢查點、以及整體最後的誤差有多少（見下面「檢查點誤差」卡片，要先在下面開 GPS 同步記錄）。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)

            Button("標記回到原點（第 \((model.currentBatch?.laps.count ?? 0) + 1) 圈）") {
                model.markLap()
            }
            .frame(maxWidth: .infinity)
            .frame(height: 46)
            .background(Color.teal)
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .disabled(!model.isTracking)

            Text("走一圈回到原地按一下，可以不停止測試、重複繞多圈——每圈會記錄五種方向法各自跟原點差多少，方便比較哪種方法漂移最小。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)

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

    private var gpsLiveCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("GPS 同步記錄（跟上面的感測器測試同時跑）")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.appTextPrimary)
            Text("現在「開始新的一輪」會自動幫你開這裡的 GPS 記錄、「停止測試」也會自動關掉，每一輪的 GPS 起訖時間會跟感測器測試對齊，不會混到別輪去，不用再手動分開按。如果你想提早開 GPS 先讓它抓到訊號、暖機一下，還是可以自己先按下面的按鈕，「開始新的一輪」看到已經開著就不會重新啟動它；但只要按「停止測試」，不管 GPS 是自動開的還是你自己先開的，都會一起關掉，確保每一輪的界線乾淨。每一筆座標一收到就直接存進手機，不會等到累積好幾筆才寫檔。按「完成」離開這個畫面時 GPS 記錄也會跟著自動停止並存檔——GPS 目前沒辦法在背景（切到別的 App／鎖螢幕）繼續記錄。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)

            HStack {
                Text("已記錄 \(model.gpsSampleCount) 筆")
                    .font(.caption)
                    .foregroundColor(.appTextSecondary)
                Spacer()
                Button(model.isGPSTracking ? "停止 GPS 記錄" : "開始 GPS 記錄") {
                    if model.isGPSTracking {
                        model.stopGPS()
                    } else {
                        model.startGPS()
                    }
                }
                .font(.caption.weight(.semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(model.isGPSTracking ? Color.red : Color.blue)
                .clipShape(Capsule())
            }

            if let error = model.gpsAuthorizationError {
                Text(error).font(.caption).foregroundColor(.red)
            }

            if let batch = model.currentBatch, !batch.gpsSamples.isEmpty {
                Button {
                    if let url = model.exportGPSCSV(batchId: batch.id) {
                        model.pendingShareURL = url
                    }
                } label: {
                    Label("匯出這一輪的 GPS CSV（\(batch.gpsSamples.count) 筆）", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold))
                .foregroundColor(.blue)
            }
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder
    private var waypointsCard: some View {
        if let batch = model.currentBatch, !batch.waypoints.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("這一輪的標記點")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(.appTextPrimary)
                    Spacer()
                    Button {
                        let urls = model.photoShareURLs(batchId: batch.id)
                        if !urls.isEmpty { model.pendingSharePhotoURLs = urls }
                    } label: {
                        Label("分享照片（\(batch.waypoints.count) 張）", systemImage: "square.and.arrow.up")
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.blue)
                }
                ForEach(batch.waypoints) { waypoint in
                    HStack(spacing: 10) {
                        if let fileName = waypoint.photoFileName, let image = model.loadWaypointImage(batchId: batch.id, fileName: fileName) {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 44, height: 44)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            Text("第 \(waypoint.index) 點")
                                .font(.caption.weight(.semibold))
                                .foregroundColor(.appTextPrimary)
                            if let rv = waypoint.positions[HeadingMethodID.rotationVector.rawValue] {
                                Text(String(format: "旋轉向量座標 (%.1f, %.1f)", rv.x, rv.y))
                                    .font(.caption2)
                                    .foregroundColor(.appTextTertiary)
                            }
                        }
                        Spacer()
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }

    /// Only shown once there's something to check — either checkpoints have been recorded,
    /// or GPS is available to grade them against. "整體結果" is graded off the batch's very
    /// last PDR point (whatever position tracking actually stopped at), not off
    /// `checkpoints.last` — that way the overall figure doesn't depend on remembering to tap
    /// "記錄檢查點" right before "停止測試".
    @ViewBuilder
    private var checkpointsCard: some View {
        if let batch = model.currentBatch, !batch.checkpoints.isEmpty || !batch.gpsSamples.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("檢查點誤差（手動記錄，共 \(batch.checkpoints.count) 個）")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.appTextPrimary)

                if batch.gpsSamples.isEmpty {
                    Text("這一輪沒有同時開 GPS 同步記錄，檢查點的位置有記下來，但沒有真正的點位可以比對誤差。")
                        .font(.caption2)
                        .foregroundColor(.appTextTertiary)
                } else {
                    Text("每個檢查點都拿當下時間最接近的 GPS 位置當「真正的點位」算誤差。「誤差率」＝誤差 ÷ 累積走過距離，越低代表這種方向法在這段路走得越準。")
                        .font(.caption2)
                        .foregroundColor(.appTextTertiary)

                    if let last = batch.points.last {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("整體結果（走完全程、最後位置）")
                                .font(.caption.weight(.semibold))
                                .foregroundColor(.appTextPrimary)
                            ForEach(HeadingMethodID.allCases, id: \.self) { method in
                                if let error = SensorComboTestModel.error(atTimestamp: last.timestamp, positions: last.positions, method: method, in: batch) {
                                    let rate = last.distanceMeters > 0 ? error / last.distanceMeters * 100 : 0
                                    HStack {
                                        Circle().fill(Self.methodColors[method] ?? .gray).frame(width: 8, height: 8)
                                        Text(method.rawValue).font(.caption2).foregroundColor(.appTextSecondary)
                                        Spacer()
                                        Text(String(format: "誤差 %.2f 公尺（誤差率 %.1f%%）", error, rate))
                                            .font(.caption2)
                                            .foregroundColor(.appTextPrimary)
                                    }
                                }
                            }
                        }
                        .padding(.bottom, 4)
                        Divider()
                    }

                    if batch.checkpoints.isEmpty {
                        Text("這一輪還沒手動記錄過任何檢查點——走的路上按「記錄檢查點」或「拍照標記點」就會出現在這裡。")
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)
                    }

                    ForEach(batch.checkpoints) { checkpoint in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("第 \(checkpoint.stepCount) 步（累積 \(String(format: "%.1f", checkpoint.distanceMeters)) 公尺）\(checkpoint.waypointIndex.map { "・拍照標記第 \($0) 點" } ?? "")")
                                .font(.caption.weight(.semibold))
                                .foregroundColor(.appTextPrimary)
                            ForEach(HeadingMethodID.allCases, id: \.self) { method in
                                if let error = SensorComboTestModel.error(for: checkpoint, method: method, in: batch) {
                                    let rate = checkpoint.distanceMeters > 0 ? error / checkpoint.distanceMeters * 100 : 0
                                    HStack {
                                        Circle().fill(Self.methodColors[method] ?? .gray).frame(width: 8, height: 8)
                                        Text(method.rawValue).font(.caption2).foregroundColor(.appTextSecondary)
                                        Spacer()
                                        Text(String(format: "誤差 %.2f 公尺（%.1f%%）", error, rate))
                                            .font(.caption2)
                                            .foregroundColor(.appTextPrimary)
                                    }
                                }
                            }
                        }
                        .padding(.vertical, 4)
                        Divider()
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }

    @ViewBuilder
    private var lapsCard: some View {
        if let batch = model.currentBatch, !batch.laps.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Text("這輪已標記回到原點的圈數")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.appTextPrimary)
                ForEach(batch.laps) { lap in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("第 \(lap.lapNumber) 圈")
                            .font(.caption.weight(.semibold))
                            .foregroundColor(.appTextPrimary)
                        ForEach(HeadingMethodID.allCases, id: \.self) { method in
                            if let p = lap.positions[method.rawValue] {
                                HStack {
                                    Circle()
                                        .fill(Self.methodColors[method] ?? .gray)
                                        .frame(width: 8, height: 8)
                                    Text(method.rawValue)
                                        .font(.caption2)
                                        .foregroundColor(.appTextSecondary)
                                    Spacer()
                                    Text(String(format: "跟原點差 %.2f 公尺", (p.x * p.x + p.y * p.y).squareRoot()))
                                        .font(.caption2)
                                        .foregroundColor(.appTextPrimary)
                                }
                            }
                        }
                    }
                    .padding(.vertical, 4)
                    Divider()
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(Color.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }

    private var gpsImportCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("（備用方案）事後從 Apple Watch 訓練匯入 GPS 疊圖，僅限戶外訓練")
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

            if let error = model.healthKitError {
                Text(error).font(.caption).foregroundColor(.red)
            }

            ForEach(model.workouts, id: \.uuid) { workout in
                HStack {
                    Text("\(workout.workoutActivityType.displayName) · \(String(format: "%.0f 分鐘", workout.duration / 60))")
                        .font(.caption)
                        .foregroundColor(.appTextPrimary)
                    Spacer()
                    Button("匯入到目前這輪") {
                        model.importGPSOverlay(workout)
                    }
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.blue)
                    .disabled(model.currentBatch == nil)
                }
            }
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var historyCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("本機所有紀錄")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.appTextPrimary)
            if model.batches.isEmpty {
                Text("目前還沒有任何紀錄")
                    .font(.caption)
                    .foregroundColor(.appTextTertiary)
            }
            ForEach(model.batches.reversed()) { batch in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(batch.startedAtDisplay)
                            .font(.caption.weight(.medium))
                            .foregroundColor(.appTextPrimary)
                        Spacer()
                        Button {
                            if let url = model.exportCSV(batchId: batch.id) {
                                model.pendingShareURL = url
                            }
                        } label: {
                            Label("匯出 CSV", systemImage: "square.and.arrow.up")
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)
                    }
                    Text("\(batch.points.count) 個步伐點，\(batch.waypoints.count) 個標記點，\(batch.laps.count) 圈，\(batch.checkpoints.count) 個檢查點")
                        .font(.caption2)
                        .foregroundColor(.appTextSecondary)

                    // Every past round's photos live on here too, not just whichever one
                    // happens to be `currentBatch` right now — this is the fix for "only the
                    // round I'm actively tracking has a way to get its photos out": before,
                    // starting a new round silently made every earlier round's photos
                    // unreachable from the UI (the files were still on disk, just nothing
                    // pointed at them anymore).
                    if !batch.waypoints.isEmpty {
                        Button {
                            let urls = model.photoShareURLs(batchId: batch.id)
                            if !urls.isEmpty { model.pendingSharePhotoURLs = urls }
                        } label: {
                            Label("分享這輪照片（\(batch.waypoints.count) 張）", systemImage: "photo.on.rectangle")
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)
                    }

                    // Same fix, same reason, but for this round's GPS track — the "匯出這一輪
                    // 的 GPS CSV" button used to live only in `gpsLiveCard`, gated on
                    // `currentBatch`, so it disappeared the moment a new round started even
                    // though the GPS samples were still sitting safely in this batch's data.
                    if !batch.gpsSamples.isEmpty {
                        Button {
                            if let url = model.exportGPSCSV(batchId: batch.id) {
                                model.pendingShareURL = url
                            }
                        } label: {
                            Label("分享這輪 GPS CSV（\(batch.gpsSamples.count) 筆）", systemImage: "location.fill")
                        }
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.blue)
                    }

                    HStack(spacing: 8) {
                        TextField("實際步數", text: model.groundTruthStepsBinding(batch.id))
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                        TextField("實際距離(m)", text: model.groundTruthDistanceBinding(batch.id))
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.decimalPad)
                    }
                    .font(.caption)
                }
                .padding(.vertical, 4)
                Divider()
            }
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .sheet(item: Binding(
            get: { model.pendingShareURL.map { ShareItemCombo(url: $0) } },
            set: { _ in model.pendingShareURL = nil }
        )) { item in
            ActivityViewCombo(activityItems: [item.url])
        }
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack {
            Text(label).foregroundColor(.appTextSecondary)
            Spacer()
            Text(value).font(.headline).foregroundColor(.appTextPrimary)
        }
    }

    private func drawTrajectories(context: GraphicsContext, size: CGSize) {
        guard let batch = model.currentBatch, !batch.points.isEmpty else { return }

        var allX: [Double] = []
        var allY: [Double] = []
        for method in HeadingMethodID.allCases {
            for point in batch.points {
                if let p = point.positions[method.rawValue] {
                    allX.append(p.x); allY.append(p.y)
                }
            }
        }
        if let gps = batch.gpsOverlayPoints {
            allX.append(contentsOf: gps.map(\.x))
            allY.append(contentsOf: gps.map(\.y))
        }
        let liveGPS = SensorComboTestModel.convertGPSSamplesToLocal(batch.gpsSamples)
        allX.append(contentsOf: liveGPS.map(\.x))
        allY.append(contentsOf: liveGPS.map(\.y))
        guard !allX.isEmpty else { return }

        let minX = allX.min() ?? 0, maxX = allX.max() ?? 0
        let minY = allY.min() ?? 0, maxY = allY.max() ?? 0
        let spanX = max(maxX - minX, 1), spanY = max(maxY - minY, 1)
        let padding: CGFloat = 24
        let usableW = size.width - padding * 2, usableH = size.height - padding * 2
        let scale = min(usableW / CGFloat(spanX), usableH / CGFloat(spanY))
        let drawnW = CGFloat(spanX) * scale, drawnH = CGFloat(spanY) * scale
        let offsetX = padding + (usableW - drawnW) / 2
        let offsetY = padding + (usableH - drawnH) / 2

        func canvasPoint(_ x: Double, _ y: Double) -> CGPoint {
            CGPoint(x: offsetX + CGFloat(x - minX) * scale, y: size.height - (offsetY + CGFloat(y - minY) * scale))
        }

        for method in HeadingMethodID.allCases {
            var path = Path()
            var started = false
            for point in batch.points {
                guard let p = point.positions[method.rawValue] else { continue }
                let cp = canvasPoint(p.x, p.y)
                if !started { path.move(to: cp); started = true } else { path.addLine(to: cp) }
            }
            context.stroke(path, with: .color(Self.methodColors[method] ?? .gray), lineWidth: 2)
        }

        if let gps = batch.gpsOverlayPoints, !gps.isEmpty {
            var path = Path()
            for (i, p) in gps.enumerated() {
                let cp = canvasPoint(p.x, p.y)
                if i == 0 { path.move(to: cp) } else { path.addLine(to: cp) }
            }
            context.stroke(path, with: .color(.black), style: StrokeStyle(lineWidth: 2, dash: [5, 4]))
        }

        if !liveGPS.isEmpty {
            var path = Path()
            for (i, p) in liveGPS.enumerated() {
                let cp = canvasPoint(p.x, p.y)
                if i == 0 { path.move(to: cp) } else { path.addLine(to: cp) }
            }
            context.stroke(path, with: .color(.gray), lineWidth: 2)
        }
    }
}

private struct ShareItemCombo: Identifiable {
    let url: URL
    var id: String { url.path }
}

/// Wraps a whole batch's worth of waypoint-photo file URLs for one `.sheet(item:)` share —
/// see `SensorComboTestModel.photoShareURLs(batchId:)`.
private struct SharePhotosItem: Identifiable {
    let urls: [URL]
    var id: String { urls.map(\.path).joined() }
}

private struct ActivityViewCombo: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

@MainActor
final class SensorComboTestModel: ObservableObject {
    @Published private(set) var isTracking = false
    @Published private(set) var stepCount = 0
    @Published private(set) var distanceMeters: Double = 0
    @Published private(set) var currentHeadings: [HeadingMethodID: Double] = [:]
    @Published private(set) var batches: [SensorComboBatch] = []
    @Published private(set) var gaitProfile: GaitProfile
    @Published private(set) var workouts: [HKWorkout] = []
    @Published private(set) var isLoadingWorkouts = false
    @Published private(set) var healthKitError: String?
    @Published var pendingShareURL: URL?
    @Published var pendingSharePhotoURLs: [URL]?

    @Published private(set) var isGPSTracking = false
    @Published private(set) var gpsSampleCount = 0
    @Published private(set) var gpsAuthorizationError: String?

    private let tracker = SensorComboTracker()
    private let gpsLogger = GPSLogger()
    private let storageKey = "sensor_combo_batches.json"
    private let pendingGPSStorageKey = "sensor_combo_pending_gps.json"
    private var currentBatchId: String?
    private var pendingWaypointSnapshot: (timestamp: Date, stepCount: Int, distanceMeters: Double, positions: [HeadingMethodID: (x: Double, y: Double)])?
    private var rawSampleCountSinceLastPersist = 0
    /// GPS samples that arrive while `startGPS()` has been called but no batch is active yet
    /// (e.g. the user starts GPS a few minutes before "開始新的一輪") — carried over into
    /// whichever batch starts next instead of being dropped. Persisted to its own file (see
    /// `persistPendingGPS()`) the moment each sample arrives, not just held in memory — this
    /// used to be memory-only, so any GPS collected before a batch existed was gone for good
    /// the instant the screen was closed or the app was backgrounded/killed by iOS, which is
    /// very likely what "資料被刪掉了" was actually seeing.
    private var pendingGPSSamplesBeforeBatch: [GPSSample] = []

    var currentBatch: SensorComboBatch? {
        guard let currentBatchId else { return nil }
        return batches.first { $0.id == currentBatchId }
    }

    private var currentBatchIndex: Int? {
        guard let currentBatchId else { return nil }
        return batches.firstIndex { $0.id == currentBatchId }
    }

    init() {
        batches = PersistenceService.shared.load([SensorComboBatch].self, from: storageKey) ?? []
        gaitProfile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
        pendingGPSSamplesBeforeBatch = PersistenceService.shared.load([GPSSample].self, from: pendingGPSStorageKey) ?? []

        tracker.onHeadingsUpdate = { [weak self] headings in
            self?.currentHeadings = headings
        }
        tracker.onPathPoint = { [weak self] point in
            guard let self, let index = self.currentBatchIndex else { return }
            self.stepCount = point.stepCount
            self.distanceMeters = point.distanceMeters
            self.batches[index].points.append(point)
            self.rawSampleCountSinceLastPersist += 1
            if self.rawSampleCountSinceLastPersist >= 5 {
                self.rawSampleCountSinceLastPersist = 0
                self.persist()
            }
        }

        // GPS arrives at roughly 1Hz, not the ~5-10Hz of the sensor stream above, so
        // persisting on every single sample (instead of throttled like `onPathPoint`) is
        // cheap and closes the loss window entirely — nothing sits unsaved in memory for
        // more than an instant.
        gpsLogger.onLocation = { [weak self] sample in
            guard let self else { return }
            self.gpsSampleCount += 1
            if let index = self.currentBatchIndex {
                self.batches[index].gpsSamples.append(sample)
                self.persist()
            } else {
                self.pendingGPSSamplesBeforeBatch.append(sample)
                self.persistPendingGPS()
            }
        }
        gpsLogger.onAuthorizationDenied = { [weak self] in
            self?.gpsAuthorizationError = "沒有定位權限，請到「設定」開啟這個 App 的定位權限"
            self?.isGPSTracking = false
        }
    }

    // MARK: - Live GPS (independent start/stop, see gpsLiveCard's doc text)

    func startGPS() {
        guard !isGPSTracking else { return }
        gpsAuthorizationError = nil
        gpsLogger.start()
        // Read back the *actual* result instead of assuming success: when permission is
        // denied/restricted, `gpsLogger.start()` synchronously fires `onAuthorizationDenied`
        // (wired below to set `isGPSTracking = false` + the error text) before returning —
        // but this line used to unconditionally stomp that right back to `true` immediately
        // after, so the button displayed "停止 GPS 記錄" and looked like it was recording
        // even though `CLLocationManager` never actually started and no samples ever arrived.
        // Mirroring `gpsLogger.isTracking` here means the UI only ever shows what's real.
        isGPSTracking = gpsLogger.isTracking
    }

    func stopGPS() {
        guard isGPSTracking else { return }
        gpsLogger.stop()
        isGPSTracking = false
        persist()
    }

    /// Stops *everything* — the sensor tracker and the independent GPS logger — and flushes
    /// both to disk. Called when this screen disappears (see `SensorComboTestView`'s
    /// `.onDisappear`): GPS recording isn't truly backgroundable (no background-location
    /// entitlement, and the `@StateObject` this all lives on gets torn down once the screen
    /// closes), so leaving it "running" when the screen goes away used to mean it silently
    /// stopped delivering updates anyway, while looking to a freshly-reopened screen like it
    /// had never been started — coming back never showed it as still tracking, and anything
    /// not yet flushed to disk was gone. Stopping and persisting cleanly here instead means
    /// the state on screen next time always matches what's actually on disk.
    func stopAll() {
        stop()
        if isGPSTracking { stopGPS() }
    }

    static func convertGPSSamplesToLocal(_ samples: [GPSSample]) -> [(x: Double, y: Double)] {
        guard let origin = samples.first else { return [] }
        let earthRadius = 6_371_000.0
        let originLatRadians = origin.latitude * .pi / 180
        return samples.map { s in
            let dLat = (s.latitude - origin.latitude) * .pi / 180
            let dLon = (s.longitude - origin.longitude) * .pi / 180
            let y = dLat * earthRadius
            let x = dLon * earthRadius * cos(originLatRadians)
            return (x, y)
        }
    }

    func exportGPSCSV(batchId: String) -> URL? {
        guard let batch = batches.first(where: { $0.id == batchId }), !batch.gpsSamples.isEmpty else { return nil }
        var csv = "# SensorComboTest live GPS export — \(batch.startedAtDisplay)\n"
        csv += "timestamp_ms,latitude,longitude,altitude_m,horizontal_accuracy_m,speed_mps,course_deg\n"
        for s in batch.gpsSamples {
            csv += String(format: "%.0f,%.7f,%.7f,%.2f,%.2f,%.2f,%.1f\n", s.timestamp, s.latitude, s.longitude, s.altitude, s.horizontalAccuracy, s.speed, s.course)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sensor_combo_gps_\(batch.id).csv")
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            return nil
        }
    }

    func startNewBatch() {
        guard !isTracking, SensorComboTracker.isAvailable else { return }
        tracker.reloadGaitProfile()
        tracker.reloadGyroBias()
        gaitProfile = PersistenceService.shared.load(GaitProfile.self, from: "gait_profile.json") ?? .identity
        isTracking = true
        stepCount = 0
        distanceMeters = 0

        var batch = SensorComboBatch(id: UUID().uuidString, startedAt: Date().timeIntervalSince1970 * 1000)
        if !pendingGPSSamplesBeforeBatch.isEmpty {
            batch.gpsSamples = pendingGPSSamplesBeforeBatch
            pendingGPSSamplesBeforeBatch = []
            persistPendingGPS()
        }
        batches.append(batch)
        currentBatchId = batch.id
        persist()
        tracker.start()

        // Auto-bind GPS to this round's lifetime — see `stop()`'s matching half. Only if it
        // isn't already running: if the user manually started GPS early (e.g. to let it get a
        // fix before walking), that's left alone rather than being restarted out from under
        // them. This used to be fully manual/independent, which meant GPS samples got routed
        // by whatever `currentBatchId` happened to be at the moment they arrived — technically
        // correct, but easy to end up with GPS spanning the wrong round's boundaries (started
        // too early, stopped too late) since nothing tied its start/stop to the round itself.
        if !isGPSTracking { startGPS() }
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        tracker.stop()
        // Matching half of `startNewBatch()`'s auto-start — ends this round's GPS window
        // right when the round itself ends, so a round's `gpsSamples` never bleed into
        // whatever comes after it.
        if isGPSTracking { stopGPS() }
        rawSampleCountSinceLastPersist = 0
        persist()
    }

    /// Manually confirmed checkpoint — see `ComboCheckpoint`'s doc comment. Records this
    /// exact instant's position for every method against the current step count / distance,
    /// same underlying snapshot `markLap` uses.
    func markCheckpoint() {
        guard isTracking, let index = currentBatchIndex else { return }
        var positions: [String: PointXY] = [:]
        for (method, pos) in tracker.currentPositionsSnapshot() {
            positions[method.rawValue] = PointXY(x: pos.x, y: pos.y)
        }
        appendCheckpoint(timestamp: Date().timeIntervalSince1970 * 1000, stepCount: stepCount,
                          distanceMeters: distanceMeters, positions: positions, waypointIndex: nil, batchIndex: index)
        persist()
    }

    private func appendCheckpoint(timestamp: Double, stepCount: Int, distanceMeters: Double,
                                   positions: [String: PointXY], waypointIndex: Int?, batchIndex: Int) {
        batches[batchIndex].checkpoints.append(ComboCheckpoint(
            id: UUID().uuidString,
            stepCount: stepCount,
            timestamp: timestamp,
            distanceMeters: distanceMeters,
            positions: positions,
            waypointIndex: waypointIndex
        ))
    }

    // MARK: - Checkpoint accuracy (vs. live GPS)

    /// Nearest-timestamp GPS ground truth, in the same local-meter frame the PDR
    /// trajectories use (see `convertGPSSamplesToLocal`). Recomputed on demand — GPS keeps
    /// arriving after a checkpoint is recorded, so this always matches against the complete
    /// track, not just what had arrived by record-time.
    static func gpsGroundTruth(atTimestamp timestamp: Double, in batch: SensorComboBatch) -> (x: Double, y: Double)? {
        guard !batch.gpsSamples.isEmpty else { return nil }
        let local = convertGPSSamplesToLocal(batch.gpsSamples)
        var best: (dt: Double, pos: (x: Double, y: Double))?
        for (sample, pos) in zip(batch.gpsSamples, local) {
            let dt = abs(sample.timestamp - timestamp)
            if best == nil || dt < best!.dt {
                best = (dt, pos)
            }
        }
        return best?.pos
    }

    static func gpsGroundTruth(for checkpoint: ComboCheckpoint, in batch: SensorComboBatch) -> (x: Double, y: Double)? {
        gpsGroundTruth(atTimestamp: checkpoint.timestamp, in: batch)
    }

    /// Straight-line distance (metres) between one method's estimated position at some moment
    /// and the GPS ground truth matched to that same moment — nil when there's no GPS to
    /// compare against.
    static func error(atTimestamp timestamp: Double, positions: [String: PointXY], method: HeadingMethodID, in batch: SensorComboBatch) -> Double? {
        guard let truth = gpsGroundTruth(atTimestamp: timestamp, in: batch), let p = positions[method.rawValue] else { return nil }
        let dx = p.x - truth.x, dy = p.y - truth.y
        return (dx * dx + dy * dy).squareRoot()
    }

    static func error(for checkpoint: ComboCheckpoint, method: HeadingMethodID, in batch: SensorComboBatch) -> Double? {
        error(atTimestamp: checkpoint.timestamp, positions: checkpoint.positions, method: method, in: batch)
    }

    /// See `ComboLap`'s doc comment — purely a diagnostic mark, applies no correction to
    /// any of the five trajectories. Reads the same position snapshot `beginWaypointCapture`
    /// uses, just for all five methods instead of one.
    func markLap() {
        guard isTracking, let index = currentBatchIndex else { return }
        var positions: [String: PointXY] = [:]
        for (method, pos) in tracker.currentPositionsSnapshot() {
            positions[method.rawValue] = PointXY(x: pos.x, y: pos.y)
        }
        let lap = ComboLap(
            id: UUID().uuidString,
            lapNumber: batches[index].laps.count + 1,
            timestamp: Date().timeIntervalSince1970 * 1000,
            positions: positions
        )
        batches[index].laps.append(lap)
        persist()
    }

    /// Captures the position snapshot at button-tap time, before the camera sheet even
    /// opens — see `SensorComboTracker.currentPositionsSnapshot()`'s doc comment. Also grabs
    /// the step count / distance at that same instant so `finishWaypointCapture` can log a
    /// matching `ComboCheckpoint`, not just the `ComboWaypoint`.
    func beginWaypointCapture() {
        guard isTracking else { return }
        pendingWaypointSnapshot = (Date(), stepCount, distanceMeters, tracker.currentPositionsSnapshot())
    }

    func finishWaypointCapture(image: UIImage?) {
        guard let snapshot = pendingWaypointSnapshot, let index = currentBatchIndex else {
            pendingWaypointSnapshot = nil
            return
        }
        pendingWaypointSnapshot = nil

        let waypointId = UUID().uuidString
        var positions: [String: PointXY] = [:]
        for (method, pos) in snapshot.positions {
            positions[method.rawValue] = PointXY(x: pos.x, y: pos.y)
        }

        var photoFileName: String?
        if let image, let data = image.jpegData(compressionQuality: 0.8) {
            let fileName = "\(waypointId).jpg"
            let dir = Self.photosDirectory(batchId: batches[index].id)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try? data.write(to: dir.appendingPathComponent(fileName))
            photoFileName = fileName
        }

        let waypointIndex = batches[index].waypoints.count + 1
        let waypoint = ComboWaypoint(
            id: waypointId,
            index: waypointIndex,
            timestamp: snapshot.timestamp.timeIntervalSince1970 * 1000,
            photoFileName: photoFileName,
            positions: positions
        )
        batches[index].waypoints.append(waypoint)
        // A photo mark doubles as an accuracy checkpoint — see `ComboCheckpoint`'s doc
        // comment — so it shows up in "檢查點誤差" without a separate "記錄檢查點" press.
        appendCheckpoint(timestamp: waypoint.timestamp, stepCount: snapshot.stepCount,
                          distanceMeters: snapshot.distanceMeters, positions: positions,
                          waypointIndex: waypointIndex, batchIndex: index)
        persist()
    }

    func loadWaypointImage(batchId: String, fileName: String) -> UIImage? {
        let url = Self.photosDirectory(batchId: batchId).appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }

    /// File URLs for every waypoint photo in this batch, in waypoint order — handed straight
    /// to `UIActivityViewController` (AirDrop, Save to Files, etc.). These are the same files
    /// already sitting in the batch's own sandboxed photos folder, not copies, so nothing
    /// extra gets written to disk just to share them.
    func photoShareURLs(batchId: String) -> [URL] {
        guard let batch = batches.first(where: { $0.id == batchId }) else { return [] }
        let dir = Self.photosDirectory(batchId: batchId)
        return batch.waypoints.compactMap { $0.photoFileName }.map { dir.appendingPathComponent($0) }
    }

    private static func photosDirectory(batchId: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("sensor_combo_photos")
            .appendingPathComponent(batchId)
    }

    func groundTruthStepsBinding(_ batchId: String) -> Binding<String> {
        batchStringBinding(batchId: batchId, keyPath: \.groundTruthSteps)
    }
    func groundTruthDistanceBinding(_ batchId: String) -> Binding<String> {
        batchStringBinding(batchId: batchId, keyPath: \.groundTruthDistanceM)
    }
    private func batchStringBinding(batchId: String, keyPath: WritableKeyPath<SensorComboBatch, String>) -> Binding<String> {
        Binding(
            get: { self.batches.first(where: { $0.id == batchId })?[keyPath: keyPath] ?? "" },
            set: { newValue in
                guard let idx = self.batches.firstIndex(where: { $0.id == batchId }) else { return }
                self.batches[idx][keyPath: keyPath] = newValue
                self.persist()
            }
        )
    }

    // MARK: - GPS overlay import (HealthKit / Apple Watch workout route)

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

    func importGPSOverlay(_ workout: HKWorkout) {
        guard let index = currentBatchIndex else { return }
        Task {
            do {
                let locations = try await HealthKitService.shared.fetchRoute(for: workout)
                await MainActor.run {
                    guard !locations.isEmpty else {
                        self.healthKitError = "這筆訓練沒有 GPS 路徑資料"
                        return
                    }
                    self.batches[index].gpsOverlayPoints = HealthKitService.shared.convertToLocalPoints(locations)
                    self.persist()
                }
            } catch {
                await MainActor.run {
                    self.healthKitError = "匯入失敗: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - CSV export

    func exportCSV(batchId: String) -> URL? {
        guard let batch = batches.first(where: { $0.id == batchId }), let first = batch.points.first else { return nil }

        var csv = "# SensorComboTest Export — \(batch.startedAtDisplay)\n"
        csv += "# ground_truth_steps=\(batch.groundTruthSteps),ground_truth_distance_m=\(batch.groundTruthDistanceM)\n"
        for waypoint in batch.waypoints {
            let posText = HeadingMethodID.allCases.map { m -> String in
                let p = waypoint.positions[m.rawValue]
                return "\(m.rawValue)=(\(String(format: "%.2f", p?.x ?? 0)),\(String(format: "%.2f", p?.y ?? 0)))"
            }.joined(separator: " ")
            csv += "# waypoint \(waypoint.index): t=\(waypoint.timestamp) photo=\(waypoint.photoFileName ?? "none") \(posText)\n"
        }
        for lap in batch.laps {
            let posText = HeadingMethodID.allCases.map { m -> String in
                let p = lap.positions[m.rawValue]
                return "\(m.rawValue)=(\(String(format: "%.2f", p?.x ?? 0)),\(String(format: "%.2f", p?.y ?? 0)))"
            }.joined(separator: " ")
            csv += "# lap \(lap.lapNumber): t=\(lap.timestamp) \(posText)\n"
        }
        for checkpoint in batch.checkpoints {
            let posText = HeadingMethodID.allCases.map { m -> String in
                let p = checkpoint.positions[m.rawValue]
                let err = SensorComboTestModel.error(for: checkpoint, method: m, in: batch)
                let errText = err.map { String(format: "%.2f", $0) } ?? "n/a"
                return "\(m.rawValue)=(\(String(format: "%.2f", p?.x ?? 0)),\(String(format: "%.2f", p?.y ?? 0)))err=\(errText)m"
            }.joined(separator: " ")
            let waypointTag = checkpoint.waypointIndex.map { " waypoint=\($0)" } ?? ""
            csv += "# checkpoint step=\(checkpoint.stepCount) dist=\(String(format: "%.2f", checkpoint.distanceMeters))m t=\(checkpoint.timestamp)\(waypointTag) \(posText)\n"
        }

        let methods = HeadingMethodID.allCases
        var header = "timestamp_ms,elapsed_sec,step_count,distance_m"
        for m in methods { header += ",heading_\(m.rawValue),x_\(m.rawValue),y_\(m.rawValue)" }
        csv += header + "\n"

        for point in batch.points {
            let elapsed = (point.timestamp - first.timestamp) / 1000
            var row = String(format: "%.0f,%.3f,%d,%.3f", point.timestamp, elapsed, point.stepCount, point.distanceMeters)
            for m in methods {
                let h = point.headings[m.rawValue] ?? 0
                let p = point.positions[m.rawValue] ?? PointXY(x: 0, y: 0)
                row += String(format: ",%.1f,%.3f,%.3f", h, p.x, p.y)
            }
            csv += row + "\n"
        }

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("sensor_combo_\(batch.id).csv")
        do {
            try csv.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            return nil
        }
    }

    private func persist() {
        PersistenceService.shared.save(batches, to: storageKey)
    }

    private func persistPendingGPS() {
        PersistenceService.shared.save(pendingGPSSamplesBeforeBatch, to: pendingGPSStorageKey)
    }
}
