import Combine
import SwiftUI

struct GPSBatch: Codable, Identifiable {
    let id: String
    var startedAt: Double
    var samples: [GPSSample] = []

    var startedAtDisplay: String {
        let date = Date(timeIntervalSince1970: startedAt / 1000)
        let formatter = DateFormatter()
        formatter.dateFormat = "M/d HH:mm:ss"
        return formatter.string(from: date)
    }
}

/// Standalone GPS recorder — deliberately has nothing to do with the PDR sensor trackers
/// (see `GPSLogger`'s doc comment). Run this alongside `SensorComboTestView` (or anything
/// else) as an independent ground-truth reference; it never reads from or writes to any
/// other tracker's state.
struct GPSLoggerView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = GPSLoggerModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    statusCard
                    controlsCard
                    historyCard
                }
                .padding(20)
                .padding(.top, 4)
            }
            .background(Color.appBg)
            .navigationTitle("GPS 記錄（獨立）")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("完成") { dismiss() }
                }
            }
        }
        .onDisappear { model.stop() }
    }

    private var statusCard: some View {
        VStack(spacing: 12) {
            metricRow(label: "狀態", value: model.isTracking ? "記錄中" : "已停止")
            metricRow(label: "已記錄點數", value: "\(model.currentBatch?.samples.count ?? 0)")
            if let last = model.currentBatch?.samples.last {
                metricRow(label: "最新座標", value: String(format: "%.5f, %.5f", last.latitude, last.longitude))
                metricRow(label: "水平精確度", value: String(format: "±%.0f 公尺", last.horizontalAccuracy))
            }
            if let error = model.authorizationError {
                Text(error).font(.caption).foregroundColor(.red)
            }
        }
        .padding(20)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var controlsCard: some View {
        VStack(spacing: 10) {
            Button(model.isTracking ? "停止記錄" : "開始記錄") {
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

            Text("這個記錄跟計步器測試／感測器組合測試完全獨立，互不影響，可以同時開著跑，事後拿這份 GPS 軌跡當地面真值比對。室內收不到 GPS，這個工具只適合戶外。")
                .font(.caption2)
                .foregroundColor(.appTextTertiary)
                .multilineTextAlignment(.center)
        }
        .padding(20)
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
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(batch.startedAtDisplay)
                            .font(.caption.weight(.medium))
                            .foregroundColor(.appTextPrimary)
                        Text("\(batch.samples.count) 個點")
                            .font(.caption2)
                            .foregroundColor(.appTextSecondary)
                    }
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
                .padding(.vertical, 4)
            }
        }
        .padding(16)
        .background(Color.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .sheet(item: Binding(
            get: { model.pendingShareURL.map { ShareItemGPS(url: $0) } },
            set: { _ in model.pendingShareURL = nil }
        )) { item in
            ActivityViewGPS(activityItems: [item.url])
        }
    }

    private func metricRow(label: String, value: String) -> some View {
        HStack {
            Text(label).foregroundColor(.appTextSecondary)
            Spacer()
            Text(value).font(.headline).foregroundColor(.appTextPrimary)
        }
    }
}

private struct ShareItemGPS: Identifiable {
    let url: URL
    var id: String { url.path }
}

private struct ActivityViewGPS: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

@MainActor
final class GPSLoggerModel: ObservableObject {
    @Published private(set) var isTracking = false
    @Published private(set) var batches: [GPSBatch] = []
    @Published private(set) var authorizationError: String?
    @Published var pendingShareURL: URL?

    private let logger = GPSLogger()
    private let storageKey = "gps_logger_batches.json"
    private var currentBatchId: String?
    private var sampleCountSinceLastPersist = 0

    var currentBatch: GPSBatch? {
        guard let currentBatchId else { return nil }
        return batches.first { $0.id == currentBatchId }
    }

    init() {
        batches = PersistenceService.shared.load([GPSBatch].self, from: storageKey) ?? []

        logger.onLocation = { [weak self] sample in
            guard let self, let index = self.currentBatchIndex else { return }
            self.batches[index].samples.append(sample)
            self.sampleCountSinceLastPersist += 1
            if self.sampleCountSinceLastPersist >= 5 {
                self.sampleCountSinceLastPersist = 0
                self.persist()
            }
        }
        logger.onAuthorizationDenied = { [weak self] in
            self?.authorizationError = "沒有定位權限，請到「設定」開啟這個 App 的定位權限"
            self?.isTracking = false
        }
    }

    private var currentBatchIndex: Int? {
        guard let currentBatchId else { return nil }
        return batches.firstIndex { $0.id == currentBatchId }
    }

    func start() {
        guard !isTracking else { return }
        authorizationError = nil
        isTracking = true
        let batch = GPSBatch(id: UUID().uuidString, startedAt: Date().timeIntervalSince1970 * 1000)
        batches.append(batch)
        currentBatchId = batch.id
        persist()
        logger.start()
    }

    func stop() {
        guard isTracking else { return }
        isTracking = false
        logger.stop()
        sampleCountSinceLastPersist = 0
        persist()
    }

    func exportCSV(batchId: String) -> URL? {
        guard let batch = batches.first(where: { $0.id == batchId }), !batch.samples.isEmpty else { return nil }
        var csv = "# GPSLogger Export — \(batch.startedAtDisplay)\n"
        csv += "timestamp_ms,latitude,longitude,altitude_m,horizontal_accuracy_m,speed_mps,course_deg\n"
        for s in batch.samples {
            csv += String(format: "%.0f,%.7f,%.7f,%.2f,%.2f,%.2f,%.1f\n", s.timestamp, s.latitude, s.longitude, s.altitude, s.horizontalAccuracy, s.speed, s.course)
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("gps_log_\(batch.id).csv")
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
}
