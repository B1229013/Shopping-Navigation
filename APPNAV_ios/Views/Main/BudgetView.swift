import SwiftUI
import PhotosUI
import Vision

enum BudgetViewMode: String, CaseIterable {
    case category = "分類"
    case receipt = "收據"
    case day = "日"
    case month = "月"
}

struct BudgetView: View {
    @Binding var shoppingItems: [ShoppingItem]
    @Binding var budgetTotal: String

    @State private var viewMode: BudgetViewMode = .category
    @State private var expandedId: String? = nil
    @State private var showAddSheet = false
    @State private var showBudgetDialog = false
    @State private var tempBudget = ""
    @State private var isProcessing = false
    @State private var showSuccess = false
    @State private var extractedCount = 0
    @State private var processingError: String? = nil

    // Photo picker
    @State private var selectedPhotoItem: PhotosPickerItem? = nil
    @State private var showCamera = false

    // API keys — Settings' @AppStorage overrides take priority over the Info.plist default
    private let groqApiKey = UserDefaults.standard.string(forKey: "groqApiKey") ?? AppConfig.groqApiKey
    private let paddleOcrApiUrl = UserDefaults.standard.string(forKey: "paddleOcrApiUrl") ?? AppConfig.paddleOcrApiUrl
    private let paddleOcrToken = UserDefaults.standard.string(forKey: "paddleOcrToken") ?? AppConfig.paddleOcrToken

    private var purchasedItems: [ShoppingItem] { shoppingItems.filter { $0.isChecked } }
    private var totalSpent: Int { purchasedItems.reduce(0) { $0 + $1.price * $1.qty } }
    private var budgetValue: Int { Int(budgetTotal) ?? 0 }

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                headerRow
                donutSection
                    .padding(.horizontal, 20)
                filterChips
                    .padding(.top, 16)

                Divider().padding(.vertical, 8)

                contentArea
                    .padding(.bottom, 24)

                if isProcessing { processingBanner }
                if showSuccess { successBanner }
                if let err = processingError { errorBanner(err) }
            }
            .adaptiveWidth(760)
        }
        .background(Color.appBg)
        .sheet(isPresented: $showCamera) {
            CameraPickerView { image in
                if let image { processImage(image) }
            }
        }
        .sheet(isPresented: $showAddSheet) {
            addOptionsSheet
                .presentationDetents([.height(200)])
        }
        .photosPicker(isPresented: .constant(false), selection: $selectedPhotoItem, matching: .images)
        .onChange(of: selectedPhotoItem) { newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    processImage(image)
                }
            }
        }
        .sheet(isPresented: $showBudgetDialog) { budgetSheet }
    }

    // MARK: - Header
    private var headerRow: some View {
        HStack {
            Text("預算")
                .font(.largeTitle.bold())
                .foregroundColor(.appTextPrimary)
            Spacer()
            Button {
                showAddSheet = true
            } label: {
                Text("新增")
                    .font(.headline)
                    .foregroundColor(.appGold)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color.appGold.opacity(0.1))
                    .cornerRadius(10)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 8)
    }

    // MARK: - Donut Chart
    private var donutSection: some View {
        ZStack {
            DonutChartView(items: shoppingItems)
                .frame(width: 200, height: 200)
            VStack(spacing: 2) {
                Text("已支出")
                    .font(.caption)
                    .foregroundColor(.appTextSecondary)
                Text("$\(totalSpent.formattedWithCommas)")
                    .font(.title2.bold())
                    .foregroundColor(.appTextPrimary)
                Text(Date().monthDisplayString)
                    .font(.caption)
                    .foregroundColor(.appTextTertiary)
            }
        }
        .frame(height: 200)
        .cardStyle()
        .padding(.vertical, 8)
    }

    // MARK: - Filter Chips
    private var filterChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(BudgetViewMode.allCases, id: \.self) { mode in
                    let isSelected = viewMode == mode
                    Text(mode.rawValue)
                        .font(.subheadline)
                        .foregroundColor(isSelected ? .appGold : .appTextSecondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(isSelected ? Color.appGold.opacity(0.12) : Color.appSurface)
                        .cornerRadius(20)
                        .overlay(RoundedRectangle(cornerRadius: 20).stroke(isSelected ? Color.appGold.opacity(0.3) : Color.appBorder, lineWidth: 1))
                        .onTapGesture { withAnimation { viewMode = mode; expandedId = nil } }
                }
                Button {
                    showBudgetDialog = true
                } label: {
                    Text("預算: $\(budgetTotal.isEmpty ? "0" : budgetTotal)")
                        .font(.subheadline)
                        .foregroundColor(.appTextTertiary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Color.appSurface)
                        .cornerRadius(20)
                        .overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.appBorder, lineWidth: 1))
                }
            }
            .padding(.horizontal, 20)
        }
    }

    // MARK: - Content Area
    @ViewBuilder
    private var contentArea: some View {
        switch viewMode {
        case .category: categoryView
        case .receipt: receiptView
        case .day: dayView
        case .month: monthView
        }
    }

    // MARK: - Category View
    private var categoryView: some View {
        VStack(spacing: 10) {
            ForEach(dashboardCategories) { cat in
                let catItems = purchasedItems.filter { $0.location == cat.catId }
                let catSpent = catItems.reduce(0) { $0 + $1.price * $1.qty }
                let catBudget = budgetValue > 0 ? budgetValue / dashboardCategories.count : 0
                let catLeft = catBudget - catSpent
                let isExpanded = expandedId == cat.catId

                CategoryBudgetRow(
                    category: cat,
                    spent: catSpent,
                    left: catLeft,
                    budgetPerCategory: catBudget,
                    isExpanded: isExpanded,
                    items: catItems,
                    onToggle: { withAnimation { expandedId = isExpanded ? nil : cat.catId } }
                )
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
    }

    // MARK: - Receipt View
    private var receiptView: some View {
        VStack(spacing: 10) {
            let groups = Dictionary(grouping: purchasedItems) { item in
                item.receiptId ?? "manual_\(Int(item.createdAt / 60000))"
            }
            let sorted = groups.sorted { a, b in
                (a.value.first?.createdAt ?? 0) > (b.value.first?.createdAt ?? 0)
            }

            if sorted.isEmpty {
                EmptyBudgetState(title: "尚無收據紀錄", subtitle: "掃描收據以開始追蹤")
            } else {
                ForEach(sorted, id: \.key) { groupId, groupItems in
                    let store = groupItems.first?.storeName ?? "手動輸入"
                    let date = Date.from(milliseconds: groupItems.first?.createdAt ?? 0).shortDateString
                    let total = groupItems.reduce(0) { $0 + $1.price * $1.qty }
                    let isExpanded = expandedId == groupId

                    ExpandableCard(
                        icon: "receipt",
                        title: store,
                        subtitle: date,
                        trailingValue: "$\(total.formattedWithCommas)",
                        trailingCaption: "\(groupItems.count) 項",
                        isExpanded: isExpanded,
                        accentColor: .appGold
                    ) {
                        withAnimation { expandedId = isExpanded ? nil : groupId }
                    } content: {
                        ForEach(groupItems) { item in ItemDetailRow(item: item) }
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
    }

    // MARK: - Day View
    private var dayView: some View {
        VStack(spacing: 10) {
            let groups = Dictionary(grouping: purchasedItems) { item in
                Date.from(milliseconds: item.createdAt).shortDateString
            }
            let sorted = groups.sorted { $0.key > $1.key }

            if sorted.isEmpty {
                EmptyBudgetState(title: "尚無消費紀錄", subtitle: "掃描收據以開始追蹤")
            } else {
                ForEach(sorted, id: \.key) { dayKey, dayItems in
                    let dayTotal = dayItems.reduce(0) { $0 + $1.price * $1.qty }
                    let isExpanded = expandedId == dayKey

                    ExpandableCard(
                        icon: "calendar",
                        title: dayKey,
                        subtitle: "\(dayItems.count) 筆消費",
                        trailingValue: "$\(dayTotal.formattedWithCommas)",
                        trailingCaption: nil,
                        isExpanded: isExpanded,
                        accentColor: .chartBlue
                    ) {
                        withAnimation { expandedId = isExpanded ? nil : dayKey }
                    } content: {
                        ForEach(dayItems) { item in ItemDetailRow(item: item) }
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
    }

    // MARK: - Month View
    private var monthView: some View {
        VStack(spacing: 10) {
            let groups = Dictionary(grouping: purchasedItems) { item in
                Date.from(milliseconds: item.createdAt).monthKey
            }
            let sorted = groups.sorted { $0.key > $1.key }

            if sorted.isEmpty {
                EmptyBudgetState(title: "尚無消費紀錄", subtitle: "掃描收據以開始追蹤")
            } else {
                ForEach(sorted, id: \.key) { monthKey, monthItems in
                    let monthTotal = monthItems.reduce(0) { $0 + $1.price * $1.qty }
                    let isExpanded = expandedId == monthKey

                    ExpandableCard(
                        icon: "calendar.badge.clock",
                        title: monthKey,
                        subtitle: "\(monthItems.count) 筆消費",
                        trailingValue: "$\(monthTotal.formattedWithCommas)",
                        trailingCaption: nil,
                        isExpanded: isExpanded,
                        accentColor: .chartGreen
                    ) {
                        withAnimation { expandedId = isExpanded ? nil : monthKey }
                    } content: {
                        ForEach(monthItems) { item in ItemDetailRow(item: item) }
                    }
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 8)
    }

    // MARK: - Add Options Sheet
    private var addOptionsSheet: some View {
        VStack(spacing: 12) {
            Text("新增支出").font(.headline).padding(.top, 16)
            Button {
                showAddSheet = false
                showCamera = true
            } label: {
                Label("掃描收據", systemImage: "camera.fill")
                    .frame(maxWidth: .infinity).padding().background(Color.appGold).foregroundColor(.black).cornerRadius(14)
            }
            PhotosPicker(selection: $selectedPhotoItem, matching: .images) {
                Label("從相簿選取", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity).padding().background(Color.appSurface).foregroundColor(.appTextPrimary).cornerRadius(14)
                    .overlay(RoundedRectangle(cornerRadius: 14).stroke(Color.appBorder, lineWidth: 1))
            }
            .onChange(of: selectedPhotoItem) { _ in showAddSheet = false }
            .padding(.bottom, 16)
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Budget Sheet
    private var budgetSheet: some View {
        NavigationView {
            VStack(spacing: 16) {
                Text("設定每月預算總額:")
                    .font(.subheadline)
                    .foregroundColor(.appTextSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                TextField("預算金額", text: $tempBudget)
                    .keyboardType(.numberPad)
                    .padding(14)
                    .background(Color.appBg)
                    .cornerRadius(14)
                Spacer()
            }
            .padding()
            .navigationTitle("預算設定")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("儲存") {
                        budgetTotal = tempBudget
                        showBudgetDialog = false
                    }.foregroundColor(.appGold)
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { showBudgetDialog = false }
                }
            }
            .onAppear { tempBudget = budgetTotal }
        }
        .presentationDetents([.height(220)])
    }

    // MARK: - Overlays
    private var processingBanner: some View {
        HStack(spacing: 12) {
            ProgressView().tint(.appGold)
            Text("正在辨識圖片中...").foregroundColor(.appGold)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.appSurface)
        .cornerRadius(16)
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(Color.appGold.opacity(0.3), lineWidth: 1))
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
    }

    private var successBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill").foregroundColor(.appSuccess)
            Text("成功擷取 \(extractedCount) 個項目").font(.subheadline).foregroundColor(.appTextPrimary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.appSuccess.opacity(0.1))
        .cornerRadius(16)
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                withAnimation { showSuccess = false }
            }
        }
    }

    private func errorBanner(_ msg: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundColor(.appDanger)
            Text(msg).font(.caption).foregroundColor(.appTextPrimary)
            Spacer()
            Button("✕") { processingError = nil }.foregroundColor(.appTextTertiary)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.appDanger.opacity(0.1))
        .cornerRadius(16)
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
    }

    // MARK: - Image Processing
    private func processImage(_ image: UIImage) {
        isProcessing = true
        processingError = nil

        Task {
            do {
                let rawText = try await OCRService.shared.recognizeText(from: image)
                guard !rawText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    await MainActor.run {
                        isProcessing = false
                        processingError = "未能從圖片中識別文字"
                    }
                    return
                }

                let result = try await GroqService.shared.parseReceipt(ocrText: rawText, apiKey: groqApiKey)
                let entry = result.budget_entry

                let receiptId = UUID().uuidString
                let df = DateFormatter(); df.dateFormat = "yyyy/MM/dd"
                let receiptDate = df.date(from: entry.timestamp) ?? Date()
                let receiptMs = receiptDate.millisecondsSince1970

                let newItems = entry.line_items.map { line -> ShoppingItem in
                    ShoppingItem(
                        name: line.name,
                        qty: max(line.qty, 1),
                        price: Int((line.total_price ?? 0) / Double(max(line.qty, 1))),
                        isChecked: true,
                        createdAt: receiptMs,
                        purchasedAt: receiptMs,
                        storeName: entry.store,
                        location: normalizeCategory(line.cat, line.name),
                        receiptId: receiptId
                    )
                }

                await MainActor.run {
                    shoppingItems.append(contentsOf: newItems)
                    extractedCount = newItems.count
                    isProcessing = false
                    withAnimation { showSuccess = true }
                }
            } catch {
                await MainActor.run {
                    isProcessing = false
                    processingError = "識別錯誤: \(error.localizedDescription)"
                }
            }
        }
    }
}

// MARK: - Supporting Views

struct DonutChartView: View {
    let items: [ShoppingItem]

    private var total: Double {
        Double(items.filter { $0.isChecked }.reduce(0) { $0 + $1.price * $1.qty })
    }

    @State private var animated = false

    var body: some View {
        ZStack {
            if total == 0 {
                Circle().stroke(Color.appBorder, lineWidth: 14)
            } else {
                ForEach(segments, id: \.catId) { seg in
                    Circle()
                        .trim(from: seg.start * (animated ? 1 : 0), to: seg.end * (animated ? 1 : 0))
                        .stroke(seg.color, style: StrokeStyle(lineWidth: 14, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                }
            }
        }
        .onAppear { withAnimation(.easeOut(duration: 0.9)) { animated = true } }
    }

    private struct Segment {
        let catId: String
        let color: Color
        let start: CGFloat
        let end: CGFloat
    }

    private var segments: [Segment] {
        var start: CGFloat = 0
        return dashboardCategories.compactMap { cat in
            let catTotal = Double(items.filter { $0.isChecked && $0.location == cat.catId }.reduce(0) { $0 + $1.price * $1.qty })
            guard catTotal > 0 else { return nil }
            let fraction = CGFloat(catTotal / total)
            let seg = Segment(catId: cat.catId, color: Color(hex: cat.colorHex), start: start, end: start + fraction)
            start += fraction
            return seg
        }
    }
}

struct CategoryBudgetRow: View {
    let category: CategoryInfo
    let spent: Int
    let left: Int
    let budgetPerCategory: Int
    let isExpanded: Bool
    let items: [ShoppingItem]
    let onToggle: () -> Void

    private var catColor: Color { Color(hex: category.colorHex) }
    private var progress: CGFloat {
        guard budgetPerCategory > 0 else { return 0 }
        return min(CGFloat(spent) / CGFloat(budgetPerCategory), 1.0)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(catColor.opacity(0.1))
                        .frame(width: 46, height: 46)
                    Image(systemName: category.icon)
                        .foregroundColor(catColor)
                        .font(.system(size: 20))
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(category.name)
                        .font(.headline)
                        .foregroundColor(.appTextPrimary)
                    Text(category.description)
                        .font(.caption)
                        .foregroundColor(.appTextTertiary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("$\(spent.formattedWithCommas)")
                        .font(.headline)
                        .foregroundColor(.appTextPrimary)
                    Text(left >= 0 ? "$\(left) 剩餘" : "$\(-left) 超支")
                        .font(.caption)
                        .foregroundColor(left >= 0 ? .appSuccess : .appDanger)
                }
                Image(systemName: "chevron.down")
                    .foregroundColor(.appTextTertiary)
                    .font(.caption)
                    .rotationEffect(.degrees(isExpanded ? 180 : 0))
                    .animation(.spring(response: 0.3), value: isExpanded)
            }

            if budgetPerCategory > 0 {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2).fill(Color.appBorder).frame(height: 4)
                        RoundedRectangle(cornerRadius: 2)
                            .fill(left >= 0 ? catColor : Color.appDanger)
                            .frame(width: geo.size.width * progress, height: 4)
                    }
                }
                .frame(height: 4)
                .padding(.top, 10)
            }

            if isExpanded {
                VStack(spacing: 0) {
                    Divider().padding(.vertical, 10)
                    if items.isEmpty {
                        Text("尚無此分類的消費")
                            .font(.caption)
                            .foregroundColor(.appTextTertiary)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(items) { item in ItemDetailRow(item: item) }
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .background(Color.appSurface)
        .cornerRadius(16)
        .onTapGesture { onToggle() }
    }
}

struct ExpandableCard<Content: View>: View {
    let icon: String
    let title: String
    let subtitle: String
    let trailingValue: String
    let trailingCaption: String?
    let isExpanded: Bool
    let accentColor: Color
    let onToggle: () -> Void
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(accentColor.opacity(0.1))
                        .frame(width: 46, height: 46)
                    Image(systemName: icon)
                        .foregroundColor(accentColor)
                        .font(.system(size: 20))
                }
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline).foregroundColor(.appTextPrimary)
                    Text(subtitle).font(.caption).foregroundColor(.appTextTertiary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(trailingValue).font(.headline).foregroundColor(.appTextPrimary)
                    if let cap = trailingCaption {
                        Text(cap).font(.caption).foregroundColor(.appTextTertiary)
                    }
                }
                Image(systemName: "chevron.down")
                    .foregroundColor(.appTextTertiary)
                    .font(.caption)
                    .rotationEffect(.degrees(isExpanded ? 180 : 0))
                    .animation(.spring(response: 0.3), value: isExpanded)
            }

            if isExpanded {
                VStack(spacing: 0) {
                    Divider().padding(.vertical, 10)
                    content()
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(16)
        .background(Color.appSurface)
        .cornerRadius(16)
        .onTapGesture { onToggle() }
    }
}

struct ItemDetailRow: View {
    let item: ShoppingItem
    private var catColor: Color {
        Color(hex: dashboardCategories.first(where: { $0.catId == item.location })?.colorHex ?? 0x94A3B8)
    }

    var body: some View {
        HStack(spacing: 10) {
            Circle().fill(catColor).frame(width: 6, height: 6)
            Text(item.name).font(.subheadline).foregroundColor(.appTextPrimary)
            Spacer()
            if item.qty > 1 {
                Text("×\(item.qty)").font(.caption).foregroundColor(.appTextTertiary)
            }
            Text("$\((item.price * item.qty).formattedWithCommas)")
                .font(.subheadline.bold())
                .foregroundColor(.appTextPrimary)
        }
        .padding(.vertical, 5)
    }
}

struct EmptyBudgetState: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "doc.text")
                .font(.system(size: 48))
                .foregroundColor(.appTextTertiary.opacity(0.5))
            Text(title).font(.subheadline).foregroundColor(.appTextTertiary)
            Text(subtitle).font(.caption).foregroundColor(.appTextTertiary.opacity(0.7))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 48)
    }
}

// MARK: - Camera Picker
struct CameraPickerView: UIViewControllerRepresentable {
    let onCapture: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onCapture: onCapture) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let onCapture: (UIImage?) -> Void
        init(onCapture: @escaping (UIImage?) -> Void) { self.onCapture = onCapture }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            let image = info[.originalImage] as? UIImage
            picker.dismiss(animated: true) { self.onCapture(image) }
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            picker.dismiss(animated: true) { self.onCapture(nil) }
        }
    }
}

// MARK: - Date extension for display
private extension Date {
    var monthDisplayString: String {
        let f = DateFormatter(); f.dateFormat = "MMMM"; f.locale = Locale(identifier: "en_US")
        return f.string(from: self)
    }
}
