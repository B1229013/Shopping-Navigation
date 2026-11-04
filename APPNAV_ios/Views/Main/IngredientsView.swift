import SwiftUI

/// Whole numbers without a decimal, otherwise one decimal place.
func calStr(_ v: Double) -> String {
    v == v.rounded() ? String(format: "%.0f", v) : String(format: "%.1f", v)
}

enum AnalysisRange: String, CaseIterable {
    case today = "今日", week = "本週", month = "本月"
}

struct IngredientsView: View {
    @Binding var records: [DietRecord]

    @State private var showSheet = false
    @State private var editingRecord: DietRecord? = nil
    @State private var searchText = ""
    @State private var selectedDate = Date()
    @State private var selectedExercise = "慢跑"
    @State private var range: AnalysisRange = .today

    private let exercises: [(name: String, rate: Double)] = [
        ("慢跑", 8.1), ("走路", 3.5), ("騎腳踏車", 5.5), ("健走", 5.5), ("爬樓梯", 8.5)
    ]
    private let df: DateFormatter = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"; return f
    }()

    private func inRange(_ dateStr: String) -> Bool {
        guard let d = df.date(from: dateStr) else { return false }
        let cal = Calendar.current
        switch range {
        case .today: return cal.isDate(d, inSameDayAs: selectedDate)
        case .week: return cal.isDate(d, equalTo: selectedDate, toGranularity: .weekOfYear)
        case .month: return cal.isDate(d, equalTo: selectedDate, toGranularity: .month)
        }
    }

    private var dayRecords: [DietRecord] {
        records.filter {
            inRange($0.date) &&
            (searchText.isEmpty || $0.name.contains(searchText) || $0.foodCategory.contains(searchText))
        }
    }
    private var dayCal: Double { dayRecords.reduce(0) { $0 + $1.totalCalories } }
    private var dayProtein: Double { dayRecords.reduce(0) { $0 + $1.protein } }
    private var dayCarbs: Double { dayRecords.reduce(0) { $0 + $1.carbs } }
    private var daySugar: Double { dayRecords.reduce(0) { $0 + $1.sugar } }
    private var exerciseMinutes: Int {
        let rate = exercises.first { $0.name == selectedExercise }?.rate ?? 1
        return dayCal > 0 ? Int(dayCal / rate) : 0
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            searchBar
            dateSelector
            summaryCard
            exerciseCard
            recordsList
        }
        .adaptiveWidth(760)
        .background(Color.appBg)
        .sheet(isPresented: $showSheet) {
            AddDietRecordSheet(existing: editingRecord, defaultDate: selectedDate) { rec in
                upsert(rec)
            }
        }
    }

    private func upsert(_ rec: DietRecord) {
        if let idx = records.firstIndex(where: { $0.id == rec.id }) {
            records[idx] = rec
        } else {
            records.insert(rec, at: 0)
        }
    }

    private var header: some View {
        HStack {
            Text("分析").font(.largeTitle.bold()).foregroundColor(.appTextPrimary)
            Spacer()
            Button {
                editingRecord = nil
                showSheet = true
            } label: {
                Image(systemName: "plus.circle.fill").font(.title2).foregroundColor(.appAccent)
            }
        }
        .padding(.horizontal, 20).padding(.top, 16).padding(.bottom, 8)
    }

    private var searchBar: some View {
        HStack {
            Image(systemName: "magnifyingglass").foregroundColor(.appTextSecondary)
            ZStack(alignment: .leading) {
                if searchText.isEmpty { Text("搜尋食品...").foregroundColor(.appTextSecondary) }
                TextField("", text: $searchText).foregroundColor(.appTextPrimary)
            }
        }
        .padding(12).background(Color.appSurface).cornerRadius(12)
        .padding(.horizontal, 20).padding(.bottom, 10)
    }

    private var dateSelector: some View {
        VStack(spacing: 8) {
            Picker("", selection: $range) {
                ForEach(AnalysisRange.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            HStack(spacing: 10) {
                Image(systemName: "calendar").foregroundColor(.appTextSecondary)
                DatePicker("", selection: $selectedDate, displayedComponents: .date).labelsHidden()
                Spacer()
                Button("今天") { selectedDate = Date() }
                    .font(.caption.bold()).foregroundColor(.appAccent)
            }
        }
        .padding(.horizontal, 20).padding(.bottom, 10)
    }

    private var summaryCard: some View {
        HStack(spacing: 14) {
            NutriStat(label: "熱量", value: calStr(dayCal), unit: "kcal", color: Color(hex: 0xEF4444))
            NutriStat(label: "蛋白質", value: calStr(dayProtein), unit: "g", color: Color(hex: 0x3B82F6))
            NutriStat(label: "碳水", value: calStr(dayCarbs), unit: "g", color: Color(hex: 0xF59E0B))
            NutriStat(label: "糖", value: calStr(daySugar), unit: "g", color: Color(hex: 0xDB2777))
        }
        .padding()
        .background(Color.appSurface)
        .cornerRadius(16)
        .padding(.horizontal, 20).padding(.bottom, 10)
    }

    private var exerciseCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("\(range.rawValue)運動消耗換算").font(.subheadline.bold()).foregroundColor(Color(hex: 0x16A34A))
            Picker("", selection: $selectedExercise) {
                ForEach(exercises, id: \.name) { Text($0.name).tag($0.name) }
            }
            .pickerStyle(.segmented)
            HStack {
                Spacer()
                Text("消耗\(range.rawValue) \(calStr(dayCal)) kcal 約需 \(selectedExercise) ")
                    .font(.subheadline).foregroundColor(Color(hex: 0x16A34A))
                Text("\(exerciseMinutes)")
                    .font(.title3.bold()).foregroundColor(Color(hex: 0x16A34A))
                Text(" 分鐘").font(.subheadline).foregroundColor(Color(hex: 0x16A34A))
                Spacer()
            }
        }
        .padding()
        .background(Color(hex: 0xF0FDF4))
        .cornerRadius(16)
        .padding(.horizontal, 20).padding(.bottom, 10)
    }

    private var recordsList: some View {
        ScrollView {
            LazyVStack(spacing: 10) {
                if dayRecords.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "leaf").font(.system(size: 44)).foregroundColor(Color.appBorder)
                        Text(searchText.isEmpty ? "這個區間還沒有飲食紀錄" : "找不到相關食品")
                            .foregroundColor(.appTextTertiary)
                    }
                    .padding(.top, 36).frame(maxWidth: .infinity)
                } else {
                    ForEach(dayRecords) { record in
                        DietRecordRow(
                            record: record,
                            onEdit: { editingRecord = record; showSheet = true },
                            onDelete: { withAnimation { records.removeAll { $0.id == record.id } } }
                        )
                        .padding(.horizontal, 20)
                    }
                }
            }
            .padding(.bottom, 24)
        }
    }
}

struct NutriStat: View {
    let label: String
    let value: String
    let unit: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            HStack(alignment: .lastTextBaseline, spacing: 2) {
                Text(value).font(.headline.bold()).foregroundColor(.appTextPrimary)
                Text(unit).font(.caption2).foregroundColor(.appTextTertiary)
            }
            Text(label).font(.caption2).foregroundColor(.appTextSecondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct DietRecordRow: View {
    let record: DietRecord
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12).fill(Color.appAccent.opacity(0.1)).frame(width: 46, height: 46)
                Image(systemName: "leaf.fill").foregroundColor(.appAccent)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(record.name).font(.headline).foregroundColor(.appTextPrimary).lineLimit(1)
                Text("\(calStr(record.totalCalories)) kcal · \(record.foodCategory) · \(record.date)")
                    .font(.caption).foregroundColor(.appTextTertiary)
            }
            Spacer()
            Button {
                onEdit()
            } label: {
                Image(systemName: "pencil").font(.caption).foregroundColor(.appTextSecondary)
                    .padding(6).background(Color.appBg).cornerRadius(8)
            }
            Button {
                onDelete()
            } label: {
                Image(systemName: "trash").font(.caption).foregroundColor(.appDanger)
                    .padding(6).background(Color.appDanger.opacity(0.1)).cornerRadius(8)
            }
        }
        .padding(14).background(Color.appSurface).cornerRadius(14)
        .contentShape(Rectangle())
        .onTapGesture { onEdit() }
    }
}

struct AddDietRecordSheet: View {
    let existing: DietRecord?
    let onSave: (DietRecord) -> Void
    @Environment(\.dismiss) private var dismiss
    @AppStorage("profileAllergies") private var profileAllergies = ""

    @State private var name: String
    @State private var ingredientText: String
    @State private var date: Date
    @State private var unitCalorie: String
    @State private var portion: String
    @State private var protein: String
    @State private var fat: String
    @State private var saturatedFat: String
    @State private var transFat: String
    @State private var carbs: String
    @State private var sugar: String
    @State private var sodium: String
    @State private var sugarAlcohol: String
    @State private var category: String
    @State private var showCamera = false
    @State private var isOCRProcessing = false

    private let keepId: String
    private let categories = ["全榖雜糧", "豆魚蛋肉", "乳品類", "蔬菜類", "水果類", "油脂類", "其他"]

    init(existing: DietRecord?, defaultDate: Date, onSave: @escaping (DietRecord) -> Void) {
        self.existing = existing
        self.onSave = onSave
        self.keepId = existing?.id ?? UUID().uuidString
        func s(_ v: Double) -> String { v == 0 ? "" : calStr(v) }
        func so(_ v: Double?) -> String { (v == nil || v == 0) ? "" : calStr(v!) }
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"
        _name = State(initialValue: existing?.name ?? "")
        _ingredientText = State(initialValue: existing?.ingredients ?? "")
        _date = State(initialValue: existing.flatMap { f.date(from: $0.date) } ?? defaultDate)
        _unitCalorie = State(initialValue: s(existing?.unitCalorie ?? 0))
        _portion = State(initialValue: existing.map { String($0.portion) } ?? "1.0")
        _protein = State(initialValue: s(existing?.protein ?? 0))
        _fat = State(initialValue: s(existing?.fat ?? 0))
        _saturatedFat = State(initialValue: so(existing?.saturatedFat))
        _transFat = State(initialValue: so(existing?.transFat))
        _carbs = State(initialValue: s(existing?.carbs ?? 0))
        _sugar = State(initialValue: s(existing?.sugar ?? 0))
        _sodium = State(initialValue: s(existing?.sodium ?? 0))
        _sugarAlcohol = State(initialValue: so(existing?.sugarAlcohol))
        _category = State(initialValue: existing?.foodCategory ?? "全榖雜糧")
    }

    private var totalCalories: Double {
        (Double(unitCalorie) ?? 0) * (Double(portion) ?? 0)
    }
    private var detectedAllergens: [String] {
        let list = profileAllergies
            .split(whereSeparator: { $0 == "," || $0 == "，" })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !ingredientText.isEmpty else { return [] }
        return list.filter { ingredientText.contains($0) }
    }

    var body: some View {
        NavigationView {
            Form {
                Section("基本資訊") {
                    TextField("食品名稱 *", text: $name)
                    DatePicker("日期", selection: $date, displayedComponents: .date)
                    Picker("分類", selection: $category) {
                        ForEach(categories, id: \.self) { Text($0) }
                    }
                }

                Section("成分表 / 營養標示") {
                    TextField("可拍照自動辨識，或手動輸入成分", text: $ingredientText, axis: .vertical)
                        .lineLimit(2...5)
                    Button {
                        showCamera = true
                    } label: {
                        HStack {
                            if isOCRProcessing { ProgressView() } else { Image(systemName: "camera.fill") }
                            Text(isOCRProcessing ? "辨識中..." : "拍照辨識")
                        }
                    }
                    .disabled(isOCRProcessing)
                    if !detectedAllergens.isEmpty {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill").foregroundColor(.appDanger)
                            Text("警告：含過敏原（\(detectedAllergens.joined(separator: "、"))）")
                                .font(.caption.bold()).foregroundColor(.appDanger)
                        }
                    }
                }

                Section("熱量與份量") {
                    nutriRow("單份熱量", $unitCalorie, "kcal")
                    nutriRow("份量", $portion, "份")
                    HStack {
                        Text("計算總熱量").foregroundColor(.appTextSecondary)
                        Spacer()
                        Text("\(calStr(totalCalories)) kcal").font(.headline.bold()).foregroundColor(Color(hex: 0xEA580C))
                    }
                }

                Section("營養標示（每份）") {
                    nutriRow("蛋白質", $protein, "g")
                    nutriRow("脂肪", $fat, "g")
                    nutriRow("飽和脂肪", $saturatedFat, "g")
                    nutriRow("反式脂肪", $transFat, "g")
                    nutriRow("碳水化合物", $carbs, "g")
                    nutriRow("糖", $sugar, "g")
                    nutriRow("鈉", $sodium, "mg")
                    nutriRow("糖醇", $sugarAlcohol, "g")
                }
            }
            .navigationTitle(existing == nil ? "新增飲食紀錄" : "編輯飲食紀錄")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("儲存") { save() }
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { dismiss() }
                }
            }
            .sheet(isPresented: $showCamera) {
                CameraPickerView { image in
                    guard let image else { return }
                    runOCR(image)
                }
            }
        }
    }

    private func nutriRow(_ label: String, _ value: Binding<String>, _ unit: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            TextField("0", text: value).keyboardType(.decimalPad).multilineTextAlignment(.trailing)
            Text(unit).foregroundColor(.appTextTertiary)
        }
    }

    private func runOCR(_ image: UIImage) {
        isOCRProcessing = true
        Task {
            let raw = (try? await OCRService.shared.recognizeText(from: image)) ?? ""
            var parsed: NutritionParse? = nil
            if !raw.isEmpty {
                parsed = try? await GroqService.shared.parseNutritionLabel(ocrText: raw, apiKey: AppConfig.groqApiKey)
            }
            await MainActor.run {
                if let p = parsed {
                    if var v = p.ingredients, !v.isEmpty {
                        if let r = v.range(of: "^.*?(成分|成份|原料)\\s*[：:]\\s*", options: .regularExpression) { v.removeSubrange(r) }
                        ingredientText = v.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    if let v = p.name, !v.isEmpty, name.isEmpty { name = v }
                    if let v = p.calorie { unitCalorie = calStr(v) }
                    if let v = p.protein { protein = calStr(v) }
                    if let v = p.fat { fat = calStr(v) }
                    if let v = p.saturatedFat { saturatedFat = calStr(v) }
                    if let v = p.transFat { transFat = calStr(v) }
                    if let v = p.carbs { carbs = calStr(v) }
                    if let v = p.sugar { sugar = calStr(v) }
                    if let v = p.sodium { sodium = calStr(v) }
                    if let v = p.sugarAlcohol { sugarAlcohol = calStr(v) }
                } else if !raw.isEmpty {
                    ingredientText = raw.replacingOccurrences(of: "\n", with: " ")
                }
                isOCRProcessing = false
            }
        }
    }

    private func save() {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd"
        func d(_ s: String) -> Double { Double(s) ?? 0 }
        func dopt(_ s: String) -> Double? { s.isEmpty ? nil : Double(s) }
        let record = DietRecord(
            id: keepId,
            date: f.string(from: date),
            name: name.isEmpty ? "未命名食品" : name,
            ingredients: ingredientText,
            unitCalorie: d(unitCalorie),
            portion: Double(portion) ?? 1,
            totalCalories: totalCalories,
            carbs: d(carbs),
            sugar: d(sugar),
            protein: d(protein),
            fat: d(fat),
            sodium: d(sodium),
            saturatedFat: dopt(saturatedFat),
            transFat: dopt(transFat),
            sugarAlcohol: dopt(sugarAlcohol),
            foodCategory: category
        )
        onSave(record)
        dismiss()
    }
}
