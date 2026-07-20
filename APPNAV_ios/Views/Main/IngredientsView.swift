import SwiftUI

struct IngredientsView: View {
    @Binding var records: [DietRecord]

    @State private var showAddSheet = false
    @State private var selectedRecord: DietRecord? = nil
    @State private var searchText = ""

    private var filtered: [DietRecord] {
        searchText.isEmpty ? records : records.filter { $0.name.contains(searchText) || $0.foodCategory.contains(searchText) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("分析")
                    .font(.largeTitle.bold())
                    .foregroundColor(.appTextPrimary)
                Spacer()
                Button { showAddSheet = true } label: {
                    Image(systemName: "plus.circle.fill")
                        .font(.title2)
                        .foregroundColor(.appAccent)
                }
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)
            .padding(.bottom, 8)

            HStack {
                Image(systemName: "magnifyingglass").foregroundColor(.appTextSecondary)
                TextField("搜尋食品...", text: $searchText).foregroundColor(.appTextPrimary)
            }
            .padding(12)
            .background(Color.appSurface)
            .cornerRadius(12)
            .padding(.horizontal, 20)
            .padding(.bottom, 12)

            // Summary card
            if !records.isEmpty {
                HStack(spacing: 16) {
                    NutriStat(label: "總卡路里", value: "\(records.reduce(0) { $0 + $1.totalCalories })", unit: "kcal", color: Color(hex: 0xEF4444))
                    NutriStat(label: "蛋白質", value: String(format: "%.1f", records.reduce(0) { $0 + $1.protein }), unit: "g", color: Color(hex: 0x3B82F6))
                    NutriStat(label: "碳水", value: String(format: "%.1f", records.reduce(0) { $0 + $1.carbs }), unit: "g", color: Color(hex: 0xF59E0B))
                }
                .padding()
                .background(Color.appSurface)
                .cornerRadius(16)
                .padding(.horizontal, 20)
                .padding(.bottom, 12)
            }

            ScrollView {
                LazyVStack(spacing: 10) {
                    if filtered.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "leaf").font(.system(size: 48)).foregroundColor(Color.appBorder)
                            Text(searchText.isEmpty ? "尚無飲食紀錄" : "找不到相關食品")
                                .foregroundColor(.appTextTertiary)
                            if searchText.isEmpty {
                                Text("點擊 + 新增食品紀錄")
                                    .font(.caption).foregroundColor(.appTextTertiary)
                            }
                        }
                        .padding(.top, 48)
                        .frame(maxWidth: .infinity)
                    } else {
                        ForEach(filtered) { record in
                            DietRecordRow(record: record)
                                .padding(.horizontal, 20)
                                .onTapGesture { selectedRecord = record }
                                .swipeActions(edge: .trailing) {
                                    Button(role: .destructive) {
                                        withAnimation { records.removeAll { $0.id == record.id } }
                                    } label: {
                                        Label("刪除", systemImage: "trash")
                                    }
                                }
                        }
                    }
                }
                .padding(.bottom, 24)
            }
        }
        .adaptiveWidth(760)
        .background(Color.appBg)
        .sheet(isPresented: $showAddSheet) {
            AddDietRecordSheet { newRecord in
                records.insert(newRecord, at: 0)
            }
        }
        .sheet(item: $selectedRecord) { record in
            DietRecordDetailSheet(record: record)
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

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.appAccent.opacity(0.1))
                    .frame(width: 46, height: 46)
                Image(systemName: "leaf.fill")
                    .foregroundColor(.appAccent)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(record.name).font(.headline).foregroundColor(.appTextPrimary).lineLimit(1)
                Text(record.date).font(.caption).foregroundColor(.appTextTertiary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(record.totalCalories) kcal")
                    .font(.subheadline.bold())
                    .foregroundColor(Color(hex: 0xEF4444))
                Text(record.foodCategory)
                    .font(.caption2)
                    .foregroundColor(.appTextTertiary)
            }
        }
        .padding(14)
        .background(Color.appSurface)
        .cornerRadius(14)
    }
}

struct AddDietRecordSheet: View {
    let onAdd: (DietRecord) -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var date = Date()
    @State private var calories = ""
    @State private var protein = ""
    @State private var carbs = ""
    @State private var fat = ""
    @State private var category = "未分類"

    private let categories = ["未分類", "蔬菜", "水果", "肉類", "海鮮", "穀物", "乳製品", "飲料", "零食"]

    var body: some View {
        NavigationView {
            Form {
                Section("基本資訊") {
                    TextField("食品名稱", text: $name)
                    DatePicker("日期", selection: $date, displayedComponents: .date)
                    Picker("分類", selection: $category) {
                        ForEach(categories, id: \.self) { Text($0) }
                    }
                }
                Section("營養資訊") {
                    HStack {
                        Text("卡路里")
                        Spacer()
                        TextField("0", text: $calories).keyboardType(.numberPad).multilineTextAlignment(.trailing)
                        Text("kcal").foregroundColor(.appTextTertiary)
                    }
                    HStack {
                        Text("蛋白質")
                        Spacer()
                        TextField("0", text: $protein).keyboardType(.decimalPad).multilineTextAlignment(.trailing)
                        Text("g").foregroundColor(.appTextTertiary)
                    }
                    HStack {
                        Text("碳水")
                        Spacer()
                        TextField("0", text: $carbs).keyboardType(.decimalPad).multilineTextAlignment(.trailing)
                        Text("g").foregroundColor(.appTextTertiary)
                    }
                    HStack {
                        Text("脂肪")
                        Spacer()
                        TextField("0", text: $fat).keyboardType(.decimalPad).multilineTextAlignment(.trailing)
                        Text("g").foregroundColor(.appTextTertiary)
                    }
                }
            }
            .navigationTitle("新增飲食紀錄")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("新增") {
                        let df = DateFormatter(); df.dateFormat = "yyyy-MM-dd"
                        let record = DietRecord(
                            date: df.string(from: date),
                            name: name.isEmpty ? "未命名食品" : name,
                            totalCalories: Int(calories) ?? 0,
                            carbs: Double(carbs) ?? 0,
                            protein: Double(protein) ?? 0,
                            fat: Double(fat) ?? 0,
                            foodCategory: category
                        )
                        onAdd(record)
                        dismiss()
                    }
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("取消") { dismiss() }
                }
            }
        }
    }
}

struct DietRecordDetailSheet: View {
    let record: DietRecord
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            List {
                Section("基本資訊") {
                    LabeledContent("名稱", value: record.name)
                    LabeledContent("日期", value: record.date)
                    LabeledContent("分類", value: record.foodCategory)
                }
                Section("營養資訊") {
                    LabeledContent("卡路里", value: "\(record.totalCalories) kcal")
                    LabeledContent("蛋白質", value: String(format: "%.1f g", record.protein))
                    LabeledContent("碳水化合物", value: String(format: "%.1f g", record.carbs))
                    LabeledContent("脂肪", value: String(format: "%.1f g", record.fat))
                    if record.sugar > 0 { LabeledContent("糖", value: String(format: "%.1f g", record.sugar)) }
                    if record.sodium > 0 { LabeledContent("鈉", value: String(format: "%.1f mg", record.sodium)) }
                }
            }
            .navigationTitle(record.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("關閉") { dismiss() }
                }
            }
        }
    }
}
