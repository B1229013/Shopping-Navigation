import SwiftUI

struct HistoryView: View {
    let shoppingItems: [ShoppingItem]

    @State private var selectedMonth: String = ""

    private var purchasedItems: [ShoppingItem] { shoppingItems.filter { $0.isChecked } }

    private var monthGroups: [(key: String, value: [ShoppingItem])] {
        let grouped = Dictionary(grouping: purchasedItems) { item in
            Date.from(milliseconds: item.createdAt).monthKey
        }
        return grouped.sorted { $0.key > $1.key }
    }

    private var availableMonths: [String] {
        monthGroups.map { $0.key }
    }

    private var displayItems: [ShoppingItem] {
        if selectedMonth.isEmpty { return purchasedItems }
        return purchasedItems.filter { Date.from(milliseconds: $0.createdAt).monthKey == selectedMonth }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("紀錄")
                    .font(.largeTitle.bold())
                    .foregroundColor(.appTextPrimary)
                Spacer()
                Text("共 \(purchasedItems.count) 筆")
                    .font(.subheadline)
                    .foregroundColor(.appTextSecondary)
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)
            .padding(.bottom, 12)

            // Month picker
            if !availableMonths.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        monthChip(label: "全部", key: "")
                        ForEach(availableMonths, id: \.self) { month in
                            monthChip(label: monthDisplay(month), key: month)
                        }
                    }
                    .padding(.horizontal, 20)
                }
                .padding(.bottom, 12)
            }

            // Summary row
            if !displayItems.isEmpty {
                HStack(spacing: 12) {
                    SummaryTile(
                        icon: "bag.fill",
                        label: "項目",
                        value: "\(displayItems.count)",
                        color: .appAccent
                    )
                    SummaryTile(
                        icon: "banknote.fill",
                        label: "總支出",
                        value: "$\(displayItems.reduce(0) { $0 + $1.price * $1.qty }.formattedWithCommas)",
                        color: .appGold
                    )
                    SummaryTile(
                        icon: "building.2.fill",
                        label: "門市",
                        value: "\(Set(displayItems.compactMap { $0.storeName }).count)",
                        color: .chartBlue
                    )
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 12)
            }

            ScrollView {
                LazyVStack(spacing: 10) {
                    if displayItems.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "clock.arrow.circlepath")
                                .font(.system(size: 48))
                                .foregroundColor(Color.appBorder)
                            Text("尚無購物紀錄")
                                .foregroundColor(.appTextTertiary)
                            Text("完成購物後，紀錄會顯示在這裡")
                                .font(.caption)
                                .foregroundColor(.appTextTertiary)
                        }
                        .padding(.top, 60)
                        .frame(maxWidth: .infinity)
                    } else {
                        ForEach(displayItems.sorted { $0.createdAt > $1.createdAt }) { item in
                            HistoryItemRow(item: item)
                                .padding(.horizontal, 20)
                        }
                    }
                }
                .padding(.bottom, 24)
            }
        }
        .adaptiveWidth(760)
        .background(Color.appBg)
    }

    private func monthChip(label: String, key: String) -> some View {
        let isSelected = selectedMonth == key
        return Text(label)
            .font(.subheadline)
            .foregroundColor(isSelected ? .white : .appTextSecondary)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(isSelected ? Color.appAccent : Color.appSurface)
            .cornerRadius(20)
            .onTapGesture { withAnimation { selectedMonth = key } }
    }

    private func monthDisplay(_ key: String) -> String {
        let parts = key.split(separator: "-")
        guard parts.count == 2,
              let month = Int(parts[1]) else { return key }
        return "\(parts[0])年\(month)月"
    }
}

struct SummaryTile: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon).font(.title3).foregroundColor(color)
            Text(value).font(.headline.bold()).foregroundColor(.appTextPrimary)
            Text(label).font(.caption).foregroundColor(.appTextSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(Color.appSurface)
        .cornerRadius(14)
    }
}

struct HistoryItemRow: View {
    let item: ShoppingItem
    private var catColor: Color {
        Color(hex: dashboardCategories.first(where: { $0.catId == item.location })?.colorHex ?? 0x94A3B8)
    }

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(catColor.opacity(0.1))
                    .frame(width: 44, height: 44)
                Image(systemName: catIcon)
                    .foregroundColor(catColor)
                    .font(.system(size: 18))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.subheadline.bold())
                    .foregroundColor(.appTextPrimary)
                HStack(spacing: 6) {
                    if let store = item.storeName {
                        Text(store).font(.caption).foregroundColor(.appTextTertiary)
                        Text("·").foregroundColor(.appTextTertiary).font(.caption)
                    }
                    Text(Date.from(milliseconds: item.createdAt).shortDateString)
                        .font(.caption)
                        .foregroundColor(.appTextTertiary)
                }
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("$\((item.price * item.qty).formattedWithCommas)")
                    .font(.subheadline.bold())
                    .foregroundColor(.appTextPrimary)
                if item.qty > 1 {
                    Text("×\(item.qty)").font(.caption).foregroundColor(.appTextTertiary)
                }
            }
        }
        .padding(14)
        .background(Color.appSurface)
        .cornerRadius(14)
    }

    private var catIcon: String {
        switch item.location {
        case "Food": return "fork.knife"
        case "Beverages": return "cup.and.saucer.fill"
        case "Groceries": return "bag.fill"
        default: return "ellipsis.circle.fill"
        }
    }
}
