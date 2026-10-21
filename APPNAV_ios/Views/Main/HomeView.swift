import SwiftUI

struct HomeView: View {
    let shoppingItems: [ShoppingItem]
    let onNavigateToList: () -> Void
    let onNavigateToBudget: () -> Void

    @State private var selectedCategory: String? = nil
    @State private var searchText = ""

    private let groceryCategories: [(name: String, icon: String, color: Color, bg: Color)] = [
        ("蔬菜", "leaf.fill", Color(hex: 0x059669), Color(hex: 0xD1FAE5)),
        ("水果", "apple.logo", Color(hex: 0xD97706), Color(hex: 0xFEF3C7)),
        ("零食", "birthday.cake.fill", Color(hex: 0xDB2777), Color(hex: 0xFCE7F3)),
        ("蛋奶", "cup.and.saucer.fill", Color(hex: 0x92400E), Color(hex: 0xFDE68A)),
        ("飲品", "drop.fill", Color(hex: 0x2563EB), Color(hex: 0xDBEAFE)),
        ("調味", "flame.fill", Color(hex: 0x7C3AED), Color(hex: 0xEDE9FE)),
    ]

    private var pendingItems: [ShoppingItem] { shoppingItems.filter { !$0.isChecked } }
    private var recentPurchased: [ShoppingItem] { Array(shoppingItems.filter { $0.isChecked }.suffix(6).reversed()) }
    private var totalSpent: Int { shoppingItems.filter { $0.isChecked }.reduce(0) { $0 + $1.price * $1.qty } }

    private var filteredItems: [ShoppingItem] {
        var items = pendingItems
        if !searchText.isEmpty {
            items = items.filter { $0.name.contains(searchText) }
        }
        if let cat = selectedCategory {
            items = items.filter { matchesHomeCategory($0.name, cat) }
        }
        return items
    }

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0, pinnedViews: []) {
                headerSection
                summaryBanner
                searchBar
                categoriesSection
                productGridSection
                if !recentPurchased.isEmpty { recentSection }
                statsSection
                    .padding(.bottom, 24)
            }
            .adaptiveWidth(760)
        }
        .background(Color.appBg)
    }

    // MARK: - Header
    private var headerSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(Date().greeting), \(AuthService.shared.displayName)")
                    .font(.title2.bold())
                    .foregroundColor(.appTextPrimary)
                Text("今天想買什麼？")
                    .font(.subheadline)
                    .foregroundColor(.appTextSecondary)
            }
            Spacer()
            ZStack {
                Circle()
                    .fill(Color.appAccent.opacity(0.15))
                    .frame(width: 44, height: 44)
                Text(AuthService.shared.displayName.prefix(1).uppercased())
                    .font(.headline.bold())
                    .foregroundColor(.appAccent)
            }
        }
        .padding(.horizontal, 24)
        .padding(.top, 16)
    }

    // MARK: - Summary Banner
    private var summaryBanner: some View {
        HStack {
            VStack(alignment: .leading, spacing: 6) {
                Text("購物清單總覽")
                    .font(.headline.bold())
                    .foregroundColor(.white)
                Text("\(pendingItems.count) 項待購  ·  已花費 $\(totalSpent)")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.85))
                Button(action: onNavigateToList) {
                    Text("查看清單")
                        .font(.caption.bold())
                        .foregroundColor(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.white.opacity(0.2))
                        .cornerRadius(10)
                }
                .padding(.top, 8)
            }
            Spacer()
            Image(systemName: "cart.fill")
                .resizable()
                .scaledToFit()
                .frame(width: 56)
                .foregroundColor(.white.opacity(0.3))
        }
        .padding(20)
        .background(
            LinearGradient(colors: [Color(hex: 0x059669), Color(hex: 0x5BA85A)], startPoint: .leading, endPoint: .trailing)
        )
        .cornerRadius(20)
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    // MARK: - Search
    private var searchBar: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.appTextSecondary)
            TextField("搜尋商品...", text: $searchText)
                .foregroundColor(.appTextPrimary)
        }
        .padding(14)
        .background(Color.appSurface)
        .cornerRadius(14)
        .padding(.horizontal, 24)
    }

    // MARK: - Categories
    private var categoriesSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("分類")
                    .font(.headline.bold())
                    .foregroundColor(.appTextPrimary)
                Spacer()
                Button("全部") { selectedCategory = nil }
                    .font(.subheadline)
                    .foregroundColor(.appAccent)
            }
            .padding(.horizontal, 24)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 16) {
                    ForEach(groceryCategories, id: \.name) { cat in
                        let isSelected = selectedCategory == cat.name
                        VStack(spacing: 6) {
                            ZStack {
                                Circle()
                                    .fill(isSelected ? cat.color.opacity(0.15) : cat.bg)
                                    .frame(width: 56, height: 56)
                                    .overlay(
                                        Circle()
                                            .stroke(isSelected ? cat.color : Color.clear, lineWidth: 2)
                                    )
                                Image(systemName: cat.icon)
                                    .font(.title3)
                                    .foregroundColor(cat.color)
                            }
                            Text(cat.name)
                                .font(isSelected ? .caption.bold() : .caption)
                                .foregroundColor(isSelected ? cat.color : .appTextSecondary)
                        }
                        .onTapGesture {
                            withAnimation(.spring(response: 0.3)) {
                                selectedCategory = selectedCategory == cat.name ? nil : cat.name
                            }
                        }
                    }
                }
                .padding(.horizontal, 24)
            }
        }
        .padding(.top, 20)
    }

    // MARK: - Product Grid
    private var productGridSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(selectedCategory != nil ? "\(selectedCategory!) 商品" : "待購商品")
                    .font(.headline.bold())
                    .foregroundColor(.appTextPrimary)
                Spacer()
                Text("\(filteredItems.count) 項")
                    .font(.caption)
                    .foregroundColor(.appTextSecondary)
            }
            .padding(.horizontal, 24)

            if filteredItems.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "cart")
                        .font(.system(size: 48))
                        .foregroundColor(Color.appBorder)
                    Text(selectedCategory != nil ? "此分類目前沒有商品" : "購物清單是空的")
                        .font(.subheadline)
                        .foregroundColor(.appTextTertiary)
                    Text("到清單頁面新增商品吧！")
                        .font(.caption)
                        .foregroundColor(.appTextTertiary)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 48)
            } else {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150, maximum: 220), spacing: 14)], spacing: 14) {
                    ForEach(filteredItems) { item in
                        ProductCard(item: item)
                    }
                }
                .padding(.horizontal, 24)
            }
        }
        .padding(.top, 22)
    }

    // MARK: - Recent Purchased
    private var recentSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("最近已購")
                    .font(.headline.bold())
                    .foregroundColor(.appTextPrimary)
                Spacer()
                Button("查看全部", action: onNavigateToList)
                    .font(.caption)
                    .foregroundColor(.appAccent)
            }
            .padding(.horizontal, 24)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(recentPurchased) { item in
                        RecentCard(item: item)
                    }
                }
                .padding(.horizontal, 24)
            }
        }
        .padding(.top, 8)
    }

    // MARK: - Stats
    private var statsSection: some View {
        HStack(spacing: 12) {
            StatCard(label: "待購", value: "\(pendingItems.count)", icon: "cart.fill", color: .appAccent, bg: Color(hex: 0xD1FAE5))
            StatCard(label: "已花費", value: "$\(totalSpent)", icon: "banknote.fill", color: .appGold, bg: Color(hex: 0xFEF3C7), action: onNavigateToBudget)
        }
        .padding(.horizontal, 24)
        .padding(.top, 16)
    }
}

// MARK: - Sub-components

private struct ProductCard: View {
    let item: ShoppingItem

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topTrailing) {
                Color(hex: 0xD1FAE5)
                    .frame(height: 130)
                Image(systemName: categoryIcon(item.location))
                    .font(.system(size: 44))
                    .foregroundColor(Color.appAccent.opacity(0.4))

                Image(systemName: "heart")
                    .font(.system(size: 14))
                    .foregroundColor(Color(hex: 0xEF4444))
                    .padding(8)
                    .background(Color.white.opacity(0.9))
                    .clipShape(Circle())
                    .padding(8)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(item.name)
                    .font(.subheadline.bold())
                    .foregroundColor(.appTextPrimary)
                    .lineLimit(1)
                if let loc = item.location {
                    Text(loc)
                        .font(.caption)
                        .foregroundColor(.appTextTertiary)
                }
                HStack {
                    Text("$\(item.price)")
                        .font(.headline.bold())
                        .foregroundColor(.appAccent)
                    Spacer()
                    Text("x\(item.qty)")
                        .font(.caption.bold())
                        .foregroundColor(.appAccent)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color(hex: 0xD1FAE5))
                        .cornerRadius(8)
                }
            }
            .padding(12)
        }
        .background(Color.appSurface)
        .cornerRadius(16)
    }

    private func categoryIcon(_ location: String?) -> String {
        switch location {
        case "Food": return "fork.knife"
        case "Beverages": return "cup.and.saucer.fill"
        case "Groceries": return "bag.fill"
        default: return "cart.fill"
        }
    }
}

private struct RecentCard: View {
    let item: ShoppingItem

    var body: some View {
        HStack(spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(hex: 0xD1FAE5))
                    .frame(width: 48, height: 48)
                Image(systemName: "cart.fill")
                    .foregroundColor(.appAccent.opacity(0.5))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.caption.bold())
                    .foregroundColor(.appTextPrimary)
                    .lineLimit(1)
                Text("$\(item.price * item.qty)")
                    .font(.caption2)
                    .foregroundColor(.appAccent)
            }
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.appAccent.opacity(0.5))
                .font(.system(size: 16))
        }
        .padding(10)
        .frame(width: 200)
        .background(Color.appSurface)
        .cornerRadius(14)
    }
}

private struct StatCard: View {
    let label: String
    let value: String
    let icon: String
    let color: Color
    let bg: Color
    var action: (() -> Void)? = nil

    var body: some View {
        HStack(spacing: 12) {
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(bg)
                    .frame(width: 40, height: 40)
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.system(size: 16))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.headline.bold())
                    .foregroundColor(.appTextPrimary)
                Text(label)
                    .font(.caption)
                    .foregroundColor(.appTextSecondary)
            }
            Spacer()
        }
        .padding(16)
        .background(Color.appSurface)
        .cornerRadius(16)
        .onTapGesture { action?() }
        .frame(maxWidth: .infinity)
    }
}
