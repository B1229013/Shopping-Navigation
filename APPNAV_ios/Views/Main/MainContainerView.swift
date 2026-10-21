import SwiftUI
import Combine

struct MainContainerView: View {
    @StateObject private var store = AppStore()
    @State private var selectedTab = 0
    @State private var showSettings = false
    @State private var showNavigation = false

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView(
                shoppingItems: store.shoppingItems,
                onNavigateToList: { selectedTab = 1 },
                onNavigateToBudget: { selectedTab = 3 }
            )
            .tag(0)

            ShoppingListView(items: $store.shoppingItems)
                .tag(1)

            IngredientsView(records: $store.dietRecords)
                .tag(2)

            BudgetView(shoppingItems: $store.shoppingItems, budgetTotal: $store.budgetTotal)
                .tag(3)

            HistoryView(shoppingItems: store.shoppingItems)
                .tag(4)

            AIView(
                shoppingItems: store.shoppingItems,
                dietRecords: store.dietRecords,
                budgetTotal: Int(store.budgetTotal) ?? 0
            )
            .tag(5)
        }
        .tabViewStyle(.page(indexDisplayMode: .never))
        // safeAreaInset measures each bar's own height and reserves exactly that much
        // space above/below the TabView content — it automatically adapts to every
        // device's notch/Dynamic Island/home-indicator, so no magic-number padding
        // is needed anywhere else in the app.
        .safeAreaInset(edge: .top, spacing: 0) { topBar }
        .safeAreaInset(edge: .bottom, spacing: 0) { customTabBar }
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
        .fullScreenCover(isPresented: $showNavigation) {
            NavigationEntryView(shoppingItems: store.shoppingItems)
        }
        .onReceive(store.$shoppingItems) { _ in store.save() }
        .onReceive(store.$dietRecords) { _ in store.save() }
        .onReceive(store.$budgetTotal) { _ in store.save() }
    }

    // MARK: - Top Bar
    private var topBar: some View {
        HStack {
            Button { showSettings = true } label: {
                Image(systemName: "gearshape")
                    .foregroundColor(.appTextSecondary)
                    .font(.title3)
                    .padding(10)
            }
            Spacer()
            Text(tabTitle(selectedTab))
                .font(.headline)
                .foregroundColor(.appTextPrimary)
            Spacer()
            Button { showNavigation = true } label: {
                Image(systemName: "mappin.and.ellipse")
                    .foregroundColor(.appTextSecondary)
                    .font(.title3)
                    .padding(10)
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .padding(.bottom, 8)
        .background(
            // The background alone extends behind the status bar/notch; the
            // interactive content stays clear of it on every device.
            Color.appBg.opacity(0.95).ignoresSafeArea(edges: .top)
        )
    }

    // MARK: - Custom Tab Bar
    private var customTabBar: some View {
        HStack(spacing: 0) {
            ForEach(Array(tabs.enumerated()), id: \.offset) { index, tab in
                tabItem(icon: tab.icon, label: tab.label, index: index)
            }
        }
        .padding(.vertical, 6)
        .background(
            Color.appSurface
                .overlay(Divider(), alignment: .top)
                .ignoresSafeArea(edges: .bottom)
        )
    }

    private let tabs: [(icon: String, label: String)] = [
        ("house.fill", "首頁"),
        ("list.bullet", "清單"),
        ("chart.bar.fill", "分析"),
        ("banknote.fill", "預算"),
        ("clock.fill", "紀錄"),
        ("brain.head.profile", "助理"),
    ]

    private func tabItem(icon: String, label: String, index: Int) -> some View {
        let isSelected = selectedTab == index
        return Button {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                selectedTab = index
            }
        } label: {
            VStack(spacing: 3) {
                ZStack {
                    if isSelected {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.appAccent.opacity(0.12))
                            .frame(width: 44, height: 30)
                    }
                    Image(systemName: icon)
                        .font(.system(size: 18))
                        .foregroundColor(isSelected ? .appAccent : .appTextTertiary)
                        .scaleEffect(isSelected ? 1.08 : 1.0)
                }
                Text(label)
                    .font(.system(size: 10, weight: isSelected ? .semibold : .regular))
                    .foregroundColor(isSelected ? .appAccent : .appTextTertiary)

                if isSelected {
                    Circle()
                        .fill(Color.appAccent)
                        .frame(width: 4, height: 4)
                        .transition(.scale.combined(with: .opacity))
                } else {
                    Color.clear.frame(width: 4, height: 4)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func tabTitle(_ index: Int) -> String {
        tabs[safe: index]?.label ?? ""
    }
}

// MARK: - App State Store

@MainActor
final class AppStore: ObservableObject {
    @Published var shoppingItems: [ShoppingItem] = []
    @Published var dietRecords: [DietRecord] = []
    @Published var budgetTotal: String = "0"

    private let persistence = PersistenceService.shared

    init() { load() }

    func load() {
        shoppingItems = persistence.load([ShoppingItem].self, from: "shopping_list.json") ?? []
        dietRecords = persistence.load([DietRecord].self, from: "diet_records.json") ?? []
        budgetTotal = persistence.loadString(from: "monthly_budget.txt") ?? "0"
    }

    func save() {
        persistence.save(shoppingItems, to: "shopping_list.json")
        persistence.save(dietRecords, to: "diet_records.json")
        persistence.saveString(budgetTotal, to: "monthly_budget.txt")
    }
}

// MARK: - Safe Array Extension
extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
