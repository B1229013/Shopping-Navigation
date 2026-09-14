import SwiftUI

/// Goal-confirmation screen shown before starting a camera navigation session.
/// Mirrors Android's `TeammateHomeScreen` (minus the Google Places "附近商店" lookup,
/// which isn't ported — this project has no Places/Maps SDK integration).
struct NavigationEntryView: View {
    let shoppingItems: [ShoppingItem]

    @Environment(\.dismiss) private var dismiss
    @State private var isCreatingSession = false
    @State private var backendError: String?
    @State private var session: (sessionId: String, goal: String, guidance: String)?

    // Place selection
    @State private var availablePlaces: [PlaceInfo] = []
    @State private var selectedPlace: PlaceInfo?
    @State private var isExploreMode = false   // 「新地圖」— no pre-built reference map
    @State private var isLoadingPlaces = false
    @State private var placesError: String?

    /// Whether a valid map choice has been made (existing place or explore mode)
    private var hasMapSelection: Bool {
        selectedPlace != nil || isExploreMode
    }

    private var mapSelectionLabel: String {
        if isExploreMode { return "新地圖（即時探索）" }
        if let place = selectedPlace { return place.name }
        return "請選擇地圖"
    }

    private var pendingItems: [ShoppingItem] {
        shoppingItems.filter { !$0.isChecked }
    }

    private var goalText: String {
        pendingItems.map { "\($0.name) x\($0.qty)" }.joined(separator: ", ")
    }

    var body: some View {
        Group {
            if let session {
                NavigationCameraView(
                    sessionId: session.sessionId,
                    goal: session.goal,
                    initialGuidance: session.guidance,
                    onExit: { dismiss() }
                )
            } else {
                confirmContent
            }
        }
    }

    private var confirmContent: some View {
        NavigationView {
            VStack(alignment: .leading, spacing: 16) {
                // ── Place selector ──────────────────────────────
                VStack(alignment: .leading, spacing: 6) {
                    Text("選擇地圖")
                        .font(.subheadline.weight(.semibold))
                        .foregroundColor(.appTextPrimary)

                    if isLoadingPlaces {
                        HStack(spacing: 8) {
                            ProgressView()
                            Text("載入地圖列表...")
                                .font(.subheadline)
                                .foregroundColor(.appTextSecondary)
                        }
                        .padding(.vertical, 8)
                    } else if let placesError {
                        Text(placesError)
                            .font(.caption)
                            .foregroundColor(.red)
                    } else if availablePlaces.isEmpty {
                        Text("無可用地圖")
                            .font(.subheadline)
                            .foregroundColor(.appTextTertiary)
                    } else {
                        Menu {
                            ForEach(availablePlaces) { place in
                                Button {
                                    selectedPlace = place
                                    isExploreMode = false
                                } label: {
                                    HStack {
                                        Text(place.name)
                                        if place == selectedPlace && !isExploreMode {
                                            Image(systemName: "checkmark")
                                        }
                                    }
                                }
                            }

                            Divider()

                            Button {
                                selectedPlace = nil
                                isExploreMode = true
                            } label: {
                                HStack {
                                    Text("新地圖（即時探索）")
                                    if isExploreMode {
                                        Image(systemName: "checkmark")
                                    }
                                }
                            }
                        } label: {
                            HStack {
                                Image(systemName: isExploreMode ? "location.magnifyingglass" : "map")
                                    .foregroundColor(.appAccent)
                                Text(mapSelectionLabel)
                                    .foregroundColor(hasMapSelection ? .appTextPrimary : .appTextTertiary)
                                Spacer()
                                Image(systemName: "chevron.up.chevron.down")
                                    .font(.caption)
                                    .foregroundColor(.appTextSecondary)
                            }
                            .padding(12)
                            .background(Color.appSurface)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.appAccent.opacity(hasMapSelection ? 0.6 : 0.2), lineWidth: 1)
                            )
                        }

                        if isExploreMode {
                            Text("將以即時拍照分析的方式尋找物品，不使用預建地圖")
                                .font(.caption2)
                                .foregroundColor(.appTextTertiary)
                        }
                    }
                }

                // ── Shopping items ────────────────────────────────
                if pendingItems.isEmpty {
                    Spacer()
                    Text("購物清單是空的，請先回主頁新增商品")
                        .font(.subheadline)
                        .foregroundColor(.appTextTertiary)
                        .frame(maxWidth: .infinity, alignment: .center)
                    Spacer()
                } else {
                    Text("以下是您清單中的待購物品：")
                        .font(.subheadline)
                        .foregroundColor(.appTextSecondary)

                    List(pendingItems) { item in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(item.name)
                                    .font(.body.weight(.medium))
                                    .foregroundColor(.appTextPrimary)
                                if let storeName = item.storeName, !storeName.isEmpty {
                                    Text("預定地點: \(storeName)")
                                        .font(.caption)
                                        .foregroundColor(.appAccent)
                                }
                            }
                            Spacer()
                            Text("x\(item.qty)")
                                .font(.subheadline)
                                .foregroundColor(.appTextSecondary)
                        }
                        .listRowBackground(Color.appSurface)
                    }
                    .listStyle(.plain)
                    .scrollContentBackground(.hidden)
                }

                if let backendError {
                    Text(backendError)
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }

                Button(action: startNavigation) {
                    HStack {
                        if isCreatingSession {
                            ProgressView().tint(.white)
                        }
                        Text(isCreatingSession ? "建立導航中..." : "開始導航")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 54)
                    .background(Color.appAccent)
                    .foregroundColor(.white)
                    .cornerRadius(14)
                }
                .disabled(pendingItems.isEmpty || isCreatingSession || !hasMapSelection)
                .opacity(pendingItems.isEmpty || isCreatingSession || !hasMapSelection ? 0.6 : 1)
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .background(Color.appBg)
            .navigationTitle("確認購買商品")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Color.appBg, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("返回") { dismiss() }
                        .foregroundColor(.appAccent)
                }
            }
            .onAppear { loadPlaces() }
        }
    }

    private func loadPlaces() {
        isLoadingPlaces = true
        placesError = nil
        Task {
            do {
                let places = try await NavigationAPI.shared.getPlaces()
                await MainActor.run {
                    availablePlaces = places
                    // Auto-select if only one place
                    if places.count == 1 {
                        selectedPlace = places.first
                    }
                    isLoadingPlaces = false
                }
            } catch {
                await MainActor.run {
                    placesError = "無法載入地圖列表: \(error.localizedDescription)"
                    isLoadingPlaces = false
                }
            }
        }
    }

    private func startNavigation() {
        guard !pendingItems.isEmpty, hasMapSelection else { return }
        isCreatingSession = true
        backendError = nil

        let goal = goalText
        // Explore mode sends empty string → backend skips reference map
        let placeName: String? = isExploreMode ? "" : selectedPlace?.name
        Task {
            do {
                let response = try await NavigationAPI.shared.startSession(goal: goal, place: placeName)
                await MainActor.run {
                    session = (response.sessionId, goal, response.guidance)
                    isCreatingSession = false
                }
            } catch {
                await MainActor.run {
                    backendError = "無法連線後端伺服器: \(error.localizedDescription)"
                    isCreatingSession = false
                }
            }
        }
    }
}
