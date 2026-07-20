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
                .disabled(pendingItems.isEmpty || isCreatingSession)
                .opacity(pendingItems.isEmpty || isCreatingSession ? 0.6 : 1)
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
        }
    }

    private func startNavigation() {
        guard !pendingItems.isEmpty else { return }
        isCreatingSession = true
        backendError = nil

        let goal = goalText
        Task {
            do {
                let response = try await NavigationAPI.shared.startSession(goal: goal)
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
