import SwiftUI

struct ShoppingListView: View {
    @Binding var items: [ShoppingItem]

    @State private var newName = ""
    @State private var newQty = "1"
    @State private var newPrice = ""
    @State private var editingId: String? = nil
    @State private var editName = ""
    @State private var editQty = ""
    @State private var editPrice = ""
    @State private var showAddSection = false

    private let accent = Color.appAccent
    private let surface = Color.appSurface

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                // Add item panel
                addPanel
                    .padding(.horizontal, 20)

                // Item list
                if items.isEmpty {
                    emptyState
                } else {
                    ForEach(items) { item in
                        if editingId == item.id {
                            editRow(item: item)
                                .padding(.horizontal, 20)
                        } else {
                            itemRow(item: item)
                                .padding(.horizontal, 20)
                        }
                    }
                }
            }
            .padding(.vertical, 8)
            .padding(.bottom, 24)
            .adaptiveWidth(700)
        }
        .background(Color.appBg)
    }

    // MARK: - Add Panel
    private var addPanel: some View {
        VStack(spacing: 10) {
            HStack {
                TextField("商品名稱", text: $newName)
                    .foregroundColor(.appTextPrimary)
                    .padding(10)
                    .background(Color.appBg)
                    .cornerRadius(10)
                    .frame(maxWidth: .infinity)
                TextField("價格", text: $newPrice)
                    .keyboardType(.numberPad)
                    .foregroundColor(.appTextPrimary)
                    .padding(10)
                    .background(Color.appBg)
                    .cornerRadius(10)
                    .frame(width: 80)
                TextField("數量", text: $newQty)
                    .keyboardType(.numberPad)
                    .foregroundColor(.appTextPrimary)
                    .padding(10)
                    .background(Color.appBg)
                    .cornerRadius(10)
                    .frame(width: 60)
            }
            Button(action: addItem) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("新增商品")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(accent)
                .foregroundColor(.white)
                .cornerRadius(14)
            }
            .disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty)
        }
        .padding(16)
        .background(surface)
        .cornerRadius(18)
    }

    // MARK: - Item Row
    private func itemRow(item: ShoppingItem) -> some View {
        HStack(spacing: 12) {
            Button {
                toggleCheck(item)
            } label: {
                Image(systemName: item.isChecked ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .foregroundColor(item.isChecked ? accent : .appTextTertiary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(item.name)
                    .font(.body)
                    .strikethrough(item.isChecked)
                    .foregroundColor(item.isChecked ? .appTextTertiary : .appTextPrimary)
                HStack(spacing: 6) {
                    if let loc = item.location {
                        Text(loc)
                            .font(.caption2)
                            .foregroundColor(.appTextTertiary)
                    }
                    Text("×\(item.qty)")
                        .font(.caption)
                        .foregroundColor(.appTextSecondary)
                    if item.price > 0 {
                        Text("$\(item.price * item.qty)")
                            .font(.caption.bold())
                            .foregroundColor(accent)
                    }
                }
            }

            Spacer()

            HStack(spacing: 8) {
                Button {
                    startEdit(item)
                } label: {
                    Image(systemName: "pencil")
                        .font(.caption)
                        .foregroundColor(.appTextSecondary)
                        .padding(6)
                        .background(Color.appBg)
                        .cornerRadius(8)
                }
                Button {
                    deleteItem(item)
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundColor(.appDanger)
                        .padding(6)
                        .background(Color.appDanger.opacity(0.1))
                        .cornerRadius(8)
                }
            }
        }
        .padding(14)
        .background(surface)
        .cornerRadius(14)
    }

    // MARK: - Edit Row
    private func editRow(item: ShoppingItem) -> some View {
        VStack(spacing: 10) {
            HStack(spacing: 8) {
                TextField("名稱", text: $editName)
                    .foregroundColor(.appTextPrimary)
                    .padding(10)
                    .background(Color.appBg)
                    .cornerRadius(10)
                TextField("價格", text: $editPrice)
                    .keyboardType(.numberPad)
                    .foregroundColor(.appTextPrimary)
                    .padding(10)
                    .background(Color.appBg)
                    .cornerRadius(10)
                    .frame(width: 80)
                TextField("數量", text: $editQty)
                    .keyboardType(.numberPad)
                    .foregroundColor(.appTextPrimary)
                    .padding(10)
                    .background(Color.appBg)
                    .cornerRadius(10)
                    .frame(width: 60)
            }
            HStack(spacing: 8) {
                Button("取消") { editingId = nil }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(Color.appBg)
                    .foregroundColor(.appTextSecondary)
                    .cornerRadius(10)
                Button("儲存") { saveEdit(id: item.id) }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(accent)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding(14)
        .background(surface)
        .cornerRadius(14)
        .overlay(RoundedRectangle(cornerRadius: 14).stroke(accent.opacity(0.4), lineWidth: 1.5))
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "cart")
                .font(.system(size: 48))
                .foregroundColor(Color.appBorder)
            Text("購物清單是空的")
                .foregroundColor(.appTextTertiary)
            Text("在上方輸入商品來新增")
                .font(.caption)
                .foregroundColor(.appTextTertiary)
        }
        .padding(.top, 60)
        .frame(maxWidth: .infinity)
    }

    // MARK: - Actions
    private func addItem() {
        let trimmed = newName.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        let item = ShoppingItem(
            name: trimmed,
            qty: Int(newQty) ?? 1,
            price: Int(newPrice) ?? 0,
            location: guessCategory(from: trimmed)
        )
        withAnimation { items.insert(item, at: 0) }
        newName = ""; newQty = "1"; newPrice = ""
    }

    private func toggleCheck(_ item: ShoppingItem) {
        if let idx = items.firstIndex(where: { $0.id == item.id }) {
            withAnimation {
                items[idx].isChecked.toggle()
                items[idx].purchasedAt = items[idx].isChecked ? Date().millisecondsSince1970 : nil
            }
        }
    }

    private func deleteItem(_ item: ShoppingItem) {
        withAnimation { items.removeAll { $0.id == item.id } }
    }

    private func startEdit(_ item: ShoppingItem) {
        editingId = item.id
        editName = item.name
        editQty = "\(item.qty)"
        editPrice = "\(item.price)"
    }

    private func saveEdit(id: String) {
        if let idx = items.firstIndex(where: { $0.id == id }) {
            items[idx].name = editName
            items[idx].qty = Int(editQty) ?? 1
            items[idx].price = Int(editPrice) ?? 0
            items[idx].location = guessCategory(from: editName)
        }
        editingId = nil
    }
}
