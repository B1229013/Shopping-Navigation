import Foundation

struct ShoppingItem: Codable, Identifiable, Equatable {
    var id: String = UUID().uuidString
    var name: String
    var qty: Int = 1
    var price: Int = 0
    var isChecked: Bool = false
    var createdAt: Double = Date().timeIntervalSince1970 * 1000
    var purchasedAt: Double? = nil
    var dueDate: Double? = nil
    var storeName: String? = nil
    var location: String? = nil
    var receiptId: String? = nil
    var source: String? = nil   // nil/"list" = shopping list, "receipt", "budget"
}

extension ShoppingItem {
    /// What the Budget tab's expense sheet saves. Editing changes only the fields the sheet
    /// shows, so a shopping-list item stays on the list with its due date and creation time;
    /// a new expense is a "budget" item. `category` is the user's choice, "Other" included.
    static func expense(editing existing: ShoppingItem?, name: String, qty: Int, price: Int,
                        store: String, purchasedAt ms: Double, category: String) -> ShoppingItem {
        var item = existing ?? ShoppingItem(name: name, createdAt: ms, source: "budget")
        item.name = name
        item.qty = qty
        item.price = price
        item.isChecked = true
        item.purchasedAt = ms
        item.storeName = store.isEmpty ? nil : store
        item.location = category
        return item
    }
}

struct DietRecord: Codable, Identifiable, Equatable {
    var id: String = UUID().uuidString
    var date: String
    var name: String
    var ingredients: String = ""
    var expiryDate: String = ""
    var unitCalorie: Double = 0
    var portion: Double = 1.0
    var totalCalories: Double = 0
    var carbs: Double = 0.0
    var sugar: Double = 0.0
    var protein: Double = 0.0
    var fat: Double = 0.0
    var cholesterol: Double = 0.0
    var sodium: Double = 0.0
    var saturatedFat: Double? = nil
    var transFat: Double? = nil
    var sugarAlcohol: Double? = nil
    var foodCategory: String = "未分類"
}
