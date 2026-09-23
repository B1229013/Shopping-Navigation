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
}

struct DietRecord: Codable, Identifiable, Equatable {
    var id: String = UUID().uuidString
    var date: String
    var name: String
    var ingredients: String = ""
    var expiryDate: String = ""
    var unitCalorie: Int = 0
    var portion: Double = 1.0
    var totalCalories: Int = 0
    var carbs: Double = 0.0
    var sugar: Double = 0.0
    var protein: Double = 0.0
    var fat: Double = 0.0
    var cholesterol: Double = 0.0
    var sodium: Double = 0.0
    var foodCategory: String = "未分類"
}
