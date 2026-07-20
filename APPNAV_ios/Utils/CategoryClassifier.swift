import Foundation

func normalizeCategory(_ cat: String, _ name: String) -> String {
    let catLower = cat.lowercased()
    if catLower.contains("food") || catLower.contains("食") { return "Food" }
    if catLower.contains("bev") || catLower.contains("飲") { return "Beverages" }
    if catLower.contains("groc") || catLower.contains("生活") { return "Groceries" }
    return guessCategory(from: name)
}

func guessCategory(from name: String) -> String {
    let foodKeywords = ["肉", "菜", "魚", "蛋", "豆腐", "米", "麵", "雞", "豬", "牛", "蔬", "水果", "果", "菇", "蝦", "海鮮", "麵包", "饅頭", "包子"]
    let bevKeywords = ["茶", "水", "飲料", "咖啡", "奶", "牛奶", "果汁", "啤酒", "酒", "汽水", "可樂"]
    let grocKeywords = ["醬", "油", "鹽", "糖", "醋", "調味", "洗碗", "洗髮", "衛生紙", "清潔", "牙刷", "牙膏", "肥皂", "洗手", "毛巾"]

    if foodKeywords.contains(where: { name.contains($0) }) { return "Food" }
    if bevKeywords.contains(where: { name.contains($0) }) { return "Beverages" }
    if grocKeywords.contains(where: { name.contains($0) }) { return "Groceries" }
    return "Other"
}

func matchesHomeCategory(_ itemName: String, _ category: String) -> Bool {
    switch category {
    case "蔬菜": return ["菜", "蔬", "菠菜", "高麗", "白菜", "花椰", "空心菜", "韭菜", "芹菜", "番茄", "茄子", "黃瓜", "玉米"].contains(where: { itemName.contains($0) })
    case "水果": return ["果", "蘋果", "香蕉", "橘子", "葡萄", "西瓜", "芒果", "草莓", "桃", "李", "梨", "柿"].contains(where: { itemName.contains($0) })
    case "零食": return ["餅乾", "薯片", "巧克力", "糖果", "零食", "糕", "餅", "蛋糕"].contains(where: { itemName.contains($0) })
    case "蛋奶": return ["蛋", "奶", "乳", "起司", "優格", "奶油", "乳酪"].contains(where: { itemName.contains($0) })
    case "飲品": return ["茶", "水", "飲料", "咖啡", "奶", "果汁", "啤酒", "酒", "汽水"].contains(where: { itemName.contains($0) })
    case "調味": return ["醬", "油", "鹽", "糖", "醋", "醬油", "沙拉", "番茄醬", "辣椒醬"].contains(where: { itemName.contains($0) })
    default: return false
    }
}

struct CategoryInfo: Identifiable {
    let id = UUID()
    let name: String
    let catId: String
    let icon: String
    let colorHex: UInt
    let description: String
}

let dashboardCategories: [CategoryInfo] = [
    CategoryInfo(name: "食品", catId: "Food", icon: "fork.knife", colorHex: 0xFF3B82F6, description: "肉類、蔬菜、生鮮食品"),
    CategoryInfo(name: "飲品", catId: "Beverages", icon: "cup.and.saucer.fill", colorHex: 0xFF10B981, description: "牛奶、茶、飲料"),
    CategoryInfo(name: "生活用品", catId: "Groceries", icon: "bag.fill", colorHex: 0xFFF59E0B, description: "調味料、居家用品"),
    CategoryInfo(name: "其他", catId: "Other", icon: "ellipsis.circle.fill", colorHex: 0xFF8B5CF6, description: "點數、手續費、其他項目"),
]
