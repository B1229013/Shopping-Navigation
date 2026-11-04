import Foundation

// MARK: - Category IDs (BudgetView + HomeView filter on these exact strings)
let CAT_FOOD = "Food"
let CAT_BEVERAGES = "Beverages"
let CAT_GROCERIES = "Groceries"
let CAT_OTHER = "Other"

// Substring keywords -> category. Order matters: more specific first
// (e.g. 豆漿 before 漿, 奶茶 before 奶). Ported from the Android build.
private let beverageKeywords = [
    "珍珠奶茶", "奶茶", "豆漿", "豆奶", "鮮奶", "牛奶", "優酪乳", "養樂多",
    "milk", "soymilk", "soy milk",
    "綠茶", "紅茶", "烏龍", "奶綠", "茶",
    "拿鐵", "咖啡", "美式", "卡布", "摩卡", "espresso", "latte", "coffee", "mocha", "cappuccino", "americano",
    "果汁", "柳橙汁", "蘋果汁", "葡萄汁", "juice", "smoothie",
    "汽水", "可樂", "雪碧", "七喜", "氣泡", "soda", "cola", "coke", "sprite", "pepsi", "fanta",
    "啤酒", "紅酒", "白酒", "威士忌", "高粱", "清酒", "beer", "wine", "whisky", "whiskey", "vodka", "sake",
    "礦泉水", "飲用水", "water",
    "運動飲料", "舒跑", "寶礦力", "能量飲", "飲料", "drink", "beverage"
]

// Sub-lists surfaced for HomeView's fine-grained chips.
let vegetableKeywords = [
    "高麗菜", "番茄", "紅蘿蔔", "胡蘿蔔", "洋蔥", "青椒", "甜椒", "青花菜", "花椰菜",
    "菠菜", "玉米", "馬鈴薯", "土豆", "地瓜", "番薯", "蘑菇", "香菇", "茄子",
    "小黃瓜", "黃瓜", "蒜", "薑", "蔥", "萵苣", "生菜", "芹菜", "空心菜", "菜",
    "tomato", "carrot", "onion", "pepper", "broccoli", "spinach", "corn", "potato",
    "mushroom", "eggplant", "cucumber", "lettuce", "vegetable", "cabbage", "garlic", "ginger"
]

let fruitKeywords = [
    "蘋果", "香蕉", "橘子", "柳丁", "橙", "葡萄", "西瓜", "芒果", "草莓", "檸檬",
    "鳳梨", "酪梨", "藍莓", "奇異果", "桃", "梨", "櫻桃", "柚子", "柚", "文旦", "榴槤", "榴蓮", "火龍果", "芭樂", "蓮霧", "木瓜", "哈密瓜", "龍眼", "荔枝", "山竹", "果",
    "apple", "banana", "orange", "grape", "watermelon", "mango", "strawberry",
    "lemon", "pineapple", "avocado", "blueberry", "kiwi", "peach", "pear", "cherry", "fruit"
]

let snackKeywords = [
    "餅乾", "巧克力", "蛋糕", "甜甜圈", "冰淇淋", "洋芋片", "糖果", "布丁", "麻糬", "軟糖",
    "零食", "點心",
    "cookie", "biscuit", "chocolate", "cake", "donut", "icecream", "chips", "candy", "pudding", "snack"
]

let eggDairyKeywords = [
    "雞蛋", "鴨蛋", "皮蛋", "鹹蛋", "蛋",
    "起司", "乳酪", "優格", "奶油", "鮮奶油", "奶粉", "牛奶", "鮮奶", "優酪乳",
    "egg", "cheese", "yogurt", "yoghurt", "butter", "cream", "milk", "dairy"
]

let seasoningKeywords = [
    "鹽", "糖", "醬油", "醋", "味噌", "辣椒醬", "番茄醬", "美乃滋", "沙拉醬",
    "胡椒", "油", "橄欖油", "沙拉油", "芝麻油", "料酒", "太白粉", "麵粉", "澱粉",
    "蜂蜜", "果醬", "調味",
    "salt", "sugar", "sauce", "vinegar", "oil", "flour", "honey", "jam", "seasoning", "spice"
]

private let foodKeywords = [
    // meats / seafood
    "豬肉", "牛肉", "雞肉", "羊肉", "鴨肉", "火腿", "培根", "香腸", "肉",
    "鮭魚", "鮪魚", "鯖魚", "魚", "蝦", "蟹", "蛤蜊", "蚵", "干貝", "海鮮",
    "pork", "beef", "chicken", "lamb", "bacon", "ham", "sausage", "fish", "salmon", "tuna", "shrimp", "crab",
    // eggs / dairy / plant protein
    "雞蛋", "鴨蛋", "皮蛋", "鹹蛋", "蛋",
    "起司", "乳酪", "優格", "cheese", "yogurt",
    "豆腐", "豆皮", "豆乾", "tofu",
    "堅果", "杏仁", "核桃", "腰果", "花生", "nut",
    // staples
    "白飯", "米飯", "糙米", "米", "rice",
    "麵包", "吐司", "貝果", "可頌", "bread", "toast",
    "麵條", "拉麵", "烏龍麵", "義大利麵", "泡麵", "麵", "noodle", "pasta", "ramen",
    "饅頭", "包子", "水餃", "餃子", "餛飩", "燒賣", "dumpling",
    "燕麥", "麥片", "穀片", "oat", "cereal",
    // vegetables
    "高麗菜", "番茄", "紅蘿蔔", "胡蘿蔔", "洋蔥", "青椒", "甜椒", "青花菜", "花椰菜",
    "菠菜", "玉米", "馬鈴薯", "土豆", "地瓜", "番薯", "蘑菇", "香菇", "茄子",
    "小黃瓜", "黃瓜", "蒜", "薑", "蔥", "萵苣", "生菜", "芹菜", "空心菜", "菜",
    "tomato", "carrot", "onion", "pepper", "broccoli", "spinach", "corn", "potato", "mushroom", "eggplant", "cucumber", "lettuce", "vegetable",
    // fruits
    "蘋果", "香蕉", "橘子", "柳丁", "橙", "葡萄", "西瓜", "芒果", "草莓", "檸檬",
    "鳳梨", "酪梨", "藍莓", "奇異果", "桃", "梨", "櫻桃", "柚子", "柚", "文旦", "榴槤", "榴蓮", "火龍果", "芭樂", "蓮霧", "木瓜", "哈密瓜", "龍眼", "荔枝", "山竹", "果",
    "apple", "banana", "orange", "grape", "watermelon", "mango", "strawberry", "lemon", "pineapple", "avocado", "blueberry", "kiwi", "peach", "pear", "cherry", "fruit",
    // snacks / sweets
    "餅乾", "巧克力", "蛋糕", "甜甜圈", "冰淇淋", "洋芋片", "糖果", "布丁", "麻糬", "軟糖", "奶糖", "奶油",
    "cookie", "biscuit", "chocolate", "cake", "donut", "icecream", "chips", "candy", "pudding",
    // dishes
    "沙拉", "壽司", "披薩", "漢堡", "薯條", "三明治", "湯", "火鍋", "咖哩", "便當", "義大利",
    "salad", "sushi", "pizza", "burger", "fries", "sandwich", "soup", "hotpot", "curry", "bento",
    // condiments treated as food ingredients
    "蜂蜜", "honey", "果醬", "jam"
]

private let groceriesKeywords = [
    // seasonings / cooking
    "鹽", "糖", "醬油", "醋", "味噌", "辣椒醬", "番茄醬", "美乃滋", "沙拉醬",
    "胡椒", "油", "橄欖油", "沙拉油", "芝麻油", "料酒", "太白粉", "麵粉", "澱粉",
    "salt", "sugar", "sauce", "vinegar", "oil", "flour",
    // cleaning / household
    "衛生紙", "面紙", "紙巾", "廚房紙", "濕紙巾",
    "洗碗精", "洗衣精", "洗衣粉", "柔軟精", "漂白水", "清潔劑", "去汙", "菜瓜布",
    "洗髮精", "潤髮", "護髮", "沐浴乳", "肥皂", "香皂", "洗手乳",
    "牙膏", "牙刷", "牙線", "漱口水", "棉花棒",
    "垃圾袋", "保鮮膜", "鋁箔紙", "夾鏈袋",
    "毛巾", "抹布", "海綿",
    "電池", "燈泡", "插座", "延長線",
    "tissue", "paper towel", "detergent", "soap", "shampoo", "toothpaste", "battery"
]

private let validCategoryIds: Set<String> = [CAT_FOOD, CAT_BEVERAGES, CAT_GROCERIES, CAT_OTHER]

/// Best-effort keyword classification into Food / Beverages / Groceries / Other.
/// Of all keywords found in the name (case-insensitive), the one ending last wins — a
/// compound name's head comes last (茶葉「蛋」, 抹茶「餅乾」, 菜瓜「布」, watermelon) — then
/// the longest one (番茄醬 over 番茄); ties go Beverages, Food, Groceries.
func guessCategory(from name: String) -> String {
    let n = name.trimmingCharacters(in: .whitespaces).lowercased()
    if n.isEmpty { return CAT_OTHER }
    var best: (end: Int, length: Int, category: String)?
    for (category, keywords) in [(CAT_BEVERAGES, beverageKeywords), (CAT_FOOD, foodKeywords),
                                 (CAT_GROCERIES, groceriesKeywords)] {
        for k in keywords {
            let kw = k.lowercased()
            guard let r = n.range(of: kw, options: .backwards) else { continue }
            let end = n.distance(from: n.startIndex, to: r.upperBound)
            if let b = best, (end, kw.count) <= (b.end, b.length) { continue }
            best = (end, kw.count, category)
        }
    }
    return best?.category ?? CAT_OTHER
}

/// Normalize a raw category hint (e.g. an LLM response saying "食品" / "food" / blank)
/// into one of the canonical catIds, falling back to keyword guessing on the name.
func normalizeCategory(_ rawCategory: String, _ itemName: String) -> String {
    // Beverage names are easy to misclassify — the receipt-tidying LLM often puts
    // milk / juice into Food. If the name reads as a drink, override.
    if guessCategory(from: itemName) == CAT_BEVERAGES { return CAT_BEVERAGES }

    let raw = rawCategory.trimmingCharacters(in: .whitespaces).lowercased()
    if raw.isEmpty { return guessCategory(from: itemName) }
    if ["food", "食品", "食物"].contains(raw) { return CAT_FOOD }
    if ["beverages", "beverage", "飲品", "飲料"].contains(raw) { return CAT_BEVERAGES }
    if ["groceries", "grocery", "生活用品", "日用品", "調味料"].contains(raw) { return CAT_GROCERIES }
    if ["other", "others", "其他"].contains(raw) { return guessCategory(from: itemName) }
    if let match = validCategoryIds.first(where: { $0.lowercased() == raw }) { return match }
    return guessCategory(from: itemName)
}

/// Matches an item against HomeView's finer-grained chips (蔬菜/水果/零食/蛋奶/飲品/調味).
/// Uses the item name directly — independent of the coarse 4-category `location`.
func matchesHomeCategory(_ itemName: String, _ category: String) -> Bool {
    if itemName.trimmingCharacters(in: .whitespaces).isEmpty { return false }
    let n = itemName.lowercased()
    let list: [String]
    switch category {
    case "蔬菜": list = vegetableKeywords
    case "水果": list = fruitKeywords
    case "零食": list = snackKeywords
    case "蛋奶": list = eggDairyKeywords
    case "飲品": list = beverageKeywords
    case "調味": list = seasoningKeywords
    default: return false
    }
    return list.contains(where: { n.contains($0.lowercased()) })
}

/// Re-run classification so items with nil / stale locations get assigned properly.
func reclassifyAll(_ items: [ShoppingItem]) -> [ShoppingItem] {
    items.map { item in
        let correct = normalizeCategory(item.location ?? "", item.name)
        if item.location != correct {
            var copy = item
            copy.location = correct
            return copy
        }
        return item
    }
}

// MARK: - Dashboard category metadata (UI)
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
