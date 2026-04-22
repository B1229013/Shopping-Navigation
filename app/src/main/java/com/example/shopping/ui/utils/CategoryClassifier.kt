package com.example.shopping.ui.utils

// Category IDs used throughout the app (BudgetScreen + HomeScreen filter on these exact strings).
const val CAT_FOOD = "Food"
const val CAT_BEVERAGES = "Beverages"
const val CAT_GROCERIES = "Groceries"
const val CAT_OTHER = "Other"

// Substring keywords → category. Order matters: more specific first (e.g. 豆漿 before 漿, 奶茶 before 奶).
private val beverageKeywords = listOf(
    "珍珠奶茶", "奶茶", "豆漿", "豆奶", "鮮奶", "牛奶", "優酪乳", "養樂多",
    "milk", "soymilk", "soy milk",
    "綠茶", "紅茶", "烏龍", "奶綠", "茶",
    "拿鐵", "咖啡", "美式", "卡布", "摩卡", "espresso", "latte", "coffee", "mocha", "cappuccino", "americano",
    "果汁", "柳橙汁", "蘋果汁", "葡萄汁", "juice", "smoothie",
    "汽水", "可樂", "雪碧", "七喜", "氣泡", "soda", "cola", "coke", "sprite", "pepsi", "fanta",
    "啤酒", "紅酒", "白酒", "威士忌", "高粱", "清酒", "beer", "wine", "whisky", "whiskey", "vodka", "sake",
    "礦泉水", "飲用水", "water",
    "運動飲料", "舒跑", "寶礦力", "能量飲", "飲料", "drink", "beverage"
)

// Sub-lists surfaced for HomeScreen's fine-grained chips.
internal val vegetableKeywords = listOf(
    "高麗菜", "番茄", "紅蘿蔔", "胡蘿蔔", "洋蔥", "青椒", "甜椒", "青花菜", "花椰菜",
    "菠菜", "玉米", "馬鈴薯", "土豆", "地瓜", "番薯", "蘑菇", "香菇", "茄子",
    "小黃瓜", "黃瓜", "蒜", "薑", "蔥", "萵苣", "生菜", "芹菜", "空心菜", "菜",
    "tomato", "carrot", "onion", "pepper", "broccoli", "spinach", "corn", "potato",
    "mushroom", "eggplant", "cucumber", "lettuce", "vegetable", "cabbage", "garlic", "ginger"
)

internal val fruitKeywords = listOf(
    "蘋果", "香蕉", "橘子", "柳丁", "橙", "葡萄", "西瓜", "芒果", "草莓", "檸檬",
    "鳳梨", "酪梨", "藍莓", "奇異果", "桃", "梨", "櫻桃", "果",
    "apple", "banana", "orange", "grape", "watermelon", "mango", "strawberry",
    "lemon", "pineapple", "avocado", "blueberry", "kiwi", "peach", "pear", "cherry", "fruit"
)

internal val snackKeywords = listOf(
    "餅乾", "巧克力", "蛋糕", "甜甜圈", "冰淇淋", "洋芋片", "糖果", "布丁", "麻糬", "軟糖",
    "零食", "點心",
    "cookie", "biscuit", "chocolate", "cake", "donut", "icecream", "chips", "candy", "pudding", "snack"
)

internal val eggDairyKeywords = listOf(
    "雞蛋", "鴨蛋", "皮蛋", "鹹蛋", "蛋",
    "起司", "乳酪", "優格", "奶油", "鮮奶油", "奶粉", "牛奶", "鮮奶", "優酪乳",
    "egg", "cheese", "yogurt", "yoghurt", "butter", "cream", "milk", "dairy"
)

internal val seasoningKeywords = listOf(
    "鹽", "糖", "醬油", "醋", "味噌", "辣椒醬", "番茄醬", "美乃滋", "沙拉醬",
    "胡椒", "油", "橄欖油", "沙拉油", "芝麻油", "料酒", "太白粉", "麵粉", "澱粉",
    "蜂蜜", "果醬", "調味",
    "salt", "sugar", "sauce", "vinegar", "oil", "flour", "honey", "jam", "seasoning", "spice"
)

private val foodKeywords = listOf(
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
    "鳳梨", "酪梨", "藍莓", "奇異果", "桃", "梨", "櫻桃", "果",
    "apple", "banana", "orange", "grape", "watermelon", "mango", "strawberry", "lemon", "pineapple", "avocado", "blueberry", "kiwi", "peach", "pear", "cherry", "fruit",
    // snacks / sweets
    "餅乾", "巧克力", "蛋糕", "甜甜圈", "冰淇淋", "洋芋片", "糖果", "布丁", "麻糬", "軟糖",
    "cookie", "biscuit", "chocolate", "cake", "donut", "icecream", "chips", "candy", "pudding",
    // dishes
    "沙拉", "壽司", "披薩", "漢堡", "薯條", "三明治", "湯", "火鍋", "咖哩", "便當", "義大利",
    "salad", "sushi", "pizza", "burger", "fries", "sandwich", "soup", "hotpot", "curry", "bento",
    // condiments treated as food ingredients
    "蜂蜜", "honey", "果醬", "jam"
)

private val groceriesKeywords = listOf(
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
)

/**
 * Best-effort keyword classification of an item name into one of:
 * Food / Beverages / Groceries / Other. Matches are substring, case-insensitive.
 * Beverage check runs first so "奶茶" doesn't get grabbed by the generic 茶/奶 food tokens.
 */
fun guessCategory(name: String): String {
    if (name.isBlank()) return CAT_OTHER
    val n = name.lowercase()
    if (beverageKeywords.any { n.contains(it.lowercase()) }) return CAT_BEVERAGES
    if (foodKeywords.any { n.contains(it.lowercase()) }) return CAT_FOOD
    if (groceriesKeywords.any { n.contains(it.lowercase()) }) return CAT_GROCERIES
    return CAT_OTHER
}

private val validCategoryIds = setOf(CAT_FOOD, CAT_BEVERAGES, CAT_GROCERIES, CAT_OTHER)

/**
 * Normalize a raw category hint (e.g. from an LLM response that might say "食品" or "food"
 * or leave it blank) into one of the canonical catIds, falling back to keyword guessing
 * on the item name.
 */
fun normalizeCategory(rawCategory: String?, itemName: String): String {
    // Beverage names are easy to misclassify — the receipt-tidying LLM often puts milk / juice into Food
    // because "dairy" is also listed under Food. If the name clearly reads as a drink, override.
    val nameLower = itemName.lowercase()
    if (beverageKeywords.any { nameLower.contains(it.lowercase()) }) return CAT_BEVERAGES

    val raw = rawCategory?.trim()?.lowercase().orEmpty()
    return when {
        raw.isEmpty() -> guessCategory(itemName)
        raw in setOf("food", "食品", "食物") -> {
            // Double-check in case LLM lumped a beverage in; above check already handled that.
            CAT_FOOD
        }
        raw in setOf("beverages", "beverage", "飲品", "飲料") -> CAT_BEVERAGES
        raw in setOf("groceries", "grocery", "生活用品", "日用品", "調味料") -> CAT_GROCERIES
        raw in setOf("other", "others", "其他") -> {
            val guessed = guessCategory(itemName)
            if (guessed == CAT_OTHER) CAT_OTHER else guessed
        }
        validCategoryIds.any { it.equals(raw, ignoreCase = true) } -> validCategoryIds.first { it.equals(raw, ignoreCase = true) }
        else -> guessCategory(itemName)
    }
}

/**
 * Matches an item against HomeScreen's finer-grained chips (蔬菜/水果/零食/蛋奶/飲品/調味).
 * Uses the item name directly — independent of the coarse 4-category `location`.
 */
fun matchesHomeCategory(itemName: String, chipLabel: String): Boolean {
    if (itemName.isBlank()) return false
    val n = itemName.lowercase()
    val list = when (chipLabel) {
        "蔬菜" -> vegetableKeywords
        "水果" -> fruitKeywords
        "零食" -> snackKeywords
        "蛋奶" -> eggDairyKeywords
        "飲品" -> beverageKeywords
        "調味" -> seasoningKeywords
        else -> return false
    }
    return list.any { n.contains(it.lowercase()) }
}

/** Re-run classification on a list so items with null / stale locations get assigned properly. */
fun reclassifyAll(items: List<com.example.shopping.model.ShoppingItem>): List<com.example.shopping.model.ShoppingItem> {
    return items.map { item ->
        val correct = normalizeCategory(item.location, item.name)
        if (item.location != correct) item.copy(location = correct) else item
    }
}
