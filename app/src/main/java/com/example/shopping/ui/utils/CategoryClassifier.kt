package com.example.shopping.ui.utils

import com.example.shopping.model.ShoppingItem

// Budget category IDs (stored in ShoppingItem.location; BudgetScreen filters on these exact strings).
const val CAT_FOOD = "Food"
const val CAT_BEVERAGES = "Beverages"
const val CAT_GROCERIES = "Groceries"
const val CAT_OTHER = "Other"

// HomeScreen chip categories (stored in ShoppingItem.homeCategory). Each item has exactly one.
const val HOME_PRODUCE = "蔬果"
const val HOME_MEAT_SEAFOOD = "肉品海鮮"
const val HOME_EGG_DAIRY = "蛋奶"
const val HOME_STAPLES = "主食"
const val HOME_SNACKS = "零食"
const val HOME_BEVERAGES = "飲品"
const val HOME_SEASONING = "調味料"
const val HOME_CLEANING = "清潔用品"
const val HOME_KITCHEN = "廚房用品"
const val HOME_OTHER = "其他"

val HOME_CATEGORIES = listOf(
    HOME_PRODUCE, HOME_MEAT_SEAFOOD, HOME_EGG_DAIRY, HOME_STAPLES, HOME_SNACKS,
    HOME_BEVERAGES, HOME_SEASONING, HOME_CLEANING, HOME_KITCHEN, HOME_OTHER
)

/** Each home chip rolls up into exactly one budget category, so the two views never disagree. */
fun budgetCategoryFor(homeCategory: String?): String = when (homeCategory) {
    HOME_PRODUCE, HOME_MEAT_SEAFOOD, HOME_EGG_DAIRY, HOME_STAPLES, HOME_SNACKS, HOME_SEASONING -> CAT_FOOD
    HOME_BEVERAGES -> CAT_BEVERAGES
    HOME_CLEANING, HOME_KITCHEN -> CAT_GROCERIES
    else -> CAT_OTHER
}

// Manual corrections: name substring → home category. These win over both the LLM and the keyword
// fallback, so use them for products the LLM keeps getting wrong or whose names mislead the keywords
// (e.g. 可樂果 contains 可樂 and 果).
private val productOverrides = linkedMapOf(
    "可樂果" to HOME_SNACKS,
    "茶葉蛋" to HOME_EGG_DAIRY,
    "果凍" to HOME_SNACKS,
    "烏龍麵" to HOME_STAPLES,
    "鮮奶油" to HOME_EGG_DAIRY
)

internal fun findOverride(name: String): String? {
    val n = name.lowercase()
    return productOverrides.entries.firstOrNull { n.contains(it.key.lowercase()) }?.value
}

// ── Keyword fallback (used when the LLM is unavailable or returns something unusable) ──
// Substring keywords per home category. Lists are checked in `keywordOrder`; the first hit wins,
// so lists whose words hide inside other names (e.g. 蛋 in 蛋糕, 菜 in 菜瓜布) are checked later.
// Milk counts as a beverage (as the budget always did); 蛋奶 is eggs, cheese, yogurt, butter.

private val beverageKeywords = listOf(
    "珍珠奶茶", "奶茶", "豆漿", "豆奶", "鮮奶", "牛奶", "保久乳", "優酪乳", "養樂多",
    "milk", "soymilk", "soy milk",
    "綠茶", "紅茶", "烏龍", "奶綠", "茶",
    "拿鐵", "咖啡", "美式", "卡布", "摩卡", "espresso", "latte", "coffee", "mocha", "cappuccino", "americano",
    "果汁", "柳橙汁", "蘋果汁", "葡萄汁", "番茄汁", "蔬菜汁", "蔬果汁", "juice", "smoothie",
    "汽水", "可樂", "雪碧", "七喜", "氣泡", "soda", "cola", "coke", "sprite", "pepsi", "fanta",
    "啤酒", "紅酒", "白酒", "威士忌", "高粱", "清酒", "beer", "wine", "whisky", "whiskey", "vodka", "sake",
    "礦泉水", "飲用水", "mineral water", "bottled water",
    "運動飲料", "舒跑", "寶礦力", "能量飲", "飲料", "drink", "beverage"
)

private val cleaningKeywords = listOf(
    "衛生紙", "面紙", "紙巾", "濕紙巾",
    "洗碗精", "洗衣精", "洗衣粉", "柔軟精", "漂白水", "清潔劑", "去汙", "菜瓜布",
    "洗髮", "潤髮", "護髮", "沐浴", "肥皂", "香皂", "洗手乳",
    "牙膏", "牙刷", "牙線", "漱口水", "棉花棒",
    "毛巾", "抹布",
    "tissue", "detergent", "soap", "shampoo", "toothpaste"
)

// Deliberately no bare 鍋/杯/碗 (火鍋料, 杯麵, 碗粿) or pan/pot (potato).
private val kitchenKeywords = listOf(
    "廚房紙", "保鮮膜", "保鮮盒", "鋁箔紙", "夾鏈袋", "烘焙紙",
    "平底鍋", "炒鍋", "湯鍋", "砧板", "菜刀", "鍋鏟", "湯匙", "筷子", "叉子", "餐具",
    "paper towel", "aluminum foil", "plastic wrap"
)

private val snackKeywords = listOf(
    "餅乾", "巧克力", "蛋糕", "甜甜圈", "冰淇淋", "洋芋片", "糖果", "布丁", "麻糬", "軟糖",
    "零食", "點心",
    "cookie", "biscuit", "chocolate", "cake", "donut", "icecream", "ice cream", "chips", "candy", "pudding", "snack"
)

private val eggDairyKeywords = listOf(
    "雞蛋", "鴨蛋", "皮蛋", "鹹蛋", "蛋",
    "起司", "乳酪", "優格", "鮮奶油", "奶油", "奶粉",
    "egg", "cheese", "yogurt", "yoghurt", "butter", "cream", "dairy"
)

private val seasoningKeywords = listOf(
    "鹽", "糖", "醬油", "醋", "味噌", "辣椒醬", "番茄醬", "美乃滋", "沙拉醬",
    "胡椒", "橄欖油", "沙拉油", "芝麻油", "油", "料酒", "太白粉", "麵粉", "澱粉",
    "蜂蜜", "果醬", "調味",
    "salt", "sugar", "sauce", "vinegar", "oil", "flour", "honey", "jam", "seasoning", "spice"
)

private val meatSeafoodKeywords = listOf(
    "豬肉", "牛肉", "雞肉", "羊肉", "鴨肉", "火腿", "培根", "香腸", "肉",
    "鮭魚", "鮪魚", "鯖魚", "魚", "蝦", "蟹", "蛤蜊", "蚵", "干貝", "海鮮",
    "pork", "beef", "chicken", "lamb", "bacon", "ham", "sausage", "fish", "salmon", "tuna", "shrimp", "crab"
)

private val staplesKeywords = listOf(
    "白米", "糙米", "米飯", "白飯", "米粉", "rice",
    "麵包", "吐司", "貝果", "可頌", "bread", "toast",
    "麵條", "拉麵", "義大利麵", "泡麵", "麵", "noodle", "pasta", "ramen",
    "饅頭", "包子", "水餃", "餃子", "餛飩", "dumpling",
    "燕麥", "麥片", "穀片", "oat", "cereal"
)

private val produceKeywords = listOf(
    "高麗菜", "番茄", "紅蘿蔔", "胡蘿蔔", "洋蔥", "青椒", "甜椒", "青花菜", "花椰菜",
    "菠菜", "玉米", "馬鈴薯", "土豆", "地瓜", "番薯", "蘑菇", "香菇", "茄子",
    "小黃瓜", "黃瓜", "蒜", "薑", "蔥", "萵苣", "生菜", "芹菜", "空心菜", "豆腐", "菜",
    "蘋果", "香蕉", "橘子", "柳丁", "橙", "葡萄", "西瓜", "芒果", "草莓", "檸檬",
    "鳳梨", "酪梨", "藍莓", "奇異果", "桃", "梨", "櫻桃", "果",
    "tomato", "carrot", "onion", "pepper", "broccoli", "spinach", "corn", "potato",
    "mushroom", "eggplant", "cucumber", "lettuce", "vegetable", "cabbage", "garlic", "ginger",
    "apple", "banana", "orange", "grape", "watermelon", "mango", "strawberry",
    "lemon", "pineapple", "avocado", "blueberry", "kiwi", "peach", "pear", "cherry", "fruit"
)

private val keywordOrder = listOf(
    HOME_CLEANING to cleaningKeywords,
    HOME_KITCHEN to kitchenKeywords,
    HOME_BEVERAGES to beverageKeywords,
    HOME_SNACKS to snackKeywords,
    HOME_EGG_DAIRY to eggDairyKeywords,
    HOME_SEASONING to seasoningKeywords,
    HOME_MEAT_SEAFOOD to meatSeafoodKeywords,
    HOME_STAPLES to staplesKeywords,
    HOME_PRODUCE to produceKeywords
)

/** Keyword-only guess of the home category. Overrides first, then `keywordOrder`, else 其他. */
fun guessHomeCategory(name: String): String {
    if (name.isBlank()) return HOME_OTHER
    findOverride(name)?.let { return it }
    val n = name.lowercase()
    return keywordOrder.firstOrNull { (_, words) -> words.any { n.contains(it.lowercase()) } }?.first
        ?: HOME_OTHER
}

/** Keyword-only guess of the budget category, derived from the home category. */
fun guessCategory(name: String): String = budgetCategoryFor(guessHomeCategory(name))

/**
 * Map a raw LLM label onto a canonical home category, or null if it isn't one.
 * Tolerates decorated labels such as "主食（米、麵、麵包）".
 */
fun parseHomeCategory(raw: String?): String? {
    val r = raw?.trim().orEmpty()
    if (r.isEmpty()) return null
    return HOME_CATEGORIES.firstOrNull { it == r } ?: HOME_CATEGORIES.firstOrNull { r.contains(it) }
}

/** Set the home category and keep the budget category in sync with it. */
fun ShoppingItem.withHomeCategory(homeCategory: String): ShoppingItem =
    copy(homeCategory = homeCategory, location = budgetCategoryFor(homeCategory))

/** Final decision for one item: manual override > valid LLM answer > keyword fallback. */
fun resolveHomeCategory(name: String, llmCategory: String?): String =
    findOverride(name) ?: parseHomeCategory(llmCategory) ?: guessHomeCategory(name)
