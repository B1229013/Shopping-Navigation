package com.example.shopping.ui.utils

import com.example.shopping.model.ShoppingItem
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CategoryClassifierTest {

    // ── Home → budget mapping ──────────────────────────────

    @Test
    fun everyHomeCategory_mapsToExpectedBudgetCategory() {
        listOf(HOME_PRODUCE, HOME_MEAT_SEAFOOD, HOME_EGG_DAIRY, HOME_STAPLES, HOME_SNACKS, HOME_SEASONING)
            .forEach { assertEquals(it, CAT_FOOD, budgetCategoryFor(it)) }
        assertEquals(CAT_BEVERAGES, budgetCategoryFor(HOME_BEVERAGES))
        assertEquals(CAT_GROCERIES, budgetCategoryFor(HOME_CLEANING))
        assertEquals(CAT_GROCERIES, budgetCategoryFor(HOME_KITCHEN))
        assertEquals(CAT_OTHER, budgetCategoryFor(HOME_OTHER))
        assertEquals(CAT_OTHER, budgetCategoryFor(null))
    }

    @Test
    fun withHomeCategory_keepsBudgetCategoryInSync() {
        val item = ShoppingItem(name = "醬油").withHomeCategory(HOME_SEASONING)
        assertEquals(HOME_SEASONING, item.homeCategory)
        assertEquals(CAT_FOOD, item.location)
    }

    // ── resolveHomeCategory: override > LLM > keywords ─────

    @Test
    fun override_beatsLlmAnswer() {
        assertEquals(HOME_SNACKS, resolveHomeCategory("可樂果", HOME_BEVERAGES))
        assertEquals(HOME_EGG_DAIRY, resolveHomeCategory("茶葉蛋", HOME_BEVERAGES))
        assertEquals(HOME_SNACKS, resolveHomeCategory("蒟蒻果凍", HOME_PRODUCE))
    }

    @Test
    fun validLlmAnswer_beatsKeywords() {
        // Keywords would say 蔬果 (番茄); trust the LLM.
        assertEquals(HOME_STAPLES, resolveHomeCategory("番茄義大利麵", HOME_STAPLES))
    }

    @Test
    fun missingOrInvalidLlmAnswer_fallsBackToKeywords() {
        assertEquals(HOME_BEVERAGES, resolveHomeCategory("果汁", null))
        assertEquals(HOME_BEVERAGES, resolveHomeCategory("果汁", "食物"))
    }

    @Test
    fun parseHomeCategory_acceptsDecoratedLabels() {
        assertEquals(HOME_STAPLES, parseHomeCategory("主食（米、麵、麵包）"))
        assertEquals(HOME_SNACKS, parseHomeCategory(" 零食 "))
        assertNull(parseHomeCategory("Food"))
        assertNull(parseHomeCategory(""))
    }

    // ── LLM reply parsing ──────────────────────────────────

    @Test
    fun parseClassifierResponse_alignsByIndex() {
        val reply = """{"results":[{"i":1,"category":"飲品"},{"i":0,"category":"零食"}]}"""
        assertEquals(listOf(HOME_SNACKS, HOME_BEVERAGES), parseClassifierResponse(reply, 2))
    }

    @Test
    fun parseClassifierResponse_leavesGapsForMissingOrBadEntries() {
        val reply = """{"results":[{"i":0,"category":"不存在"},{"i":9,"category":"零食"},{"category":"飲品"}]}"""
        assertEquals(listOf<String?>(null, null), parseClassifierResponse(reply, 2))
    }

    @Test(expected = Exception::class)
    fun parseClassifierResponse_throwsOnNonJson() {
        parseClassifierResponse("sorry, I can't help", 1)
    }

    @Test
    fun userMessage_numbersEachName() {
        assertEquals("0. 可樂果\n1. 果汁", buildClassifierUserMessage(listOf("可樂果", "果汁")))
    }

    @Test
    fun systemPrompt_listsEveryHomeCategory() {
        HOME_CATEGORIES.forEach { assertTrue(it, CLASSIFIER_SYSTEM_PROMPT.contains(it)) }
    }

    // ── Keyword fallback ───────────────────────────────────

    @Test
    fun fallback_previouslyFixedNames() {
        assertEquals(HOME_SNACKS, guessHomeCategory("可樂果"))
        assertEquals(HOME_EGG_DAIRY, guessHomeCategory("茶葉蛋"))
        assertEquals(HOME_SNACKS, guessHomeCategory("果凍"))
        listOf("果汁", "柳橙汁", "蘋果汁", "番茄汁", "蔬果汁").forEach {
            assertEquals(it, HOME_BEVERAGES, guessHomeCategory(it))
        }
    }

    @Test
    fun fallback_eachHomeCategory() {
        mapOf(
            "高麗菜" to HOME_PRODUCE, "蘋果" to HOME_PRODUCE,
            "豬肉" to HOME_MEAT_SEAFOOD, "鮭魚" to HOME_MEAT_SEAFOOD,
            "雞蛋" to HOME_EGG_DAIRY, "起司" to HOME_EGG_DAIRY, "鮮奶油" to HOME_EGG_DAIRY,
            "白米" to HOME_STAPLES, "吐司" to HOME_STAPLES, "烏龍麵" to HOME_STAPLES,
            "洋芋片" to HOME_SNACKS, "蛋糕" to HOME_SNACKS,
            "鮮奶" to HOME_BEVERAGES, "珍珠奶茶" to HOME_BEVERAGES,
            "醬油" to HOME_SEASONING, "番茄醬" to HOME_SEASONING, "奶油" to HOME_EGG_DAIRY,
            "衛生紙" to HOME_CLEANING, "洗碗精" to HOME_CLEANING, "菜瓜布" to HOME_CLEANING,
            "保鮮膜" to HOME_KITCHEN, "鋁箔紙" to HOME_KITCHEN,
            "電池" to HOME_OTHER, "" to HOME_OTHER
        ).forEach { (name, expected) -> assertEquals(name, expected, guessHomeCategory(name)) }
    }

    @Test
    fun fallback_avoidsKnownSubstringTraps() {
        assertEquals(HOME_PRODUCE, guessHomeCategory("玉米"))          // 米 isn't a staples keyword
        assertEquals(HOME_PRODUCE, guessHomeCategory("watermelon"))   // "water" isn't a beverage keyword
        assertNotEquals(HOME_KITCHEN, guessHomeCategory("火鍋料"))     // bare 鍋 isn't a kitchen keyword
        assertNotEquals(HOME_BEVERAGES, guessHomeCategory("醬汁"))
    }

    @Test
    fun guessCategory_derivesBudgetCategory() {
        assertEquals(CAT_FOOD, guessCategory("可樂果"))
        assertEquals(CAT_BEVERAGES, guessCategory("可樂"))
        assertEquals(CAT_FOOD, guessCategory("醬油"))       // 調味料 now counts as 食品
        assertEquals(CAT_GROCERIES, guessCategory("衛生紙"))
        assertEquals(CAT_OTHER, guessCategory("購物袋"))
    }
}
