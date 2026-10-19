package com.example.shopping.ui.utils

import com.example.shopping.BuildConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Manual probe: sends a batch of product names to the real LLM and prints what each one ends up as.
 * Skipped unless RUN_LLM_PROBE=1, so normal test runs never hit the network or spend API quota.
 *
 *   RUN_LLM_PROBE=1 ./gradlew :app:testDebugUnitTest --tests com.example.shopping.ui.utils.LlmClassificationProbe
 *
 * Output goes to the test's stdout, saved in the XML report under app/build/test-results/testDebugUnitTest.
 */
class LlmClassificationProbe {

    private val products = listOf(
        // Names with misleading characters
        "可樂果", "茶葉蛋", "果凍", "烏龍麵", "鮮奶油", "蒟蒻果凍", "番茄醬", "番茄汁", "檸檬汁",
        "玉米", "玉米片", "抹茶拿鐵", "蜂蜜蛋糕", "鹹蛋黃酥", "花生醬", "蘋果醋", "牛奶糖",
        // Between two categories
        "鮮奶", "優酪乳", "豆漿", "豆腐", "冷凍水餃", "火鍋料", "魚丸", "蟹肉棒", "鮪魚罐頭",
        "茶葉", "咖啡豆", "料理米酒", "麻油", "冬粉", "泡麵", "起司片", "奶油", "鹽酥雞",
        // Snacks / staples / produce
        "乖乖", "蘇打餅", "吐司", "可頌", "雞蛋", "高麗菜", "香蕉", "雞胸肉",
        // Non-food
        "洗碗精", "菜瓜布", "廚房紙巾", "保鮮盒", "衛生紙", "牙膏", "洗衣球",
        // Not clearly covered by the 10 categories
        "電池", "購物袋", "狗飼料", "貓砂", "尿布", "口罩", "衛生棉", "酒精", "OK繃", "原子筆",
        // English
        "Coca-Cola", "Greek yogurt"
    )

    @Test
    fun probe() = runBlocking {
        assumeTrue("set RUN_LLM_PROBE=1 to run", System.getenv("RUN_LLM_PROBE") == "1")

        val llm = products.chunked(50).flatMap { requestBatch(it, BuildConfig.OPENAI_API_KEY) }
        println("name\tllm_raw\tfinal\tbudget\tkeyword_only\toverride")
        products.forEachIndexed { i, name ->
            val final = resolveHomeCategory(name, llm[i])
            println(listOf(
                name, llm[i] ?: "-", final, budgetCategoryFor(final),
                guessHomeCategory(name), if (findOverride(name) != null) "Y" else ""
            ).joinToString("\t"))
        }
    }
}
