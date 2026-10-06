package com.example.shopping.ui.utils

import android.util.Log
import com.example.shopping.ui.screens.GroqMessage
import com.example.shopping.ui.screens.GroqRequest
import com.example.shopping.ui.screens.GroqResponseFormat
import com.example.shopping.ui.screens.groqApi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

const val CLASSIFIER_MODEL = "gpt-5.6-luna"

// Keeps a single request (and its JSON reply) a manageable size when backfilling a long list.
private const val BATCH_SIZE = 50

/**
 * @param categories one home category per input name, same order. Always filled — keyword fallback
 *        covers anything the LLM didn't answer.
 * @param error set when the LLM call itself failed (network, bad model name, unparseable reply).
 */
data class ClassificationResult(val categories: List<String>, val error: String? = null)

internal val CLASSIFIER_SYSTEM_PROMPT = """
    You classify supermarket product names (usually Traditional Chinese) for a Taiwanese shopping app.
    Put every product into exactly ONE of these categories, using the category name exactly as written:
    - 蔬果: vegetables, fruits, mushrooms, tofu, fresh herbs
    - 肉品海鮮: meat, poultry, fish, seafood, processed meat (ham, bacon, sausage)
    - 蛋奶: eggs and egg products, cheese, yogurt, butter, cream (NOT drinking milk)
    - 主食: rice, noodles, bread, dumplings, oats and cereal
    - 零食: snacks, sweets, cookies, chips, candy, jelly, desserts, ice cream
    - 飲品: anything you drink, including milk, soy milk, tea, coffee, juice, soda, water, alcohol
    - 調味料: salt, sugar, sauces, oils, vinegar, spices, flour, honey, jam
    - 清潔用品: tissue and toilet paper, laundry and dish detergent, cleaning agents, shampoo, body wash, soap, toothpaste, toothbrush
    - 廚房用品: plastic wrap, foil, food containers, zip bags, kitchen paper, cookware, utensils, tableware
    - 其他: anything else (batteries, stationery, shopping bags, fees, points, discounts)
    Judge by what the product actually is, not by individual characters in its name
    (e.g. 可樂果 is a snack, not a drink; 茶葉蛋 is an egg, not tea; 果汁 is a drink, not fruit).

    The user sends a numbered list. Reply with ONLY this JSON, one entry per input number:
    {"results": [{"i": 0, "category": "零食"}]}
""".trimIndent()

internal fun buildClassifierUserMessage(names: List<String>): String =
    names.mapIndexed { i, name -> "$i. $name" }.joinToString("\n")

private val lenientJson = Json { ignoreUnknownKeys = true; isLenient = true }

/**
 * Parse the model's reply into a list aligned with the input (null where the model gave no usable
 * answer). Throws if the reply isn't the expected JSON shape at all.
 */
internal fun parseClassifierResponse(content: String, expectedCount: Int): List<String?> {
    val out = MutableList<String?>(expectedCount) { null }
    val results = lenientJson.parseToJsonElement(content).jsonObject["results"]?.jsonArray
        ?: throw IllegalArgumentException("missing \"results\" in classifier reply")
    for (entry in results) {
        val obj = entry.jsonObject
        val i = obj["i"]?.jsonPrimitive?.intOrNull ?: continue
        if (i !in 0 until expectedCount) continue
        out[i] = parseHomeCategory(obj["category"]?.jsonPrimitive?.contentOrNull)
    }
    return out
}

private suspend fun requestBatch(names: List<String>, apiKey: String): List<String?> {
    val response = withContext(Dispatchers.IO) {
        groqApi.getCompletion(
            "Bearer $apiKey",
            GroqRequest(
                model = CLASSIFIER_MODEL,
                messages = listOf(
                    GroqMessage("system", CLASSIFIER_SYSTEM_PROMPT),
                    GroqMessage("user", buildClassifierUserMessage(names))
                ),
                response_format = GroqResponseFormat()
            )
        )
    }
    val content = response.choices.firstOrNull()?.message?.content.orEmpty()
    return parseClassifierResponse(content, names.size)
}

/**
 * Classify product names into home categories with the LLM, applying manual overrides on top and
 * falling back to keywords for anything the LLM can't answer. Never throws.
 */
suspend fun classifyHomeCategories(names: List<String>, apiKey: String): ClassificationResult {
    if (names.isEmpty()) return ClassificationResult(emptyList())
    val (llmAnswers, error) = try {
        names.chunked(BATCH_SIZE).flatMap { requestBatch(it, apiKey) } to null
    } catch (e: Exception) {
        Log.e("CategoryLLM", "Classification failed, using keyword fallback", e)
        List<String?>(names.size) { null } to (e.localizedMessage ?: e.javaClass.simpleName)
    }
    val categories = names.mapIndexed { i, name -> resolveHomeCategory(name, llmAnswers[i]) }
    return ClassificationResult(categories, error)
}
