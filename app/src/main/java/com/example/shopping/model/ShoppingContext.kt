package com.example.shopping.model

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * ShoppingContext — Shared data schema for cross-module AI integration.
 *
 * This is the "Context Provider" that aggregates data from all three modules
 * (List Management, Budget & Health, AI Core) into a single snapshot the AI
 * can query. It enables bi-directional data flow:
 *
 *   ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
 *   │    List      │────▶│                  │◀────│   Budget &   │
 *   │  Management  │     │  ShoppingContext  │     │   Health     │
 *   └─────────────┘     │  (Context Provider)│     └──────────────┘
 *                        │                  │
 *                        └────────┬─────────┘
 *                                 │
 *                                 ▼
 *                        ┌──────────────────┐
 *                        │   AI Core        │
 *                        │  (LLaMA/Groq)    │
 *                        └──────────────────┘
 *
 * The AI receives a JSON snapshot of this context with every query,
 * enabling cross-referencing like:
 *   "How much did I spend on snacks?" → reads budget + item categories
 *   "Does my list exceed sodium goals?" → reads list + health targets
 *   "Plan a meal within budget" → reads inventory + budget + health
 */
@Serializable
data class ShoppingContext(
    val inventory: InventorySnapshot,
    val budget: BudgetSnapshot,
    val health: HealthSnapshot,
    val summary: ContextSummary
)

// ── Module 1: List Management ──────────────────────────────

@Serializable
data class InventorySnapshot(
    val pending_items: List<ItemEntry>,
    val purchased_items: List<ItemEntry>,
    val total_pending: Int,
    val total_purchased: Int
)

@Serializable
data class ItemEntry(
    val name: String,
    val qty: Int,
    val price: Int,
    val category: String?,
    val is_purchased: Boolean
)

// ── Module 2: Budget & Health ──────────────────────────────

@Serializable
data class BudgetSnapshot(
    val monthly_budget: Int,
    val total_spent: Int,
    val remaining: Int,
    val spending_by_category: Map<String, Int>
)

@Serializable
data class HealthSnapshot(
    val diet_records_today: List<DietEntry>,
    val total_calories_today: Int,
    val recent_ingredients: List<String>,
    val food_categories_consumed: Map<String, Int>
)

@Serializable
data class DietEntry(
    val name: String,
    val calories: Int,
    val carbs: Double,
    val protein: Double,
    val fat: Double,
    val category: String
)

// ── Cross-module summary ───────────────────────────────────

@Serializable
data class ContextSummary(
    val date: String,
    val budget_utilization_percent: Int,
    val top_spending_category: String?,
    val pending_item_names: List<String>
)

// ── Builder: constructs context from raw module data ───────

private val contextJson = Json {
    ignoreUnknownKeys = true
    prettyPrint = false
    encodeDefaults = true
}

fun buildShoppingContext(
    shoppingItems: List<ShoppingItem>,
    dietRecords: List<DietRecord>,
    budgetTotal: Int
): ShoppingContext {
    val today = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault())
        .format(java.util.Date())

    // ── Inventory ──
    val pending = shoppingItems.filter { !it.isChecked }
    val purchased = shoppingItems.filter { it.isChecked }

    val inventorySnapshot = InventorySnapshot(
        pending_items = pending.map { ItemEntry(it.name, it.qty, it.price, it.location, false) },
        purchased_items = purchased.map { ItemEntry(it.name, it.qty, it.price, it.location, true) },
        total_pending = pending.size,
        total_purchased = purchased.size
    )

    // ── Budget ──
    val totalSpent = purchased.sumOf { it.price * it.qty }
    val spendingByCategory = purchased
        .groupBy { it.location ?: "其他" }
        .mapValues { (_, items) -> items.sumOf { it.price * it.qty } }

    val budgetSnapshot = BudgetSnapshot(
        monthly_budget = budgetTotal,
        total_spent = totalSpent,
        remaining = budgetTotal - totalSpent,
        spending_by_category = spendingByCategory
    )

    // ── Health ──
    val todayRecords = dietRecords.filter { it.date == today }
    val recentRecords = dietRecords.takeLast(10)

    val healthSnapshot = HealthSnapshot(
        diet_records_today = todayRecords.map {
            DietEntry(it.name, it.totalCalories, it.carbs, it.protein, it.fat, it.foodCategory)
        },
        total_calories_today = todayRecords.sumOf { it.totalCalories },
        recent_ingredients = recentRecords.map { it.ingredients }.filter { it.isNotBlank() },
        food_categories_consumed = todayRecords
            .groupBy { it.foodCategory }
            .mapValues { (_, records) -> records.size }
    )

    // ── Summary ──
    val topCategory = spendingByCategory.maxByOrNull { it.value }?.key

    val summary = ContextSummary(
        date = today,
        budget_utilization_percent = if (budgetTotal > 0) (totalSpent * 100 / budgetTotal) else 0,
        top_spending_category = topCategory,
        pending_item_names = pending.map { it.name }
    )

    return ShoppingContext(
        inventory = inventorySnapshot,
        budget = budgetSnapshot,
        health = healthSnapshot,
        summary = summary
    )
}

/**
 * Serializes the context to a compact JSON string for the AI system prompt.
 */
fun ShoppingContext.toJsonString(): String {
    return contextJson.encodeToString(this)
}
