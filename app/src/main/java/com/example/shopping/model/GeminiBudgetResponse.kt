package com.example.shopping.model

import kotlinx.serialization.Serializable

@Serializable
data class GeminiBudgetResponse(
    val transaction_metadata: TransactionMetadata,
    val line_items: List<GeminiLineItem>,
    val budget_insight: BudgetInsight
)

@Serializable
data class TransactionMetadata(
    val store_name: String,
    val category: String,
    val timestamp: String,
    val total_amount: Double,
    val currency: String,
    val input_type: String
)

@Serializable
data class GeminiLineItem(
    val item_name: String,
    val price: Double,
    val quantity: Int
)

@Serializable
data class BudgetInsight(
    val summary_text: String,
    val is_unusual_spike: Boolean,
    val confidence_level: Double
)
