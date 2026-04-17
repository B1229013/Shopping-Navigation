package com.example.shopping.model

import kotlinx.serialization.Serializable
import java.util.*

@Serializable
data class ShoppingItem(
    val id: String = UUID.randomUUID().toString(),
    var name: String,
    var qty: Int = 1,
    var price: Int = 0,
    var isChecked: Boolean = false,
    val createdAt: Long = System.currentTimeMillis(),
    var purchasedAt: Long? = null,
    val dueDate: Long? = null,
    var storeName: String? = null,
    var location: String? = null,
    val receiptId: String? = null
)

@Serializable
data class ShoppingList(
    val id: String = UUID.randomUUID().toString(),
    val date: String, // yyyy-MM-dd
    val items: List<ShoppingItem> = emptyList(),
    val status: String = "進行中" // 進行中, 已完成
)

@Serializable
data class UserProfile(
    val name: String = "",
    val gender: String = "男",
    val age: Int = 0,
    val height: Double = 0.0,
    val weight: Double = 0.0,
    val activityLevel: String = "適度",
    val allergies: String = "" // 儲存過敏原文字，如 "牛奶, 花生"
) {
    val bmi: Double
        get() = if (height > 0) weight / ((height / 100.0) * (height / 100.0)) else 0.0
    
    val recommendedCalories: Int
        get() = when(gender) {
            "男" -> if (age >= 71) (if(activityLevel == "適度") 2150 else 1650) else 2000
            else -> if (age >= 71) (if(activityLevel == "適度") 1700 else 1300) else 1500
        }
}

@Serializable
data class DietRecord(
    val id: String = UUID.randomUUID().toString(),
    val date: String,
    val name: String,
    val ingredients: String = "",
    val expiryDate: String = "",
    val unitCalorie: Int = 0,
    val portion: Double = 1.0,
    val totalCalories: Int = 0,
    val carbs: Double = 0.0,
    val protein: Double = 0.0,
    val fat: Double = 0.0,
    val cholesterol: Double = 0.0,
    val foodCategory: String = "未分類"
)
