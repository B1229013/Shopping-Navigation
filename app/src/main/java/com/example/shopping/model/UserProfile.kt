package com.example.shopping.model

import kotlinx.serialization.Serializable

@Serializable
data class UserProfile(
    val name: String = "",
    val gender: String = "其他",
    val birthday: String = "2000-01-01",
    val height: String = "",
    val weight: String = "",
    val allergies: String = "",
    val disease: String = "無",
    val activityLevel: String = "稍低"
)
