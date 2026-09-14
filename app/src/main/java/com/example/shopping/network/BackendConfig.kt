package com.example.shopping.network

import android.content.Context
import com.example.shopping.BuildConfig

object BackendConfig {
    private const val PREFS_NAME = "backend_config"
    private const val KEY_URL = "backend_url"

    private var _currentUrl: String? = null

    val currentUrl: String
        get() = _currentUrl ?: BuildConfig.BACKEND_URL.ifBlank { "http://10.0.2.2:8000/" }

    fun init(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        _currentUrl = prefs.getString(KEY_URL, null)
    }

    fun setUrl(context: Context, url: String) {
        val normalized = if (url.endsWith("/")) url else "$url/"
        _currentUrl = normalized
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_URL, normalized).apply()
    }

    fun getSavedUrl(context: Context): String? {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_URL, null)
    }

    fun clearSavedUrl(context: Context) {
        _currentUrl = null
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().remove(KEY_URL).apply()
    }
}
