package com.example.shopping.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ── Fresh Grocery Light Scheme ─────────────────────────────
// Light, warm, organic. Matches the grocery homepage aesthetic.

private val GroceryColorScheme = lightColorScheme(
    primary = Gold,
    onPrimary = SurfaceBase,
    primaryContainer = GoldSurface,
    onPrimaryContainer = GoldDim,

    secondary = TextSecondary,
    onSecondary = SurfaceBase,
    secondaryContainer = SurfaceBright,
    onSecondaryContainer = TextPrimary,

    tertiary = Info,
    onTertiary = SurfaceBase,
    tertiaryContainer = InfoDim,
    onTertiaryContainer = Info,

    error = Danger,
    onError = SurfaceBase,
    errorContainer = DangerDim,
    onErrorContainer = Danger,

    background = Noir,
    onBackground = TextPrimary,

    surface = SurfaceDim,
    onSurface = TextPrimary,
    surfaceVariant = SurfaceBase,
    onSurfaceVariant = TextSecondary,

    outline = Border,
    outlineVariant = BorderBright,

    inverseSurface = TextPrimary,
    inverseOnSurface = Noir,
    inversePrimary = GoldBright
)

@Composable
fun ShoppingTheme(
    content: @Composable () -> Unit
) {
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            @Suppress("DEPRECATION")
            window.statusBarColor = Noir.toArgb()
            @Suppress("DEPRECATION")
            window.navigationBarColor = SurfaceBase.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = true
                isAppearanceLightNavigationBars = true
            }
        }
    }

    MaterialTheme(
        colorScheme = GroceryColorScheme,
        typography = CinemaTypography,
        content = content
    )
}
