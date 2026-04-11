package com.example.shopping.ui.theme

import android.app.Activity
import android.os.Build
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

// ── Cinematic Color Scheme ──────────────────────────────────
// Always dark. The interface is a theater — content is the performance.

private val CinemaColorScheme = darkColorScheme(
    primary = Gold,
    onPrimary = Noir,
    primaryContainer = GoldDim,
    onPrimaryContainer = GoldBright,

    secondary = TextSecondary,
    onSecondary = Noir,
    secondaryContainer = SurfaceBright,
    onSecondaryContainer = TextPrimary,

    tertiary = Info,
    onTertiary = Noir,
    tertiaryContainer = InfoDim,
    onTertiaryContainer = Info,

    error = Danger,
    onError = TextPrimary,
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
    inversePrimary = GoldDim
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
            window.navigationBarColor = Noir.toArgb()
            WindowCompat.getInsetsController(window, view).apply {
                isAppearanceLightStatusBars = false
                isAppearanceLightNavigationBars = false
            }
        }
    }

    MaterialTheme(
        colorScheme = CinemaColorScheme,
        typography = CinemaTypography,
        content = content
    )
}
