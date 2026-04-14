package com.example.shopping.ui.components

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.shopping.ui.theme.*
import kotlinx.coroutines.delay

// ── Staggered Entrance ──────────────────────────────────────
// Each child fades + slides in with a delay, creating a cinematic reveal.

@Composable
fun StaggeredItem(
    index: Int,
    delayPerItem: Long = 50L,
    baseDelay: Long = 80L,
    content: @Composable () -> Unit
) {
    var visible by remember { mutableStateOf(false) }
    val alpha by animateFloatAsState(
        targetValue = if (visible) 1f else 0f,
        animationSpec = tween(
            durationMillis = 450,
            easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)
        ),
        label = "stagger_alpha"
    )
    val offsetY by animateFloatAsState(
        targetValue = if (visible) 0f else 20f,
        animationSpec = tween(
            durationMillis = 450,
            easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)
        ),
        label = "stagger_offset"
    )

    LaunchedEffect(Unit) {
        delay(baseDelay + index * delayPerItem)
        visible = true
    }

    Box(
        modifier = Modifier.graphicsLayer {
            this.alpha = alpha
            translationY = offsetY * density
        }
    ) {
        content()
    }
}

// ── Press-Feedback Surface with Glow ───────────────────────
// Scales down on press, lifts on hover — tactile and alive.

@Composable
fun PressableSurface(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    shape: RoundedCornerShape = RoundedCornerShape(16.dp),
    backgroundColor: Color = SurfaceBase,
    glowColor: Color = Gold.copy(alpha = 0.06f),
    content: @Composable BoxScope.() -> Unit
) {
    var isPressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.97f else 1f,
        animationSpec = tween(
            durationMillis = if (isPressed) 80 else 250,
            easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)
        ),
        label = "press_scale"
    )
    val elevation by animateFloatAsState(
        targetValue = if (isPressed) 0f else 2f,
        animationSpec = tween(150),
        label = "press_elevation"
    )

    Box(
        modifier = modifier
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
                shadowElevation = elevation
            }
            .clip(shape)
            .background(backgroundColor, shape)
            .drawBehind {
                // Subtle top-edge glow
                drawRoundRect(
                    brush = Brush.verticalGradient(
                        colors = listOf(glowColor, Color.Transparent),
                        startY = 0f,
                        endY = size.height * 0.3f
                    ),
                    cornerRadius = CornerRadius(16.dp.toPx())
                )
            }
            .pointerInput(Unit) {
                detectTapGestures(
                    onPress = {
                        isPressed = true
                        try {
                            awaitRelease()
                        } finally {
                            isPressed = false
                        }
                        onClick()
                    }
                )
            },
        content = content
    )
}

// ── Cinema Card ─────────────────────────────────────────────
// A card with subtle border glow — depth comes from color and light.

@Composable
fun CinemaCard(
    modifier: Modifier = Modifier,
    backgroundColor: Color = SurfaceBase,
    borderColor: Color = Border,
    content: @Composable ColumnScope.() -> Unit
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(20.dp))
            .background(backgroundColor)
            .background(
                brush = Brush.verticalGradient(
                    colors = listOf(borderColor.copy(alpha = 0.5f), Color.Transparent),
                    startY = 0f,
                    endY = 3f
                )
            )
            .padding(1.dp)
            .clip(RoundedCornerShape(19.dp))
            .background(backgroundColor)
            .padding(20.dp),
        content = content
    )
}

// ── Fade-In Screen Wrapper ──────────────────────────────────
// Wraps an entire screen with a fade-in on first composition.

@Composable
fun FadeInScreen(
    content: @Composable () -> Unit
) {
    var visible by remember { mutableStateOf(false) }
    val alpha by animateFloatAsState(
        targetValue = if (visible) 1f else 0f,
        animationSpec = tween(400, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "screen_fade"
    )

    LaunchedEffect(Unit) { visible = true }

    Box(modifier = Modifier.graphicsLayer { this.alpha = alpha }) {
        content()
    }
}

// ── Animated Number ─────────────────────────────────────────
// Smoothly interpolates between numeric values for dashboard displays.

@Composable
fun animateIntAsState(targetValue: Int): State<Int> {
    val animatable = remember { Animatable(targetValue.toFloat()) }
    LaunchedEffect(targetValue) {
        animatable.animateTo(
            targetValue.toFloat(),
            animationSpec = tween(700, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f))
        )
    }
    return derivedStateOf { animatable.value.toInt() }
}

// ── Cinematic Divider ───────────────────────────────────────
// Ultra-subtle horizontal rule.

@Composable
fun CinemaDivider(modifier: Modifier = Modifier) {
    Spacer(
        modifier = modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(Border)
    )
}

// ── Shimmer Loading ─────────────────────────────────────────
// Elegant loading indicator that pulses with green.

@Composable
fun CinemaShimmer(
    modifier: Modifier = Modifier,
    height: Dp = 16.dp
) {
    val infiniteTransition = rememberInfiniteTransition(label = "shimmer")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.10f,
        targetValue = 0.30f,
        animationSpec = infiniteRepeatable(
            animation = tween(900, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "shimmer_alpha"
    )

    Spacer(
        modifier = modifier
            .fillMaxWidth()
            .height(height)
            .clip(RoundedCornerShape(8.dp))
            .background(Gold.copy(alpha = alpha))
    )
}

// ── Green Gradient CTA ─────────────────────────────────────
// A fresh gradient that feels premium and alive.

val GoldGradient = Brush.horizontalGradient(
    colors = listOf(GoldDim, Gold, GoldBright)
)
