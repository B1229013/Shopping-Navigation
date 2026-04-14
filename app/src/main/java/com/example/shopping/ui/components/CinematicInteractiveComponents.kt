package com.example.shopping.ui.components

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.shopping.ui.theme.*

// ── Nutritional Ring ────────────────────────────────────────
// Circular progress indicator for macros, matching reference design.

@Composable
fun NutrientRing(
    label: String,
    value: Int,
    maxValue: Int,
    unit: String = "g",
    color: Color,
    size: Dp = 72.dp,
    strokeWidth: Dp = 5.dp
) {
    val progress = if (maxValue > 0) (value.toFloat() / maxValue).coerceIn(0f, 1f) else 0f

    var animationPlayed by remember { mutableStateOf(false) }
    val animatedProgress by animateFloatAsState(
        targetValue = if (animationPlayed) progress else 0f,
        animationSpec = tween(800, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "ring_progress"
    )
    LaunchedEffect(Unit) { animationPlayed = true }

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier.size(size),
            contentAlignment = Alignment.Center
        ) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val stroke = strokeWidth.toPx()
                val arcSize = Size(this.size.width - stroke, this.size.height - stroke)
                val topLeft = Offset(stroke / 2, stroke / 2)

                // Track
                drawArc(
                    color = color.copy(alpha = 0.12f),
                    startAngle = -90f,
                    sweepAngle = 360f,
                    useCenter = false,
                    topLeft = topLeft,
                    size = arcSize,
                    style = Stroke(width = stroke, cap = StrokeCap.Round)
                )
                // Fill
                drawArc(
                    color = color,
                    startAngle = -90f,
                    sweepAngle = 360f * animatedProgress,
                    useCenter = false,
                    topLeft = topLeft,
                    size = arcSize,
                    style = Stroke(width = stroke, cap = StrokeCap.Round)
                )
            }
            Text(
                "$value",
                style = MaterialTheme.typography.titleMedium,
                color = TextPrimary,
                fontWeight = FontWeight.Bold
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = TextSecondary
        )
    }
}

// ── Pulsing Glow Dot ────────────────────────────────────────
// Active indicator with gentle pulse animation.

@Composable
fun PulsingDot(
    color: Color = Gold,
    size: Dp = 6.dp
) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 0.85f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "dot_pulse"
    )
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.7f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "dot_alpha"
    )

    Box(
        modifier = Modifier
            .size(size)
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
                this.alpha = alpha
            }
            .background(color, CircleShape)
    )
}

// ── Ambient Glow Background ─────────────────────────────────
// A subtle radial glow for hero sections.

@Composable
fun AmbientGlow(
    modifier: Modifier = Modifier,
    color: Color = Gold.copy(alpha = 0.04f),
    height: Dp = 300.dp
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(height)
            .drawBehind {
                drawCircle(
                    brush = Brush.radialGradient(
                        colors = listOf(color, Color.Transparent),
                        center = Offset(size.width / 2, 0f),
                        radius = size.height
                    )
                )
            }
    )
}

// ── Section Header ──────────────────────────────────────────
// Consistent section title style across screens.

@Composable
fun SectionHeader(
    title: String,
    modifier: Modifier = Modifier,
    trailing: @Composable (() -> Unit)? = null
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            title,
            style = MaterialTheme.typography.headlineMedium,
            color = TextPrimary,
            fontWeight = FontWeight.SemiBold
        )
        trailing?.invoke()
    }
}
