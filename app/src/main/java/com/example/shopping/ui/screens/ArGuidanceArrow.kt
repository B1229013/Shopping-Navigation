package com.example.shopping.ui.screens

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Navigation
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlin.math.abs

/** How close (degrees) the user must be to facing the target before the arrow
 *  turns green and reads "朝這個方向前進". */
private const val ALIGNED_THRESHOLD_DEG = 22f

private fun normalizeSigned(deg: Float): Float {
    var x = ((deg % 360f) + 360f) % 360f
    if (x > 180f) x -= 360f
    return x
}

/**
 * AR guidance arrow overlaid on the live camera view.
 *
 * [directionDeg] is the egocentric bearing to the next waypoint at the moment
 * the guidance was issued (0 = straight ahead, clockwise positive). As the user
 * physically rotates the phone, the rotation-vector sensor keeps the arrow
 * pointing at the real-world target, so once they turn to face it the arrow
 * snaps to "up" and goes green — an unambiguous "you are heading the right way"
 * confirmation that plain text guidance cannot give.
 */
@Composable
fun ArGuidanceArrow(
    directionDeg: Float,
    directionLabel: String?,
    modifier: Modifier = Modifier,
) {
    val azimuth by rememberDeviceAzimuth()

    // Azimuth at the instant this direction was issued. Re-captured whenever the
    // backend sends a new direction (keyed on directionDeg).
    var baseAzimuth by remember { mutableFloatStateOf(Float.NaN) }
    LaunchedEffect(directionDeg) { baseAzimuth = azimuth }

    // Subtract how far the user has turned since the direction was issued.
    val corrected = if (baseAzimuth.isNaN()) directionDeg
        else directionDeg - (azimuth - baseAzimuth)
    val signed = normalizeSigned(corrected)
    val aligned = abs(signed) <= ALIGNED_THRESHOLD_DEG

    // Gentle pulse so the arrow reads as "live" guidance.
    val pulse = rememberInfiniteTransition(label = "arrowPulse")
    val scale by pulse.animateFloat(
        initialValue = 0.94f,
        targetValue = 1.06f,
        animationSpec = infiniteRepeatable(
            animation = tween(900),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "arrowScale",
    )

    val accent = if (aligned) Color(0xFF34D399) else Color(0xFFFFC857)

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Box(
            modifier = Modifier
                .size(132.dp)
                .graphicsLayer {
                    scaleX = scale
                    scaleY = scale
                }
                .clip(CircleShape)
                .background(Color.Black.copy(alpha = 0.32f)),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = Icons.Default.Navigation,
                contentDescription = directionLabel ?: "導航方向",
                tint = accent,
                modifier = Modifier
                    .size(84.dp)
                    .graphicsLayer { rotationZ = signed },
            )
        }

        if (!directionLabel.isNullOrBlank()) {
            Spacer(Modifier.height(12.dp))
            Surface(
                color = Color.Black.copy(alpha = 0.55f),
                shape = RoundedCornerShape(20.dp),
                border = BorderStroke(1.dp, accent.copy(alpha = 0.7f)),
            ) {
                Text(
                    text = if (aligned) "朝這個方向前進" else directionLabel,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = accent,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                )
            }
        }
    }
}
