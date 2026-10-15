package com.example.shopping.ui.screens

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Directions
import androidx.compose.material.icons.filled.FlashOff
import androidx.compose.material.icons.filled.FlashOn
import androidx.compose.material.icons.filled.NorthEast
import androidx.compose.material.icons.filled.PanTool
import androidx.compose.material.icons.filled.PersonOff
import androidx.compose.material.icons.filled.PhoneAndroid
import androidx.compose.material.icons.filled.QuestionMark
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.shopping.network.PathResponse
import com.example.shopping.viewmodel.GoalItem
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

// Navigation-screen panels, matching the iOS NavigationCameraView
// (APPNAV_ios/Views/Navigation/NavigationCameraView.swift, RouteMapView.swift).

private val PanelText = Color.White

/** Phase chip label, as on iOS. */
private fun phaseLabel(phase: String) = when (phase) {
    "checkout" -> "前往結帳"
    "exit" -> "前往出口"
    "done" -> "導航完成"
    else -> "找商品中"
}

/** Status card: title + phase chip, current target, location; flash and "?" buttons. */
@Composable
fun NavStatusCard(
    title: String,
    phase: String,
    goal: String,
    currentTarget: String?,
    currentLocation: String?,
    torchOn: Boolean,
    torchAvailable: Boolean,
    onToggleTorch: () -> Unit,
    onShowPhotoGuide: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val background = when (phase) {
        "checkout" -> Color(0xFFFF9800).copy(alpha = 0.45f)
        "exit" -> Color(0xFF9C27B0).copy(alpha = 0.45f)
        else -> Color(0xFF2196F3).copy(alpha = 0.55f)
    }
    val chip = when (phase) {
        "checkout" -> Color(0xFFFF9800).copy(alpha = 0.7f)
        "exit" -> Color(0xFF9C27B0).copy(alpha = 0.7f)
        "done" -> Color(0xFF4CAF50).copy(alpha = 0.7f)
        else -> Color(0xFF2196F3).copy(alpha = 0.5f)
    }
    Row(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(background)
            .padding(14.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(title, color = PanelText, fontWeight = FontWeight.Bold, fontSize = 17.sp)
                Spacer(Modifier.width(6.dp))
                Text(
                    phaseLabel(phase),
                    color = PanelText,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier
                        .clip(CircleShape)
                        .background(chip)
                        .padding(horizontal = 8.dp, vertical = 2.dp),
                )
            }
            Text(
                when {
                    phase == "checkout" -> "前往收銀台"
                    phase == "exit" -> "前往出口"
                    currentTarget != null -> "正在找　$currentTarget"
                    else -> "目標　$goal"
                },
                color = PanelText.copy(alpha = 0.85f),
                fontSize = 14.sp,
                maxLines = 2,
            )
            currentLocation?.let {
                Text("📍 $it", color = PanelText.copy(alpha = 0.7f), fontSize = 12.sp, maxLines = 2)
            }
        }
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            RoundIconButton(
                icon = if (torchOn) Icons.Default.FlashOn else Icons.Default.FlashOff,
                description = if (torchOn) "關閉閃光燈" else "開啟閃光燈",
                background = if (torchOn) Color(0xFFFFEB3B).copy(alpha = 0.85f) else Color.White.copy(alpha = 0.2f),
                enabled = torchAvailable,
                onClick = onToggleTorch,
            )
            RoundIconButton(
                icon = Icons.Default.QuestionMark,
                description = "拍照說明",
                background = Color.White.copy(alpha = 0.2f),
                onClick = onShowPhotoGuide,
            )
        }
    }
}

@Composable
private fun RoundIconButton(
    icon: ImageVector,
    description: String,
    background: Color,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    // A plain Box, not IconButton: IconButton's 48dp minimum touch target made the two
    // 34dp circles overlap.
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(34.dp)
            .clip(CircleShape)
            .background(if (enabled) background else background.copy(alpha = 0.1f))
            .clickable(enabled = enabled, role = Role.Button, onClickLabel = description, onClick = onClick),
    ) {
        Icon(icon, contentDescription = description, tint = PanelText.copy(alpha = if (enabled) 1f else 0.35f),
            modifier = Modifier.size(18.dp))
    }
}

/** Shopping-list chips: done = green + strikethrough, current = yellow, others dim. */
@Composable
fun ShoppingChecklist(items: List<GoalItem>, currentTarget: String?, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        items.forEach { item ->
            val isCurrent = item.name == currentTarget && !item.visited
            val tint = when {
                item.visited -> Color(0xFF4CAF50)
                isCurrent -> Color(0xFFFFEB3B)
                else -> Color.White.copy(alpha = 0.5f)
            }
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .clip(CircleShape)
                    .background(
                        when {
                            isCurrent -> Color(0xFFFFEB3B).copy(alpha = 0.15f)
                            item.visited -> Color(0xFF4CAF50).copy(alpha = 0.1f)
                            else -> Color.White.copy(alpha = 0.05f)
                        }
                    )
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            ) {
                Icon(
                    if (item.visited) Icons.Default.CheckCircle else Icons.Default.RadioButtonUnchecked,
                    contentDescription = null,
                    tint = tint,
                    modifier = Modifier.size(12.dp),
                )
                Spacer(Modifier.width(4.dp))
                Text(
                    item.name,
                    fontSize = 11.sp,
                    fontWeight = if (isCurrent) FontWeight.Bold else FontWeight.Normal,
                    color = when {
                        item.visited -> Color(0xFF4CAF50).copy(alpha = 0.7f)
                        isCurrent -> Color(0xFFFFEB3B)
                        else -> Color.White.copy(alpha = 0.6f)
                    },
                    textDecoration = if (item.visited) TextDecoration.LineThrough else null,
                )
            }
        }
    }
}

/** Pill-shaped hint above the route panel (retake / second photo / hold upright). */
@Composable
fun NavHintBanner(text: String, background: Color, modifier: Modifier = Modifier) {
    Text(
        text,
        color = PanelText,
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(background)
            .padding(horizontal = 12.dp, vertical = 8.dp),
    )
}

/**
 * Route panel: the first step + distance, and the mini-map below. Without PDR the user
 * dot sits at the route start (the last photo fix); the heading arrow is drawn only
 * when the server's heading is trustworthy.
 */
@Composable
fun RoutePanel(
    route: PathResponse?,
    prompt: String?,
    statusMessage: String?,
    userHeadingDeg: Double?,
    modifier: Modifier = Modifier,
) {
    if (route != null) {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(14.dp))
                .background(Color(0xFF00BCD4).copy(alpha = 0.18f))
                .padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    "🧭 ${prompt ?: "定位中…"}",
                    color = PanelText,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                Text("剩 ${route.distance_m.toInt()} m", color = PanelText.copy(alpha = 0.8f), fontSize = 12.sp)
            }
            RouteMap(
                route = route,
                userX = route.start.x,
                userY = route.start.y,
                userHeadingDeg = userHeadingDeg,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(150.dp),
            )
        }
    } else if (statusMessage != null) {
        Text(
            "🗺 $statusMessage",
            color = PanelText.copy(alpha = 0.8f),
            fontSize = 12.sp,
            modifier = modifier.fillMaxWidth(),
        )
    }
}

/**
 * Top-down store map: walkable edges and waypoints, the planned route (cyan), the target
 * (orange) and the user (green, with heading). Map metres → screen, y flipped
 * (map +y = heading 0° = screen up).
 */
@Composable
fun RouteMap(
    route: PathResponse,
    userX: Double?,
    userY: Double?,
    userHeadingDeg: Double?,
    modifier: Modifier = Modifier,
) {
    Canvas(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color.Black.copy(alpha = 0.55f)),
    ) {
        val pts = route.nodes
        if (pts.isEmpty()) return@Canvas
        val minX = pts.minOf { it.x }; val maxX = pts.maxOf { it.x }
        val minY = pts.minOf { it.y }; val maxY = pts.maxOf { it.y }
        val pad = 12.dp.toPx()
        val spanX = maxOf(maxX - minX, 1.0).toFloat()
        val spanY = maxOf(maxY - minY, 1.0).toFloat()
        val scale = minOf((size.width - 2 * pad) / spanX, (size.height - 2 * pad) / spanY)
        // centre the map in whichever direction has room left over
        val offX = (size.width - spanX * scale) / 2
        val offY = (size.height - spanY * scale) / 2
        fun toScreen(x: Double, y: Double) = Offset(
            offX + ((x - minX) * scale).toFloat(),
            size.height - offY - ((y - minY) * scale).toFloat(),
        )

        // walkable edges + waypoints
        val byId = pts.associateBy { it.id }
        route.edges.forEach { e ->
            val a = byId[e.from] ?: return@forEach
            val b = byId[e.to] ?: return@forEach
            drawLine(Color.White.copy(alpha = 0.25f), toScreen(a.x, a.y), toScreen(b.x, b.y), strokeWidth = 1.dp.toPx())
        }
        pts.forEach { drawCircle(Color.White.copy(alpha = 0.5f), radius = 1.5.dp.toPx(), center = toScreen(it.x, it.y)) }

        // planned route
        if (route.polyline.size >= 2) {
            val line = Path()
            route.polyline.forEachIndexed { i, v ->
                val s = toScreen(v[0], v[1])
                if (i == 0) line.moveTo(s.x, s.y) else line.lineTo(s.x, s.y)
            }
            drawPath(line, Color(0xFF00BCD4),
                style = Stroke(width = 3.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
        }

        // target
        val t = toScreen(route.target.x, route.target.y)
        drawCircle(Color(0xFFFF9800), radius = 6.dp.toPx(), center = t)
        drawCircle(Color.White, radius = 6.dp.toPx(), center = t, style = Stroke(width = 1.5.dp.toPx()))

        // user
        if (userX != null && userY != null) {
            val u = toScreen(userX, userY)
            val green = Color(0xFF4CAF50)
            drawCircle(green, radius = 6.dp.toPx(), center = u)
            drawCircle(Color.White, radius = 6.dp.toPx(), center = u, style = Stroke(width = 1.5.dp.toPx()))
            userHeadingDeg?.let { h ->
                val rad = Math.toRadians(h)
                val len = 14.dp.toPx()
                val tip = Offset(u.x + len * sin(rad).toFloat(), u.y - len * cos(rad).toFloat())
                drawLine(green, u, tip, strokeWidth = 2.5.dp.toPx(), cap = StrokeCap.Round)
            }
        }
    }
}

private data class PhotoTip(val icon: ImageVector, val title: String, val detail: String)

private val PHOTO_TIPS = listOf(
    PhotoTip(Icons.Default.PhoneAndroid, "手機拿直", "直立拿在胸前高度，不要朝地板或天花板拍。畫面歪斜時下方會提示。"),
    PhotoTip(Icons.Default.NorthEast, "對準走道前方", "沿著走道往前拍，拍到走道的延伸，而不是貼著貨架拍單一商品。"),
    PhotoTip(Icons.Default.Directions, "拍到招牌或吊牌", "走道編號、天花板吊牌、區域招牌是最好的定位線索。"),
    PhotoTip(Icons.Default.PersonOff, "避開人和推車", "前方有人或推車擋住時，等一下或換個角度；被擋住的照片不會用來校正位置。"),
    PhotoTip(Icons.Default.PanTool, "停一下再按", "走路中拍容易模糊；停一秒、拿穩再按拍照。太暗時開閃光燈。"),
)

/** How to take a navigation photo — shown on first use and from the "?" button. */
@Composable
fun PhotoGuideDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("怎麼拍定位照片") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
                PHOTO_TIPS.forEach { tip ->
                    Row(verticalAlignment = Alignment.Top) {
                        Icon(tip.icon, contentDescription = null, tint = Color(0xFF2196F3),
                            modifier = Modifier.size(24.dp))
                        Spacer(Modifier.width(14.dp))
                        Column {
                            Text(tip.title, fontWeight = FontWeight.Bold, fontSize = 15.sp)
                            Text(tip.detail, fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("知道了") } },
    )
}

private const val NAV_PREFS = "nav_prefs"
private const val KEY_PHOTO_GUIDE_SEEN = "navPhotoGuideSeen"

/** True the first time only; marks the guide as seen right away (however it is dismissed). */
fun consumeFirstPhotoGuide(context: Context): Boolean {
    val prefs = context.getSharedPreferences(NAV_PREFS, Context.MODE_PRIVATE)
    if (prefs.getBoolean(KEY_PHOTO_GUIDE_SEEN, false)) return false
    prefs.edit().putBoolean(KEY_PHOTO_GUIDE_SEEN, true).apply()
    return true
}

/** Beyond this tilt from upright the reference photos (taken upright) stop matching well. */
const val MAX_TILT_DEGREES = 25.0

/**
 * Whether the phone is held upright (portrait, facing ahead): angle between the phone's
 * long axis and vertical, from the gravity sensor (accelerometer as fallback).
 */
@Composable
fun rememberPhoneUpright(): State<Boolean> {
    val context = LocalContext.current
    val upright = remember { mutableStateOf(true) }
    DisposableEffect(Unit) {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val sensor = sm?.getDefaultSensor(Sensor.TYPE_GRAVITY) ?: sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        if (sm == null || sensor == null) return@DisposableEffect onDispose { }
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val (x, y, z) = Triple(event.values[0], event.values[1], event.values[2])
                val g = sqrt(x * x + y * y + z * z)
                if (g < 0.1f) return
                // Android reports gravity as +y when the phone stands upright
                val tilt = Math.toDegrees(acos((y / g).coerceIn(-1f, 1f).toDouble()))
                upright.value = tilt <= MAX_TILT_DEGREES
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        sm.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_UI)
        onDispose { sm.unregisterListener(listener) }
    }
    return upright
}
