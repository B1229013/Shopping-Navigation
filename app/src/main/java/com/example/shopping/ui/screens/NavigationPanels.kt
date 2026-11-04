package com.example.shopping.ui.screens

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.shopping.navigation.Vec2
import coil.compose.AsyncImage
import com.example.shopping.network.PathResponse
import com.example.shopping.network.StartPoint
import com.example.shopping.viewmodel.GoalItem
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.roundToInt
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

/** Last photo taken or picked, left of the status card (iOS lastPhotoThumbnail: 56 pt, no tap). */
@Composable
fun LastPhotoThumbnail(photo: ImageBitmap, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(10.dp)
    Image(
        bitmap = photo,
        contentDescription = "上一張照片",
        contentScale = ContentScale.Crop,
        modifier = modifier
            .size(56.dp)
            .shadow(3.dp, shape)
            .clip(shape)
            .border(1.dp, Color.White.copy(alpha = 0.5f), shape),
    )
}

/** Status card: title + phase chip, current target, location; flash and "?" buttons. */
@Composable
fun NavStatusCard(
    title: String,
    phase: String,
    goal: String,
    currentTarget: String?,
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

/** Tappable pill, same look as [NavHintBanner] (e.g. "我在入口，從這裡開始"). */
@Composable
fun NavHintButton(text: String, background: Color, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Text(
        text,
        color = PanelText,
        fontSize = 12.sp,
        fontWeight = FontWeight.SemiBold,
        modifier = modifier
            .clip(CircleShape)
            .background(background)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
    )
}

/**
 * Route panel: the live prompt + remaining metres, and the mini-map below. [userPos] is
 * the live position (PDR) when known, else the route start (the last photo fix); the
 * heading arrow is drawn only when [userHeadingDeg] is known.
 */
@Composable
fun RoutePanel(
    route: PathResponse?,
    prompt: String?,
    statusMessage: String?,
    userHeadingDeg: Double?,
    modifier: Modifier = Modifier,
    userPos: Vec2? = null,
    remainingM: Double? = null,
    offRoute: Boolean = false,
) {
    if (route != null) {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(Color(0xFFF0EDE4).copy(alpha = 0.92f))
                .padding(10.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    prompt ?: "定位中…",
                    color = Color(0xFF3D3D3A),
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f),
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(10.dp))
                        .background(Color(0xFF5B8A72).copy(alpha = 0.18f))
                        .padding(horizontal = 8.dp, vertical = 3.dp),
                ) {
                    Text(
                        "${(remainingM ?: route.distance_m).roundToInt()} m",
                        color = Color(0xFF3D6B52),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
            RouteMap(
                route = route,
                userX = userPos?.x ?: route.start.x,
                userY = userPos?.y ?: route.start.y,
                userHeadingDeg = userHeadingDeg,
                offRoute = offRoute,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp),
            )
        }
    } else if (statusMessage != null) {
        Text(
            statusMessage,
            color = PanelText.copy(alpha = 0.8f),
            fontSize = 12.sp,
            modifier = modifier.fillMaxWidth(),
        )
    }
}

// ── Mini-Motorways-style pastel palette ─────────────────────────────────────
private val MapBg           = Color(0xFFCDD4B1)   // sage green canvas
private val RoadColor       = Color(0xFFF5F0E8)   // cream-white roads
private val RoadShadow      = Color(0xFFB8B5AB)   // subtle road shadow
private val RouteTeal       = Color(0xFF4AA8A0)   // teal route line
private val TargetCoral     = Color(0xFFE8655A)   // coral-red destination
private val TargetPinStem   = Color(0xFFC04E44)   // darker stem
private val UserBlue        = Color(0xFF4B7BB5)   // calm blue user marker
private val UserOffRouteRed = Color(0xFFE8655A)   // red when off route

/**
 * Mini-Motorways-style flat top-down map: pastel background, thick cream roads,
 * teal route line, pin-drop target, clean user marker with heading arrow.
 */
@Composable
fun RouteMap(
    route: PathResponse,
    userX: Double?,
    userY: Double?,
    userHeadingDeg: Double?,
    modifier: Modifier = Modifier,
    offRoute: Boolean = false,
) {
    val infinite = rememberInfiniteTransition(label = "mapAnim")
    val pulse by infinite.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(2000, easing = LinearEasing), RepeatMode.Restart),
        label = "pulse",
    )
    val dashPhase by infinite.animateFloat(
        initialValue = 0f, targetValue = 20f,
        animationSpec = infiniteRepeatable(tween(1200, easing = LinearEasing), RepeatMode.Restart),
        label = "dashPhase",
    )

    Canvas(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(MapBg),
    ) {
        val pts = route.nodes
        if (pts.isEmpty()) return@Canvas

        val minX = pts.minOf { it.x }; val maxX = pts.maxOf { it.x }
        val minY = pts.minOf { it.y }; val maxY = pts.maxOf { it.y }
        val pad = 22.dp.toPx()
        val spanX = maxOf(maxX - minX, 1.0).toFloat()
        val spanY = maxOf(maxY - minY, 1.0).toFloat()
        val scale = minOf((size.width - 2 * pad) / spanX, (size.height - 2 * pad) / spanY)
        val offX = (size.width - spanX * scale) / 2
        val offY = (size.height - spanY * scale) / 2
        fun toScreen(x: Double, y: Double) = Offset(
            offX + ((x - minX) * scale).toFloat(),
            size.height - offY - ((y - minY) * scale).toFloat(),
        )

        // ── walkable edges as thick cream "roads" ──
        val roadWidth = 8.dp.toPx()
        val shadowWidth = 10.dp.toPx()
        val byId = pts.associateBy { it.id }
        // shadow pass
        route.edges.forEach { e ->
            val a = byId[e.from] ?: return@forEach
            val b = byId[e.to] ?: return@forEach
            drawLine(RoadShadow, toScreen(a.x, a.y), toScreen(b.x, b.y),
                strokeWidth = shadowWidth, cap = StrokeCap.Round)
        }
        // road pass
        route.edges.forEach { e ->
            val a = byId[e.from] ?: return@forEach
            val b = byId[e.to] ?: return@forEach
            drawLine(RoadColor, toScreen(a.x, a.y), toScreen(b.x, b.y),
                strokeWidth = roadWidth, cap = StrokeCap.Round)
        }

        // ── planned route: solid teal over the road ──
        if (route.polyline.size >= 2) {
            val line = Path()
            route.polyline.forEachIndexed { i, v ->
                val s = toScreen(v[0], v[1])
                if (i == 0) line.moveTo(s.x, s.y) else line.lineTo(s.x, s.y)
            }
            drawPath(line, RouteTeal,
                style = Stroke(width = 5.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
            // animated dashes for direction
            drawPath(line, Color.White.copy(alpha = 0.55f),
                style = Stroke(
                    width = 2.5.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round,
                    pathEffect = PathEffect.dashPathEffect(
                        floatArrayOf(4.dp.toPx(), 10.dp.toPx()), dashPhase.dp.toPx()),
                ))
        }

        // ── target: map-pin style ──
        val t = toScreen(route.target.x, route.target.y)
        val pinR = 8.dp.toPx()
        val pinH = 14.dp.toPx()
        // pin stem (triangle)
        val pinPath = Path().apply {
            moveTo(t.x - 4.dp.toPx(), t.y - pinR * 0.3f)
            lineTo(t.x, t.y + pinH)
            lineTo(t.x + 4.dp.toPx(), t.y - pinR * 0.3f)
            close()
        }
        drawPath(pinPath, TargetPinStem)
        // pin head (circle)
        drawCircle(TargetCoral, radius = pinR, center = Offset(t.x, t.y - pinR * 0.2f))
        drawCircle(Color.White, radius = 3.5.dp.toPx(), center = Offset(t.x, t.y - pinR * 0.2f))

        // ── user marker: circle + heading ──
        if (userX != null && userY != null) {
            val u = toScreen(userX, userY)
            val color = if (offRoute) UserOffRouteRed else UserBlue

            // gentle pulse ring
            val pulseRadius = (10 + 8 * pulse).dp.toPx()
            val pulseAlpha = (1f - pulse) * 0.25f
            drawCircle(color.copy(alpha = pulseAlpha), radius = pulseRadius, center = u)

            // outer white ring (shadow-ish)
            drawCircle(Color.White, radius = 8.dp.toPx(), center = u)
            // main filled circle
            drawCircle(color, radius = 6.dp.toPx(), center = u)
            // inner white dot
            drawCircle(Color.White, radius = 2.dp.toPx(), center = u)

            // heading arrow
            userHeadingDeg?.let { h ->
                val rad = Math.toRadians(h)
                val len = 16.dp.toPx()
                val tip = Offset(u.x + len * sin(rad).toFloat(), u.y - len * cos(rad).toFloat())
                // white border of arrow
                drawLine(Color.White, u, tip, strokeWidth = 4.dp.toPx(), cap = StrokeCap.Round)
                // colored arrow
                drawLine(color, u, tip, strokeWidth = 2.5.dp.toPx(), cap = StrokeCap.Round)
                // arrowhead
                val wingLen = 5.dp.toPx()
                val wingAngle = Math.toRadians(30.0)
                val leftRad = rad + Math.PI - wingAngle
                val rightRad = rad + Math.PI + wingAngle
                val leftWing = Offset(tip.x + wingLen * sin(leftRad).toFloat(), tip.y - wingLen * cos(leftRad).toFloat())
                val rightWing = Offset(tip.x + wingLen * sin(rightRad).toFloat(), tip.y - wingLen * cos(rightRad).toFloat())
                drawLine(Color.White, tip, leftWing, strokeWidth = 3.5.dp.toPx(), cap = StrokeCap.Round)
                drawLine(Color.White, tip, rightWing, strokeWidth = 3.5.dp.toPx(), cap = StrokeCap.Round)
                drawLine(color, tip, leftWing, strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
                drawLine(color, tip, rightWing, strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
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

/**
 * Pick where the walk begins: each start shows the view to face. Tapping one tells the
 * app "I stand here facing this", which fixes both position and heading at once.
 */
@Composable
fun StartPointDialog(
    starts: List<StartPoint>,
    photoBaseUrl: String,
    onPick: (StartPoint) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (starts.size == 1) "從「${starts[0].name}」開始" else "從哪裡開始？") },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Text("站到起點，轉身讓眼前的景象和照片一樣，再按下面的按鈕。App 會把你面對的方向當成照片的方向。",
                    fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                starts.forEach { s ->
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        s.photo_url?.let { url ->
                            AsyncImage(
                                model = photoBaseUrl.trimEnd('/') + url,
                                contentDescription = "${s.name}：要面向的畫面",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(3f / 4f)
                                    .clip(RoundedCornerShape(10.dp)),
                            )
                        }
                        Button(onClick = { onPick(s) }, modifier = Modifier.fillMaxWidth()) {
                            Text(if (starts.size == 1) "我站好了，面向照片的方向" else "我在「${s.name}」，面向照片的方向")
                        }
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = { TextButton(onClick = onDismiss) { Text("取消") } },
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
