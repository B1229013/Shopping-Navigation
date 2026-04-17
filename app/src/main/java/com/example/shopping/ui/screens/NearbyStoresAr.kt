package com.example.shopping.ui.screens

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.util.Log
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.Place
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.shopping.BuildConfig
import com.example.shopping.ui.theme.*
import com.google.android.gms.maps.model.LatLng
import com.google.android.libraries.places.api.Places
import com.google.android.libraries.places.api.model.CircularBounds
import com.google.android.libraries.places.api.model.Place
import com.google.android.libraries.places.api.net.SearchNearbyRequest
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

private const val FOV_DEG = 60f
private const val RADIUS_M = 1000.0

// ── Device azimuth from rotation vector sensor ──────────────

@Composable
private fun rememberDeviceAzimuth(): State<Float> {
    val context = LocalContext.current
    val azimuth = remember { mutableFloatStateOf(0f) }

    DisposableEffect(Unit) {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val sensor = sm?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (sm == null || sensor == null) return@DisposableEffect onDispose { }

        val rotationMatrix = FloatArray(9)
        val orientation = FloatArray(3)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                if (event.sensor.type != Sensor.TYPE_ROTATION_VECTOR) return
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                SensorManager.getOrientation(rotationMatrix, orientation)
                val deg = Math.toDegrees(orientation[0].toDouble()).toFloat()
                azimuth.floatValue = (deg + 360f) % 360f
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        sm.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_UI)
        onDispose { sm.unregisterListener(listener) }
    }
    return azimuth
}

// ── Geometry helpers ────────────────────────────────────────

private fun bearingBetween(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Float {
    val phi1 = Math.toRadians(lat1)
    val phi2 = Math.toRadians(lat2)
    val dLon = Math.toRadians(lon2 - lon1)
    val y = sin(dLon) * cos(phi2)
    val x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dLon)
    val deg = Math.toDegrees(atan2(y, x)).toFloat()
    return (deg + 360f) % 360f
}

private fun signedDelta(a: Float): Float {
    var x = ((a % 360f) + 360f) % 360f
    if (x > 180f) x -= 360f
    return x
}

private fun distanceMeters(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
    val r = 6371000.0
    val dLat = Math.toRadians(lat2 - lat1)
    val dLon = Math.toRadians(lon2 - lon1)
    val a = sin(dLat / 2) * sin(dLat / 2) +
        cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) *
        sin(dLon / 2) * sin(dLon / 2)
    return 2 * r * kotlin.math.asin(kotlin.math.sqrt(a))
}

private fun formatDist(m: Double): String =
    if (m < 1000) "${m.toInt()} m" else "%.1f km".format(m / 1000.0)

// ── Main overlay ────────────────────────────────────────────

@Composable
fun NearbyStoresAr(
    location: Location,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val azimuth by rememberDeviceAzimuth()
    var stores by remember { mutableStateOf<List<Place>>(emptyList()) }

    DisposableEffect(location.latitude, location.longitude) {
        if (BuildConfig.MAPS_API_KEY.isBlank()) return@DisposableEffect onDispose { }
        if (!Places.isInitialized()) {
            runCatching {
                Places.initializeWithNewPlacesApiEnabled(context.applicationContext, BuildConfig.MAPS_API_KEY)
            }
        }
        val client = Places.createClient(context)
        val circle = CircularBounds.newInstance(
            LatLng(location.latitude, location.longitude),
            RADIUS_M
        )
        val fields = listOf(
            Place.Field.ID,
            Place.Field.DISPLAY_NAME,
            Place.Field.LOCATION,
            Place.Field.PRIMARY_TYPE,
        )
        val request = SearchNearbyRequest.builder(circle, fields)
            .setIncludedTypes(listOf("supermarket", "convenience_store", "grocery_store"))
            .setMaxResultCount(12)
            .setRankPreference(SearchNearbyRequest.RankPreference.DISTANCE)
            .build()
        val task = client.searchNearby(request)
        task.addOnSuccessListener { stores = it.places }
        task.addOnFailureListener { Log.e("NearbyStoresAr", "search failed", it) }
        onDispose { }
    }

    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        val widthDp = maxWidth
        val density = LocalDensity.current

        // Compass badge (top centre)
        CompassBadge(
            azimuth = azimuth,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 40.dp)
        )

        // In-view store markers
        stores.forEach { place ->
            val lat = place.location?.latitude ?: return@forEach
            val lng = place.location?.longitude ?: return@forEach
            val bearing = bearingBetween(location.latitude, location.longitude, lat, lng)
            val rel = signedDelta(bearing - azimuth)
            val inView = abs(rel) <= FOV_DEG / 2f
            val dist = distanceMeters(location.latitude, location.longitude, lat, lng)

            // Horizontal fraction: -FOV/2 → 0.0, 0 → 0.5, +FOV/2 → 1.0
            val xFraction = 0.5f + rel / FOV_DEG

            AnimatedVisibility(
                visible = inView,
                enter = fadeIn(tween(220)),
                exit = fadeOut(tween(160)),
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .graphicsLayer {
                        translationX = with(density) {
                            (xFraction * widthDp.toPx()) - (110.dp.toPx() / 2f)
                        }
                        translationY = with(density) { 120.dp.toPx() }
                    }
            ) {
                StoreArMarker(
                    name = place.displayName ?: "商店",
                    distance = formatDist(dist),
                    type = place.primaryType,
                )
            }
        }

        // Closest-store summary bar (bottom-above nav buttons)
        val closest = stores.mapNotNull { p ->
            val lat = p.location?.latitude ?: return@mapNotNull null
            val lng = p.location?.longitude ?: return@mapNotNull null
            Triple(p, bearingBetween(location.latitude, location.longitude, lat, lng),
                distanceMeters(location.latitude, location.longitude, lat, lng))
        }.minByOrNull { it.third }

        if (closest != null) {
            ClosestStoreBanner(
                name = closest.first.displayName ?: "商店",
                distance = formatDist(closest.third),
                relativeAngle = signedDelta(closest.second - azimuth),
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 120.dp, start = 24.dp, end = 24.dp)
            )
        }
    }
}

// ── Marker ──────────────────────────────────────────────────

@Composable
private fun StoreArMarker(
    name: String,
    distance: String,
    type: String?,
) {
    Surface(
        modifier = Modifier.width(110.dp),
        color = Noir.copy(alpha = 0.78f),
        shape = RoundedCornerShape(12.dp),
        border = BorderStroke(1.dp, Gold.copy(alpha = 0.6f)),
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(
                modifier = Modifier
                    .size(26.dp)
                    .clip(CircleShape)
                    .background(Gold),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Place, null, tint = Noir, modifier = Modifier.size(16.dp))
            }
            Spacer(Modifier.height(4.dp))
            Text(
                name,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.SemiBold,
                color = TextPrimary,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                distance,
                style = MaterialTheme.typography.labelSmall,
                color = Gold,
            )
            if (!type.isNullOrBlank()) {
                Text(
                    prettyTypeShort(type),
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary,
                )
            }
        }
    }
}

// ── Compass ─────────────────────────────────────────────────

@Composable
private fun CompassBadge(azimuth: Float, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        color = Noir.copy(alpha = 0.7f),
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, Gold.copy(alpha = 0.4f)),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                Icons.Default.Explore,
                null,
                tint = Gold,
                modifier = Modifier
                    .size(18.dp)
                    .rotate(-azimuth)
            )
            Spacer(Modifier.width(6.dp))
            Text(
                "${azimuth.toInt()}° ${cardinal(azimuth)}",
                style = MaterialTheme.typography.labelSmall,
                color = TextPrimary,
            )
        }
    }
}

private fun cardinal(deg: Float): String = when {
    deg < 22.5f || deg >= 337.5f -> "N"
    deg < 67.5f -> "NE"
    deg < 112.5f -> "E"
    deg < 157.5f -> "SE"
    deg < 202.5f -> "S"
    deg < 247.5f -> "SW"
    deg < 292.5f -> "W"
    else -> "NW"
}

// ── Closest-store banner ───────────────────────────────────

@Composable
private fun ClosestStoreBanner(
    name: String,
    distance: String,
    relativeAngle: Float,
    modifier: Modifier = Modifier,
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        color = Noir.copy(alpha = 0.82f),
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, Gold.copy(alpha = 0.5f)),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Default.Place,
                null,
                tint = Gold,
                modifier = Modifier
                    .size(22.dp)
                    .rotate(relativeAngle),
            )
            Spacer(Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "最近的商店",
                    style = MaterialTheme.typography.labelSmall,
                    color = TextTertiary,
                )
                Text(
                    name,
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = TextPrimary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Text(
                distance,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = Gold,
            )
        }
    }
}

private fun prettyTypeShort(type: String): String = when (type) {
    "supermarket" -> "超市"
    "convenience_store" -> "便利店"
    "grocery_store" -> "雜貨"
    else -> ""
}
