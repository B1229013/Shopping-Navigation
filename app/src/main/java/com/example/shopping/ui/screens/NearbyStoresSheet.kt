package com.example.shopping.ui.screens

import android.content.Intent
import android.location.Location
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.shopping.BuildConfig
import com.example.shopping.ui.theme.*
import com.google.android.gms.maps.model.LatLng
import com.google.android.libraries.places.api.Places
import com.google.android.libraries.places.api.model.CircularBounds
import com.google.android.libraries.places.api.model.Place
import com.google.android.libraries.places.api.net.SearchNearbyRequest

// ── Category presets ────────────────────────────────────────

private data class StoreCategory(
    val id: String,
    val label: String,
    val types: List<String>,
)

private val storeCategories = listOf(
    StoreCategory("all", "全部", listOf("supermarket", "convenience_store", "grocery_store")),
    StoreCategory("supermarket", "超市", listOf("supermarket")),
    StoreCategory("convenience", "便利商店", listOf("convenience_store")),
    StoreCategory("grocery", "雜貨", listOf("grocery_store")),
)

// ── UI state ─────────────────────────────────────────────────

private sealed interface NearbyState {
    data object Loading : NearbyState
    data class Ready(val stores: List<Place>) : NearbyState
    data class Error(val message: String) : NearbyState
    data object NoKey : NearbyState
}

// ── The bottom sheet ────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NearbyStoresSheet(
    location: Location,
    onDismiss: () -> Unit,
) {
    val context = LocalContext.current
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var selectedCategory by remember { mutableStateOf(storeCategories.first()) }
    var state by remember { mutableStateOf<NearbyState>(NearbyState.Loading) }

    DisposableEffect(selectedCategory, location.latitude, location.longitude) {
        if (BuildConfig.MAPS_API_KEY.isBlank()) {
            state = NearbyState.NoKey
            return@DisposableEffect onDispose { }
        }
        if (!Places.isInitialized()) {
            runCatching {
                Places.initializeWithNewPlacesApiEnabled(context.applicationContext, BuildConfig.MAPS_API_KEY)
            }
        }
        state = NearbyState.Loading

        val client = Places.createClient(context)
        val circle = CircularBounds.newInstance(
            LatLng(location.latitude, location.longitude),
            1000.0
        )
        val fields = listOf(
            Place.Field.ID,
            Place.Field.DISPLAY_NAME,
            Place.Field.FORMATTED_ADDRESS,
            Place.Field.LOCATION,
            Place.Field.PRIMARY_TYPE,
            Place.Field.RATING,
            Place.Field.GOOGLE_MAPS_URI,
        )
        val request = SearchNearbyRequest.builder(circle, fields)
            .setIncludedTypes(selectedCategory.types)
            .setMaxResultCount(20)
            .setRankPreference(SearchNearbyRequest.RankPreference.DISTANCE)
            .build()

        val task = client.searchNearby(request)
        task.addOnSuccessListener { response ->
            state = NearbyState.Ready(response.places)
        }
        task.addOnFailureListener { e ->
            state = NearbyState.Error(e.message ?: e::class.java.simpleName)
        }

        onDispose { }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = SurfaceBase,
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(bottom = 24.dp)) {
            Text(
                "1 公里內的商店",
                style = MaterialTheme.typography.titleLarge,
                color = TextPrimary,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp)
            )
            Text(
                "以 GPS 當前位置為中心搜尋附近商店",
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 2.dp)
            )

            Spacer(Modifier.height(12.dp))

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                storeCategories.forEach { cat ->
                    val selected = cat.id == selectedCategory.id
                    Text(
                        cat.label,
                        style = MaterialTheme.typography.labelMedium,
                        color = if (selected) Noir else TextSecondary,
                        modifier = Modifier
                            .clip(RoundedCornerShape(10.dp))
                            .background(if (selected) Gold else SurfaceDim)
                            .clickable { selectedCategory = cat }
                            .padding(horizontal = 12.dp, vertical = 8.dp)
                    )
                }
            }

            Spacer(Modifier.height(14.dp))

            when (val s = state) {
                is NearbyState.Loading -> {
                    Box(
                        modifier = Modifier.fillMaxWidth().padding(40.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(color = Gold, strokeWidth = 2.dp)
                    }
                }
                is NearbyState.NoKey -> {
                    Text(
                        "尚未設定 MAPS_API_KEY。請在 local.properties 加入：\nMAPS_API_KEY=你的金鑰",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Danger,
                        modifier = Modifier.padding(20.dp)
                    )
                }
                is NearbyState.Error -> {
                    Text(
                        "搜尋失敗：${s.message}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Danger,
                        modifier = Modifier.padding(20.dp)
                    )
                }
                is NearbyState.Ready -> {
                    if (s.stores.isEmpty()) {
                        Text(
                            "附近 1 公里內沒有找到這類商店",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextTertiary,
                            modifier = Modifier.padding(20.dp)
                        )
                    } else {
                        LazyColumn(
                            modifier = Modifier.heightIn(max = 480.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            contentPadding = PaddingValues(horizontal = 20.dp)
                        ) {
                            items(s.stores) { place ->
                                val latLng = place.location
                                val dist = latLng?.let {
                                    haversineMeters(
                                        location.latitude, location.longitude,
                                        it.latitude, it.longitude
                                    )
                                }
                                StoreRow(place = place, distanceMeters = dist, onClick = {
                                    val uri = place.googleMapsUri
                                        ?: latLng?.let {
                                            Uri.parse("geo:${it.latitude},${it.longitude}?q=${it.latitude},${it.longitude}(${place.displayName ?: ""})")
                                        }
                                    if (uri != null) {
                                        runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, uri)) }
                                    }
                                })
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Row ─────────────────────────────────────────────────────

@Composable
private fun StoreRow(
    place: Place,
    distanceMeters: Double?,
    onClick: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(SurfaceDim)
            .clickable(onClick = onClick)
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(GoldSurface),
            contentAlignment = Alignment.Center
        ) {
            Icon(Icons.Default.LocationOn, null, tint = Gold, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.width(12.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                place.displayName ?: "(無名稱)",
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary,
                fontWeight = FontWeight.SemiBold,
                maxLines = 1,
            )
            val subtitle = buildString {
                place.primaryType?.let { append(prettyType(it)) }
                place.formattedAddress?.let {
                    if (isNotEmpty()) append("  ·  ")
                    append(it)
                }
            }
            if (subtitle.isNotBlank()) {
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary,
                    maxLines = 2,
                )
            }
            Spacer(Modifier.height(2.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (distanceMeters != null) {
                    Text(
                        formatDistance(distanceMeters),
                        style = MaterialTheme.typography.labelSmall,
                        color = Gold,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                place.rating?.let { r ->
                    if (distanceMeters != null) Spacer(Modifier.width(10.dp))
                    Icon(Icons.Default.Star, null, tint = Gold, modifier = Modifier.size(12.dp))
                    Spacer(Modifier.width(2.dp))
                    Text(
                        "%.1f".format(r),
                        style = MaterialTheme.typography.labelSmall,
                        color = TextSecondary,
                    )
                }
            }
        }
    }
}

// ── Helpers ────────────────────────────────────────────────

private fun prettyType(type: String): String = when (type) {
    "supermarket" -> "超市"
    "convenience_store" -> "便利商店"
    "grocery_store" -> "雜貨店"
    else -> type.replace('_', ' ')
}

private fun haversineMeters(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
    val r = 6371000.0
    val dLat = Math.toRadians(lat2 - lat1)
    val dLon = Math.toRadians(lon2 - lon1)
    val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2)
    return 2 * r * Math.asin(Math.sqrt(a))
}

private fun formatDistance(meters: Double): String =
    if (meters < 1000) "${meters.toInt()} 公尺" else "%.1f 公里".format(meters / 1000.0)
