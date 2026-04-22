package com.example.shopping.ui.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.example.shopping.model.ShoppingItem
import com.example.shopping.ui.components.FadeInScreen
import com.example.shopping.ui.components.StaggeredItem
import com.example.shopping.ui.utils.matchesHomeCategory
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.delay

// ── Home color palette — flat design, fresh grocery feel ───
// Design system: Flat Design Mobile (Touch-First)
// Primary green + food amber, zero shadow, color blocking
private val HomeBg         = Color(0xFFECFDF5)  // fresh mint background
private val HomeSurface    = Color(0xFFF0F8F6)  // muted surface
private val HomeAccent     = Color(0xFF059669)  // primary green (emerald-600)
private val HomeAccentSoft = Color(0xFFD1FAE5)  // pale green tint
private val HomeOrange     = Color(0xFFD97706)  // amber accent for CTAs
private val HomeOrangeSoft = Color(0xFFFEF3C7)  // pale amber
private val HomeText       = Color(0xFF0F172A)  // slate-900 foreground
private val HomeTextMid    = Color(0xFF475569)  // slate-600
private val HomeTextLight  = Color(0xFF94A3B8)  // slate-400
private val HomeBorder     = Color(0xFFE1F2ED)  // green-tinted border
private val HomeWhite      = Color(0xFFFFFFFF)  // card surface

// ── Category data ──────────────────────────────────────────
private data class GroceryCategory(
    val name: String,
    val icon: ImageVector,
    val tint: Color,
    val bgColor: Color
)

private val groceryCategories = listOf(
    GroceryCategory("蔬菜", Icons.Default.Grass, Color(0xFF059669), Color(0xFFD1FAE5)),
    GroceryCategory("水果", Icons.Default.Spa, Color(0xFFD97706), Color(0xFFFEF3C7)),
    GroceryCategory("零食", Icons.Default.Cookie, Color(0xFFDB2777), Color(0xFFFCE7F3)),
    GroceryCategory("蛋奶", Icons.Default.Egg, Color(0xFF92400E), Color(0xFFFDE68A)),
    GroceryCategory("飲品", Icons.Default.LocalDrink, Color(0xFF2563EB), Color(0xFFDBEAFE)),
    GroceryCategory("調味", Icons.Default.Blender, Color(0xFF7C3AED), Color(0xFFEDE9FE)),
)

// ── Product image mapping ──────────────────────────────────
// Maps common grocery keywords to free Unsplash/Pexels image URLs.
// To use your own images: place them in res/drawable or use any URL.
private val productImageMap = mapOf(
    // Fruits
    "蘋果" to "https://images.unsplash.com/photo-1568702846914-96b305d2uj68?w=400",
    "香蕉" to "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400",
    "橘子" to "https://images.unsplash.com/photo-1547514701-42782101795e?w=400",
    "柳丁" to "https://images.unsplash.com/photo-1547514701-42782101795e?w=400",
    "葡萄" to "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=400",
    "西瓜" to "https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=400",
    "芒果" to "https://images.unsplash.com/photo-1553279768-865429fa0078?w=400",
    "草莓" to "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=400",
    // Vegetables
    "高麗菜" to "https://images.unsplash.com/photo-1594282486756-fa7e0e6d1ec0?w=400",
    "番茄" to "https://images.unsplash.com/photo-1546470427-0d4db154ceb8?w=400",
    "紅蘿蔔" to "https://images.unsplash.com/photo-1598170845058-32b9d6a5da37?w=400",
    "洋蔥" to "https://images.unsplash.com/photo-1618512496248-a07fe83aa8cb?w=400",
    "青椒" to "https://images.unsplash.com/photo-1563565375-f3fdfdbefa83?w=400",
    // Proteins
    "雞蛋" to "https://images.unsplash.com/photo-1582722872445-44dc5f7e3c8f?w=400",
    "牛奶" to "https://images.unsplash.com/photo-1563636619-e9143da7973b?w=400",
    "豬肉" to "https://images.unsplash.com/photo-1602470520998-f4a52199a3d6?w=400",
    "雞肉" to "https://images.unsplash.com/photo-1604503468506-a8da13d82791?w=400",
    // Beverages
    "茶" to "https://images.unsplash.com/photo-1556679343-c7306c1976bc?w=400",
    "咖啡" to "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=400",
    // Snacks
    "餅乾" to "https://images.unsplash.com/photo-1558961363-fa8fdf82db35?w=400",
    "巧克力" to "https://images.unsplash.com/photo-1481391319762-47dff72954d9?w=400",
)

// Category icon fallbacks when no image URL matches
private val categoryIcons = mapOf(
    "食品" to Icons.Default.Restaurant,
    "飲品" to Icons.Default.LocalDrink,
    "生活用品" to Icons.Default.ShoppingBag,
    "其他" to Icons.Default.MoreHoriz,
)

private fun getProductImageUrl(itemName: String): String? {
    return productImageMap.entries.firstOrNull { (key, _) ->
        itemName.contains(key)
    }?.value
}

private fun getCategoryIcon(location: String?): ImageVector {
    return categoryIcons[location] ?: Icons.Default.ShoppingCart
}

// ── Home Screen ────────────────────────────────────────────

@Composable
fun HomeScreen(
    shoppingItems: List<ShoppingItem>,
    onNavigateToList: () -> Unit,
    onNavigateToBudget: () -> Unit
) {
    val userName = Firebase.auth.currentUser?.email?.substringBefore("@") ?: "使用者"
    val pendingItems = shoppingItems.filter { !it.isChecked }
    val recentPurchased = shoppingItems.filter { it.isChecked }.takeLast(6).reversed()
    val totalSpent = shoppingItems.filter { it.isChecked }.sumOf { it.price * it.qty }

    var selectedCategory by remember { mutableStateOf<String?>(null) }
    val filteredItems = if (selectedCategory != null) {
        pendingItems.filter { matchesHomeCategory(it.name, selectedCategory!!) }
    } else pendingItems

    // Greeting based on time
    val greeting = remember {
        val hour = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY)
        when {
            hour < 12 -> "早安"
            hour < 18 -> "午安"
            else -> "晚安"
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(HomeBg)
    ) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 24.dp)
        ) {
            // ── Header: Greeting + Stats ──
            item {
                StaggeredItem(index = 0) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 24.dp, end = 24.dp, top = 16.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(
                                    "$greeting, $userName",
                                    style = MaterialTheme.typography.headlineSmall.copy(
                                        fontWeight = FontWeight.Bold,
                                        letterSpacing = (-0.3).sp
                                    ),
                                    color = HomeText
                                )
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(
                                    "今天想買什麼？",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = HomeTextMid
                                )
                            }
                            // Avatar circle
                            Box(
                                contentAlignment = Alignment.Center,
                                modifier = Modifier
                                    .size(44.dp)
                                    .clip(CircleShape)
                                    .background(HomeAccentSoft)
                            ) {
                                Text(
                                    userName.take(1).uppercase(),
                                    style = MaterialTheme.typography.titleMedium.copy(
                                        fontWeight = FontWeight.Bold
                                    ),
                                    color = HomeAccent
                                )
                            }
                        }
                    }
                }
            }

            // ── Promo / Summary Banner ──
            item {
                StaggeredItem(index = 1) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 24.dp, vertical = 16.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(
                                Brush.horizontalGradient(
                                    colors = listOf(
                                        HomeAccent,
                                        Color(0xFF5BA85A)
                                    )
                                )
                            )
                            .padding(20.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    "購物清單總覽",
                                    style = MaterialTheme.typography.titleMedium.copy(
                                        fontWeight = FontWeight.Bold
                                    ),
                                    color = Color.White
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    "${pendingItems.size} 項待購  ·  已花費 $$totalSpent",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = Color.White.copy(alpha = 0.85f)
                                )
                                Spacer(modifier = Modifier.height(14.dp))
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(10.dp))
                                        .background(Color.White.copy(alpha = 0.2f))
                                        .clickable { onNavigateToList() }
                                        .padding(horizontal = 16.dp, vertical = 8.dp)
                                ) {
                                    Text(
                                        "查看清單",
                                        style = MaterialTheme.typography.labelMedium.copy(
                                            fontWeight = FontWeight.SemiBold
                                        ),
                                        color = Color.White
                                    )
                                }
                            }
                            // Decorative icon cluster
                            Box(
                                contentAlignment = Alignment.Center,
                                modifier = Modifier.size(72.dp)
                            ) {
                                Icon(
                                    Icons.Default.ShoppingCart,
                                    contentDescription = null,
                                    tint = Color.White.copy(alpha = 0.3f),
                                    modifier = Modifier
                                        .size(64.dp)
                                        .graphicsLayer { rotationZ = -12f }
                                )
                                Icon(
                                    Icons.Default.ShoppingCart,
                                    contentDescription = null,
                                    tint = Color.White.copy(alpha = 0.9f),
                                    modifier = Modifier.size(36.dp)
                                )
                            }
                        }
                    }
                }
            }

            // ── Search Bar ──
            item {
                StaggeredItem(index = 2) {
                    var searchQuery by remember { mutableStateOf("") }
                    OutlinedTextField(
                        value = searchQuery,
                        onValueChange = { searchQuery = it },
                        placeholder = {
                            Text("搜尋商品...", color = HomeTextLight)
                        },
                        leadingIcon = {
                            Icon(Icons.Default.Search, null, tint = HomeTextMid)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 24.dp)
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = HomeAccent,
                            unfocusedBorderColor = Color.Transparent,
                            cursorColor = HomeAccent,
                            focusedTextColor = HomeText,
                            unfocusedTextColor = HomeText,
                            focusedContainerColor = HomeWhite,
                            unfocusedContainerColor = HomeWhite,
                            focusedLeadingIconColor = HomeAccent,
                            unfocusedLeadingIconColor = HomeTextMid
                        ),
                        singleLine = true
                    )
                }
            }

            // ── Categories ──
            item {
                StaggeredItem(index = 3) {
                    Column(modifier = Modifier.padding(top = 20.dp)) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(horizontal = 24.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                "分類",
                                style = MaterialTheme.typography.titleMedium.copy(
                                    fontWeight = FontWeight.Bold
                                ),
                                color = HomeText
                            )
                            Text(
                                "全部",
                                style = MaterialTheme.typography.bodySmall,
                                color = HomeAccent,
                                modifier = Modifier.clickable { selectedCategory = null }
                            )
                        }
                        Spacer(modifier = Modifier.height(14.dp))
                        LazyRow(
                            contentPadding = PaddingValues(horizontal = 24.dp),
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            items(groceryCategories) { cat ->
                                HomeCategoryChip(
                                    category = cat,
                                    isSelected = selectedCategory == cat.name,
                                    onClick = {
                                        selectedCategory = if (selectedCategory == cat.name) null else cat.name
                                    }
                                )
                            }
                        }
                    }
                }
            }

            // ── Products Section: Pending Items ──
            item {
                StaggeredItem(index = 4) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 24.dp, end = 24.dp, top = 22.dp, bottom = 12.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            if (selectedCategory != null) "$selectedCategory 商品" else "待購商品",
                            style = MaterialTheme.typography.titleMedium.copy(
                                fontWeight = FontWeight.Bold
                            ),
                            color = HomeText
                        )
                        Text(
                            "${filteredItems.size} 項",
                            style = MaterialTheme.typography.bodySmall,
                            color = HomeTextMid
                        )
                    }
                }
            }

            // ── Product Grid (2 columns via paired rows) ──
            val pairedItems = filteredItems.chunked(2)
            itemsIndexed(pairedItems) { index, pair ->
                StaggeredItem(index = index + 5, delayPerItem = 40L) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 24.dp),
                        horizontalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        pair.forEach { item ->
                            Box(modifier = Modifier.weight(1f)) {
                                HomeProductCard(item = item)
                            }
                        }
                        // Fill empty space if odd number
                        if (pair.size == 1) {
                            Spacer(modifier = Modifier.weight(1f))
                        }
                    }
                    Spacer(modifier = Modifier.height(14.dp))
                }
            }

            // ── Empty state ──
            if (filteredItems.isEmpty()) {
                item {
                    StaggeredItem(index = 5) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 48.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                Icons.Default.ShoppingCart,
                                contentDescription = null,
                                tint = HomeBorder,
                                modifier = Modifier.size(56.dp)
                            )
                            Spacer(modifier = Modifier.height(12.dp))
                            Text(
                                if (selectedCategory != null) "此分類目前沒有商品"
                                else "購物清單是空的",
                                style = MaterialTheme.typography.bodyMedium,
                                color = HomeTextLight
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                "到清單頁面新增商品吧！",
                                style = MaterialTheme.typography.bodySmall,
                                color = HomeTextLight
                            )
                        }
                    }
                }
            }

            // ── Recently Purchased ──
            if (recentPurchased.isNotEmpty()) {
                item {
                    StaggeredItem(index = pairedItems.size + 6) {
                        Column(modifier = Modifier.padding(top = 8.dp)) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(horizontal = 24.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    "最近已購",
                                    style = MaterialTheme.typography.titleMedium.copy(
                                        fontWeight = FontWeight.Bold
                                    ),
                                    color = HomeText
                                )
                                Text(
                                    "查看全部",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = HomeAccent,
                                    modifier = Modifier.clickable { onNavigateToList() }
                                )
                            }
                            Spacer(modifier = Modifier.height(12.dp))
                            LazyRow(
                                contentPadding = PaddingValues(horizontal = 24.dp),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                items(recentPurchased) { item ->
                                    HomeRecentCard(item = item)
                                }
                            }
                        }
                    }
                }
            }

            // ── Quick Stats ──
            item {
                StaggeredItem(index = pairedItems.size + 7) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 24.dp, vertical = 16.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        HomeStatCard(
                            label = "待購",
                            value = "${pendingItems.size}",
                            icon = Icons.Default.ShoppingCart,
                            color = HomeAccent,
                            bgColor = HomeAccentSoft,
                            modifier = Modifier.weight(1f)
                        )
                        HomeStatCard(
                            label = "已花費",
                            value = "$$totalSpent",
                            icon = Icons.Default.AccountBalanceWallet,
                            color = HomeOrange,
                            bgColor = HomeOrangeSoft,
                            modifier = Modifier.weight(1f),
                            onClick = onNavigateToBudget
                        )
                    }
                }
            }
        }
    }
}

// ── Category Chip ──────────────────────────────────────────

@Composable
private fun HomeCategoryChip(
    category: GroceryCategory,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val bgColor by animateColorAsState(
        targetValue = if (isSelected) category.tint.copy(alpha = 0.15f) else category.bgColor,
        animationSpec = tween(250),
        label = "chip_bg"
    )
    val borderColor by animateColorAsState(
        targetValue = if (isSelected) category.tint else Color.Transparent,
        animationSpec = tween(250),
        label = "chip_border"
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clip(RoundedCornerShape(16.dp))
            .clickable(onClick = onClick)
            .padding(4.dp)
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(56.dp)
                .clip(CircleShape)
                .background(bgColor)
                .then(
                    if (isSelected) Modifier.drawBehind {
                        drawCircle(
                            color = borderColor,
                            style = Stroke(width = 2.dp.toPx())
                        )
                    } else Modifier
                )
        ) {
            Icon(
                category.icon,
                contentDescription = category.name,
                tint = category.tint,
                modifier = Modifier.size(26.dp)
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            category.name,
            style = MaterialTheme.typography.labelSmall.copy(
                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
            ),
            color = if (isSelected) category.tint else HomeTextMid
        )
    }
}

// ── Product Card ───────────────────────────────────────────

@Composable
private fun HomeProductCard(item: ShoppingItem) {
    val imageUrl = getProductImageUrl(item.name)

    // Press feedback — flat design: scale 0.97 on press
    var isPressed by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (isPressed) 0.97f else 1f,
        animationSpec = tween(if (isPressed) 80 else 200, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "card_press"
    )

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .graphicsLayer { scaleX = scale; scaleY = scale }
            .clip(RoundedCornerShape(16.dp))
            .background(HomeWhite)
            .clickable { /* navigate to detail */ }
    ) {
        // Image area — solid color background, no shadow
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .fillMaxWidth()
                .height(130.dp)
                .background(HomeAccentSoft)
        ) {
            if (imageUrl != null) {
                AsyncImage(
                    model = ImageRequest.Builder(LocalContext.current)
                        .data(imageUrl)
                        .crossfade(300)
                        .build(),
                    contentDescription = item.name,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                // Fallback: bold category icon in colored container
                Icon(
                    getCategoryIcon(item.location),
                    contentDescription = item.name,
                    tint = HomeAccent.copy(alpha = 0.4f),
                    modifier = Modifier.size(44.dp)
                )
            }

            // Favorite — 44x44 touch target
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(6.dp)
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(HomeWhite.copy(alpha = 0.9f))
                    .clickable { /* toggle fav */ },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.FavoriteBorder,
                    contentDescription = "收藏",
                    tint = Color(0xFFEF4444),
                    modifier = Modifier.size(18.dp)
                )
            }
        }

        // Info area
        Column(modifier = Modifier.padding(12.dp)) {
            Text(
                item.name,
                style = MaterialTheme.typography.titleSmall.copy(
                    fontWeight = FontWeight.SemiBold
                ),
                color = HomeText,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            if (item.location != null) {
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    item.location!!,
                    style = MaterialTheme.typography.labelSmall,
                    color = HomeTextLight
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "$${item.price}",
                    style = MaterialTheme.typography.titleMedium.copy(
                        fontWeight = FontWeight.Bold
                    ),
                    color = HomeAccent
                )
                // Qty badge — solid colored container
                Box(
                    contentAlignment = Alignment.Center,
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(HomeAccentSoft)
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        "x${item.qty}",
                        style = MaterialTheme.typography.labelSmall.copy(
                            fontWeight = FontWeight.SemiBold
                        ),
                        color = HomeAccent
                    )
                }
            }
        }
    }
}

// ── Recent Purchase Card (horizontal scroll) ───────────────

@Composable
private fun HomeRecentCard(item: ShoppingItem) {
    val imageUrl = getProductImageUrl(item.name)

    Row(
        modifier = Modifier
            .width(200.dp)
            .clip(RoundedCornerShape(14.dp))
            .background(HomeWhite)
            .padding(10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Thumbnail
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(48.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(HomeAccentSoft)
        ) {
            if (imageUrl != null) {
                AsyncImage(
                    model = ImageRequest.Builder(LocalContext.current)
                        .data(imageUrl)
                        .crossfade(true)
                        .build(),
                    contentDescription = item.name,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            } else {
                Icon(
                    getCategoryIcon(item.location),
                    contentDescription = null,
                    tint = HomeAccent.copy(alpha = 0.5f),
                    modifier = Modifier.size(22.dp)
                )
            }
        }
        Column(modifier = Modifier.weight(1f)) {
            Text(
                item.name,
                style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.SemiBold),
                color = HomeText,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                "$${item.price * item.qty}",
                style = MaterialTheme.typography.labelSmall,
                color = HomeAccent
            )
        }
        Icon(
            Icons.Default.CheckCircle,
            contentDescription = null,
            tint = HomeAccent.copy(alpha = 0.5f),
            modifier = Modifier.size(18.dp)
        )
    }
}

// ── Stat Card ──────────────────────────────────────────────

@Composable
private fun HomeStatCard(
    label: String,
    value: String,
    icon: ImageVector,
    color: Color,
    bgColor: Color,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null
) {
    Row(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(HomeWhite)
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(40.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(bgColor)
        ) {
            Icon(icon, null, tint = color, modifier = Modifier.size(20.dp))
        }
        Column {
            Text(
                value,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                color = HomeText
            )
            Text(
                label,
                style = MaterialTheme.typography.labelSmall,
                color = HomeTextMid
            )
        }
    }
}
