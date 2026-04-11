package com.example.shopping.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.navigation.NavController
import com.example.shopping.R
import com.example.shopping.model.DietRecord
import com.example.shopping.model.ShoppingItem
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

// --- Groq API Models ---
data class GroqRequest(val model: String = "llama-3.3-70b-versatile", val messages: List<GroqMessage>, val response_format: GroqResponseFormat? = null)
data class GroqMessage(val role: String, val content: String)
data class GroqResponseFormat(val type: String = "json_object")
data class GroqResponse(val choices: List<GroqChoice>)
data class GroqChoice(val message: GroqMessage)

interface GroqApiService {
    @POST("v1/chat/completions")
    suspend fun getCompletion(@Header("Authorization") apiKey: String, @Body request: GroqRequest): GroqResponse
}

val groqApi: GroqApiService by lazy {
    val logging = HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BODY }
    val client = OkHttpClient.Builder().addInterceptor(logging).build()
    Retrofit.Builder()
        .baseUrl("https://api.groq.com/openai/")
        .addConverterFactory(GsonConverterFactory.create())
        .client(client)
        .build()
        .create(GroqApiService::class.java)
}

@Serializable
data class TidiedReceiptItem(
    val name: String,
    val cat: String,
    val qty: Int,
    val total_price: Double?
)

@Serializable
data class BudgetEntry(
    val store: String,
    val timestamp: String,
    val total_amount: Double,
    val line_items: List<TidiedReceiptItem>
)

@Serializable
data class TidiedReceiptResponse(val budget_entry: BudgetEntry)

private val jsonFormatter = Json { ignoreUnknownKeys = true; coerceInputValues = true; isLenient = true }

// ── Tab definitions ──
private data class TabItem(
    val label: String,
    val icon: ImageVector,
    val selectedIcon: ImageVector
)

private val tabs = listOf(
    TabItem("清單", Icons.AutoMirrored.Filled.List, Icons.AutoMirrored.Filled.List),
    TabItem("分析", Icons.Default.Search, Icons.Default.Search),
    TabItem("預算", Icons.Default.AccountBalanceWallet, Icons.Default.AccountBalanceWallet),
    TabItem("紀錄", Icons.Default.CalendarMonth, Icons.Default.CalendarMonth),
    TabItem("助理", Icons.Default.Face, Icons.Default.Face)
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainContainer(rootNavController: NavController) {
    val context = LocalContext.current
    val shoppingFile = File(context.filesDir, "shopping_list.json")
    val dietFile = File(context.filesDir, "diet_records.json")
    val budgetFile = File(context.filesDir, "monthly_budget.txt")

    var shoppingItems by remember {
        val items = try {
            if (shoppingFile.exists()) jsonFormatter.decodeFromString<List<ShoppingItem>>(shoppingFile.readText()) else emptyList()
        } catch (e: Exception) { emptyList() }
        mutableStateOf(items)
    }

    var dietRecords by remember {
        val records = try {
            if (dietFile.exists()) jsonFormatter.decodeFromString<List<DietRecord>>(dietFile.readText()) else emptyList()
        } catch (e: Exception) { emptyList() }
        mutableStateOf(records)
    }

    var budgetTotalStr by remember {
        val b = try { if (budgetFile.exists()) budgetFile.readText() else "0" } catch(e: Exception) { "0" }
        mutableStateOf(b)
    }

    var selectedTab by remember { mutableIntStateOf(0) }

    LaunchedEffect(shoppingItems) { try { shoppingFile.writeText(jsonFormatter.encodeToString(shoppingItems)) } catch (e: Exception) {} }
    LaunchedEffect(dietRecords) { try { dietFile.writeText(jsonFormatter.encodeToString(dietRecords)) } catch (e: Exception) {} }
    LaunchedEffect(budgetTotalStr) { try { budgetFile.writeText(budgetTotalStr) } catch (e: Exception) {} }

    // Entrance animation
    var screenVisible by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { screenVisible = true }
    val screenAlpha by animateFloatAsState(
        targetValue = if (screenVisible) 1f else 0f,
        animationSpec = tween(400, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "screen_alpha"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Noir)
            .graphicsLayer { alpha = screenAlpha }
    ) {
        Scaffold(
            containerColor = Noir,
            topBar = {
                // Minimal top bar — just utility icons
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .statusBarsPadding()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = { rootNavController.navigate("settings") }) {
                        Icon(Icons.Default.Settings, "設定", tint = TextSecondary)
                    }
                    IconButton(onClick = {
                        val itemsJson = jsonFormatter.encodeToString(shoppingItems.filter { !it.isChecked })
                        rootNavController.currentBackStackEntry?.savedStateHandle?.set("shopping_list_json", itemsJson)
                        rootNavController.navigate("teammate_home")
                    }) {
                        Icon(Icons.Default.Place, "導覽", tint = TextSecondary)
                    }
                }
            },
            bottomBar = {
                // ── Cinematic Bottom Bar ──
                CinemaBottomBar(
                    selectedTab = selectedTab,
                    onTabSelected = { selectedTab = it }
                )
            }
        ) { innerPadding ->
            Box(modifier = Modifier.padding(innerPadding)) {
                // Crossfade between tabs for smooth transitions
                Crossfade(
                    targetState = selectedTab,
                    animationSpec = tween(250, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                    label = "tab_crossfade"
                ) { tab ->
                    when (tab) {
                        0 -> ShoppingListScreen(items = shoppingItems, onItemsUpdate = { shoppingItems = it })
                        1 -> IngredientsScreen(records = dietRecords, onRecordsUpdate = { dietRecords = it })
                        2 -> BudgetScreen(
                            shoppingItems = shoppingItems,
                            budgetTotal = budgetTotalStr,
                            onBudgetUpdate = { budgetTotalStr = it },
                            onItemsUpdate = { shoppingItems = it }
                        )
                        3 -> HistoryScreen(shoppingItems = shoppingItems)
                        4 -> AIScreen()
                    }
                }
            }
        }
    }
}

// ── Cinematic Bottom Navigation ─────────────────────────────

@Composable
private fun CinemaBottomBar(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit
) {
    Column {
        // Top edge — subtle gold line on active region
        Spacer(
            modifier = Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Border)
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(SurfaceDim)
                .navigationBarsPadding()
                .padding(vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            tabs.forEachIndexed { index, tab ->
                CinemaNavItem(
                    icon = tab.icon,
                    label = tab.label,
                    selected = selectedTab == index,
                    onClick = { onTabSelected(index) }
                )
            }
        }
    }
}

@Composable
private fun CinemaNavItem(
    icon: ImageVector,
    label: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    val tint by animateColorAsState(
        targetValue = if (selected) Gold else TextTertiary,
        animationSpec = tween(200),
        label = "nav_tint"
    )
    val scale by animateFloatAsState(
        targetValue = if (selected) 1.1f else 1f,
        animationSpec = tween(200, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "nav_scale"
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clickable(
                indication = null,
                interactionSource = remember { MutableInteractionSource() }
            ) { onClick() }
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        Icon(
            icon,
            contentDescription = label,
            tint = tint,
            modifier = Modifier
                .size(22.dp)
                .scale(scale)
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = tint,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
        // Active indicator dot
        AnimatedVisibility(
            visible = selected,
            enter = scaleIn(animationSpec = tween(200)) + fadeIn(),
            exit = scaleOut(animationSpec = tween(150)) + fadeOut()
        ) {
            Box(
                modifier = Modifier
                    .padding(top = 4.dp)
                    .size(4.dp)
                    .background(Gold, CircleShape)
            )
        }
    }
}

// ── Budget Screen Categories ────────────────────────────────

data class CategoryInfo(
    val name: String,
    val catId: String,
    val icon: ImageVector,
    val color: Color,
    val description: String
)

val dashboardCategories = listOf(
    CategoryInfo("食品", "Food", Icons.Default.Restaurant, ChartBlue, "肉類、蔬菜、生鮮食品"),
    CategoryInfo("飲品", "Beverages", Icons.Default.LocalDrink, ChartGreen, "牛奶、茶、飲料"),
    CategoryInfo("生活用品", "Groceries", Icons.Default.ShoppingBag, ChartAmber, "調味料、居家用品"),
    CategoryInfo("其他", "Other", Icons.Default.MoreHoriz, ChartViolet, "點數、手續費、其他項目")
)

// ── Budget Screen ───────────────────────────────────────────

@Composable
fun BudgetScreen(
    shoppingItems: List<ShoppingItem>,
    budgetTotal: String,
    onBudgetUpdate: (String) -> Unit,
    onItemsUpdate: (List<ShoppingItem>) -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    val totalSpent = shoppingItems.filter { it.isChecked }.sumOf { it.price * it.qty }
    val budgetValue = budgetTotal.toIntOrNull() ?: 0

    var isProcessing by remember { mutableStateOf(false) }
    var showAddDialog by remember { mutableStateOf(false) }
    var showSettingsDialog by remember { mutableStateOf(false) }

    val groqApiKey = stringResource(id = R.string.groq_api_key)
    val tempPhotoFile = remember { File(context.cacheDir, "receipt_photo.jpg") }
    val photoUri = remember { FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", tempPhotoFile) }

    val processImage: (Uri) -> Unit = { uri ->
        isProcessing = true
        scope.launch {
            val bitmap = withContext(Dispatchers.IO) { decodeUriToBitmap(context, uri) }
            if (bitmap != null) {
                val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
                val image = InputImage.fromBitmap(bitmap, 0)
                recognizer.process(image)
                    .addOnSuccessListener { visionText ->
                        val rawText = visionText.text
                        if (rawText.isBlank()) {
                            isProcessing = false
                            Toast.makeText(context, "未能從圖片中識別文字", Toast.LENGTH_SHORT).show()
                            return@addOnSuccessListener
                        }
                        scope.launch {
                            try {
                                val systemPrompt = """
                                    You are a specialized Data Extraction and Budget Logic Engine for the "Smart AI Shopping Assistant" Android app.
                                    Your sole responsibility is to process receipt images and output structured data for the Budget module.

                                    1. EXTRACTION RULES (PXMart/全聯 Structure)
                                    - Use only information visible in the provided text.
                                    - Date: Convert Minguo year (e.g. 110) to Western (2021). Format: YYYY/MM/DD.
                                    - Prices: Strip all "TX" or "$" symbols. Only store raw numbers.
                                    - Multiplier Logic: Look for * symbol (e.g., 55 * 3). Multiply these to get the line subtotal.
                                    - Language: Keep all item names in Traditional Chinese.

                                    2. CATEGORIZATION GUIDE
                                    Sort every item into these categories ONLY:
                                    - Food (食品): Meats, vegetables, dairy, snacks.
                                    - Beverages (飲品): Milk, tea, coffee, soda.
                                    - Groceries (生活用品): Cleaning supplies, toiletries, seasonings.
                                    - Other (其他): Points, fees, misc.

                                    Return ONLY a JSON object with this structure:
                                    {
                                      "budget_entry": {
                                        "store": "Store Name",
                                        "timestamp": "YYYY/MM/DD",
                                        "total_amount": 0.0,
                                        "line_items": [
                                          {"name": "Item Name", "cat": "Food/Beverages/Groceries/Other", "qty": 1, "total_price": 0.0}
                                        ]
                                      }
                                    }
                                """.trimIndent()

                                val response = withContext(Dispatchers.IO) {
                                    groqApi.getCompletion(
                                        "Bearer $groqApiKey",
                                        GroqRequest(
                                            messages = listOf(
                                                GroqMessage("system", systemPrompt),
                                                GroqMessage("user", "Receipt Text:\n$rawText")
                                            ),
                                            response_format = GroqResponseFormat()
                                        )
                                    )
                                }

                                val content = response.choices.firstOrNull()?.message?.content ?: ""
                                val tidied = jsonFormatter.decodeFromString<TidiedReceiptResponse>(content)

                                val newItems = shoppingItems.toMutableList()
                                tidied.budget_entry.line_items.forEach { line ->
                                    newItems.add(
                                        ShoppingItem(
                                            id = UUID.randomUUID().toString(),
                                            name = line.name,
                                            price = (line.total_price ?: 0.0).toInt() / line.qty.coerceAtLeast(1),
                                            qty = line.qty,
                                            location = line.cat,
                                            isChecked = true,
                                            createdAt = System.currentTimeMillis()
                                        )
                                    )
                                }
                                onItemsUpdate(newItems)
                                Toast.makeText(context, "成功擷取 ${tidied.budget_entry.line_items.size} 個項目", Toast.LENGTH_SHORT).show()
                            } catch (e: Exception) {
                                Log.e("Groq", "Error parsing receipt", e)
                                Toast.makeText(context, "解析錯誤: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                            } finally {
                                isProcessing = false
                            }
                        }
                    }
                    .addOnFailureListener {
                        isProcessing = false
                        Toast.makeText(context, "文字識別失敗", Toast.LENGTH_SHORT).show()
                    }
            } else {
                isProcessing = false
            }
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) processImage(photoUri)
    }
    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) processImage(uri)
    }

    val currentMonth = remember { SimpleDateFormat("MMMM", Locale.ENGLISH).format(Date()) }

    // Animated total
    val animatedTotal by animateIntAsState(totalSpent)

    FadeInScreen {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── Header ──
            StaggeredItem(index = 0) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "預算",
                        style = MaterialTheme.typography.displayMedium,
                        color = TextPrimary
                    )
                    Text(
                        "新增",
                        style = MaterialTheme.typography.titleMedium,
                        color = Gold,
                        modifier = Modifier
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { showAddDialog = true }
                            .padding(horizontal = 12.dp, vertical = 6.dp)
                    )
                }
            }

            // ── Donut Chart Card ──
            StaggeredItem(index = 1) {
                CinemaCard(modifier = Modifier.padding(horizontal = 20.dp)) {
                    Box(
                        contentAlignment = Alignment.Center,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp)
                    ) {
                        DonutChart(shoppingItems)
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(
                                "已支出",
                                style = MaterialTheme.typography.bodySmall,
                                color = TextSecondary
                            )
                            Text(
                                "$${String.format("%,d", animatedTotal)}",
                                style = MaterialTheme.typography.displaySmall,
                                color = TextPrimary
                            )
                            Text(
                                currentMonth,
                                style = MaterialTheme.typography.bodySmall,
                                color = TextTertiary
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Filter chips ──
            StaggeredItem(index = 2) {
                LazyRow(
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 20.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    item {
                        FilterChip(
                            selected = false,
                            onClick = {},
                            label = { Text(currentMonth, color = TextSecondary) },
                            shape = RoundedCornerShape(20.dp),
                            colors = FilterChipDefaults.filterChipColors(
                                containerColor = SurfaceBase
                            ),
                            border = FilterChipDefaults.filterChipBorder(
                                borderColor = Border,
                                enabled = true,
                                selected = false
                            )
                        )
                    }
                    item {
                        FilterChip(
                            selected = true,
                            onClick = { showSettingsDialog = true },
                            label = { Text("預算: $$budgetTotal") },
                            shape = RoundedCornerShape(20.dp),
                            colors = FilterChipDefaults.filterChipColors(
                                selectedContainerColor = Gold.copy(alpha = 0.15f),
                                selectedLabelColor = Gold
                            )
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Category List ──
            LazyColumn(
                modifier = Modifier
                    .weight(1f)
                    .padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = PaddingValues(bottom = 20.dp)
            ) {
                items(dashboardCategories.size) { index ->
                    val cat = dashboardCategories[index]
                    val catSpent = shoppingItems.filter { it.isChecked && it.location == cat.catId }.sumOf { it.price * it.qty }
                    val catBudget = if (budgetValue > 0) budgetValue / dashboardCategories.size else 0
                    val catLeft = catBudget - catSpent

                    StaggeredItem(index = index + 3) {
                        CategoryItem(cat, catSpent, catLeft)
                    }
                }
            }

            // Processing indicator
            if (isProcessing) {
                LinearProgressIndicator(
                    modifier = Modifier.fillMaxWidth(),
                    color = Gold,
                    trackColor = SurfaceBase
                )
            }
        }
    }

    // ── Add Expense Dialog ──
    if (showAddDialog) {
        AlertDialog(
            onDismissRequest = { showAddDialog = false },
            containerColor = SurfaceBase,
            titleContentColor = TextPrimary,
            title = { Text("新增支出", style = MaterialTheme.typography.titleLarge) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(
                        onClick = {
                            val permissionCheck = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                            if (permissionCheck == PackageManager.PERMISSION_GRANTED) cameraLauncher.launch(photoUri)
                            else Toast.makeText(context, "需要相機權限", Toast.LENGTH_SHORT).show()
                            showAddDialog = false
                        },
                        modifier = Modifier.fillMaxWidth().height(48.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.PhotoCamera, null)
                        Spacer(Modifier.width(8.dp))
                        Text("掃描收據", fontWeight = FontWeight.SemiBold)
                    }
                    OutlinedButton(
                        onClick = {
                            galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                            showAddDialog = false
                        },
                        modifier = Modifier.fillMaxWidth().height(48.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary),
                        border = androidx.compose.foundation.BorderStroke(1.dp, Border)
                    ) {
                        Icon(Icons.Default.PhotoLibrary, null)
                        Spacer(Modifier.width(8.dp))
                        Text("從相簿選取")
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { showAddDialog = false }) {
                    Text("取消", color = TextSecondary)
                }
            }
        )
    }

    // ── Budget Settings Dialog ──
    if (showSettingsDialog) {
        var tempBudget by remember { mutableStateOf(budgetTotal) }
        AlertDialog(
            onDismissRequest = { showSettingsDialog = false },
            containerColor = SurfaceBase,
            titleContentColor = TextPrimary,
            title = { Text("預算設定", style = MaterialTheme.typography.titleLarge) },
            text = {
                Column {
                    Text("設定每月預算總額:", style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
                    Spacer(modifier = Modifier.height(12.dp))
                    OutlinedTextField(
                        value = tempBudget,
                        onValueChange = { tempBudget = it },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(12.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Gold,
                            unfocusedBorderColor = Border,
                            cursorColor = Gold,
                            focusedTextColor = TextPrimary,
                            unfocusedTextColor = TextPrimary,
                            focusedContainerColor = SurfaceBright,
                            unfocusedContainerColor = SurfaceBright
                        )
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        onBudgetUpdate(tempBudget)
                        showSettingsDialog = false
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir)
                ) { Text("儲存", fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(onClick = { showSettingsDialog = false }) {
                    Text("取消", color = TextSecondary)
                }
            }
        )
    }
}

// ── Category Row Item ───────────────────────────────────────

@Composable
fun CategoryItem(category: CategoryInfo, spent: Int, left: Int) {
    PressableSurface(
        onClick = {},
        backgroundColor = SurfaceBase
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Icon with tinted background
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .background(category.color.copy(alpha = 0.12f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    category.icon,
                    contentDescription = null,
                    tint = category.color,
                    modifier = Modifier.size(22.dp)
                )
            }

            Spacer(modifier = Modifier.width(14.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    category.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                Text(
                    category.description,
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }

            Column(horizontalAlignment = Alignment.End) {
                Text(
                    "$${String.format("%,d", spent)}",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                Text(
                    if (left >= 0) "$${String.format("%,d", left)} 剩餘" else "$${String.format("%,d", -left)} 超支",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (left >= 0) Success else Danger
                )
            }
        }
    }
}

// ── Donut Chart ─────────────────────────────────────────────

@Composable
fun DonutChart(items: List<ShoppingItem>) {
    val total = items.filter { it.isChecked }.sumOf { it.price * it.qty }.toDouble()

    // Animate the sweep
    var animationPlayed by remember { mutableStateOf(false) }
    val animationProgress by animateFloatAsState(
        targetValue = if (animationPlayed) 1f else 0f,
        animationSpec = tween(800, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "donut_progress"
    )
    LaunchedEffect(Unit) { animationPlayed = true }

    Canvas(modifier = Modifier.fillMaxSize().padding(10.dp)) {
        if (total == 0.0) {
            drawCircle(color = SurfaceBright, style = Stroke(width = 16.dp.toPx()))
        } else {
            var startAngle = -90f
            dashboardCategories.forEach { cat ->
                val catSpent = items.filter { it.isChecked && it.location == cat.catId }.sumOf { it.price * it.qty }
                if (catSpent > 0) {
                    val sweepAngle = (catSpent / total * 360f).toFloat() * animationProgress
                    drawArc(
                        color = cat.color,
                        startAngle = startAngle,
                        sweepAngle = sweepAngle,
                        useCenter = false,
                        style = Stroke(width = 16.dp.toPx(), cap = StrokeCap.Round)
                    )
                    startAngle += (catSpent / total * 360f).toFloat() * animationProgress
                }
            }
        }
    }
}

fun decodeUriToBitmap(context: Context, uri: Uri): Bitmap? {
    return try {
        val inputStream = context.contentResolver.openInputStream(uri)
        BitmapFactory.decodeStream(inputStream)
    } catch (e: Exception) {
        null
    }
}
