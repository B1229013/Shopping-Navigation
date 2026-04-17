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
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
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
import android.util.Base64
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
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

// --- PaddleOCR API Helper ---
suspend fun callPaddleOcr(bitmap: Bitmap, apiUrl: String, accessToken: String): String {
    return withContext(Dispatchers.IO) {
        val stream = java.io.ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
        val base64Image = Base64.encodeToString(stream.toByteArray(), Base64.NO_WRAP)

        val requestBody = JSONObject().apply {
            put("file", base64Image)
            put("fileType", 1)
            put("visualize", false)
        }

        val client = OkHttpClient.Builder()
            .connectTimeout(60, java.util.concurrent.TimeUnit.SECONDS)
            .readTimeout(120, java.util.concurrent.TimeUnit.SECONDS)
            .build()

        val body = requestBody.toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())

        val request = okhttp3.Request.Builder()
            .url(apiUrl)
            .addHeader("Authorization", "token $accessToken")
            .addHeader("Content-Type", "application/json")
            .post(body)
            .build()

        val response = client.newCall(request).execute()
        val responseBody = response.body?.string() ?: ""

        if (!response.isSuccessful) {
            throw RuntimeException("PaddleOCR API error (${response.code}): $responseBody")
        }

        val json = JSONObject(responseBody)
        if (json.optInt("errorCode", -1) != 0) {
            throw RuntimeException("PaddleOCR error: ${json.optString("errorMsg", "Unknown")}")
        }

        // Extract text from ocrResults[n].prunedResult.rec_texts
        val result = json.optJSONObject("result")
        val ocrResults = result?.optJSONArray("ocrResults") ?: return@withContext ""
        val allText = StringBuilder()
        for (i in 0 until ocrResults.length()) {
            val page = ocrResults.optJSONObject(i) ?: continue
            val prunedResult = page.optJSONObject("prunedResult") ?: continue
            val recTexts = prunedResult.optJSONArray("rec_texts") ?: continue
            for (j in 0 until recTexts.length()) {
                if (allText.isNotEmpty()) allText.append("\n")
                allText.append(recTexts.optString(j, ""))
            }
        }
        allText.toString()
    }
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
    TabItem("首頁", Icons.Default.Home, Icons.Default.Home),
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
        animationSpec = tween(450, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "screen_alpha"
    )

    // Light theme colors matching HomeScreen
    val lightBg = Color(0xFFECFDF5)
    val lightAccent = Color(0xFF059669)
    val lightTextMid = Color(0xFF475569)
    val lightSurface = Color(0xFFFFFFFF)
    val lightBorder = Color(0xFFE1F2ED)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(lightBg)
            .graphicsLayer { alpha = screenAlpha }
    ) {
        Scaffold(
            containerColor = lightBg,
            topBar = {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(lightBg)
                        .statusBarsPadding()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = { rootNavController.navigate("settings") }) {
                        Icon(Icons.Default.Settings, "設定", tint = lightTextMid)
                    }
                    IconButton(onClick = {
                        val itemsJson = jsonFormatter.encodeToString(shoppingItems.filter { !it.isChecked })
                        rootNavController.currentBackStackEntry?.savedStateHandle?.set("shopping_list_json", itemsJson)
                        rootNavController.navigate("teammate_home")
                    }) {
                        Icon(Icons.Default.Place, "導覽", tint = lightTextMid)
                    }
                }
            },
            bottomBar = {
                CinemaBottomBar(
                    selectedTab = selectedTab,
                    onTabSelected = { selectedTab = it }
                )
            }
        ) { innerPadding ->
            Box(modifier = Modifier.padding(innerPadding)) {
                Crossfade(
                    targetState = selectedTab,
                    animationSpec = tween(300, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                    label = "tab_crossfade"
                ) { tab ->
                    when (tab) {
                        0 -> HomeScreen(
                            shoppingItems = shoppingItems,
                            onNavigateToList = { selectedTab = 1 },
                            onNavigateToBudget = { selectedTab = 3 }
                        )
                        1 -> ShoppingListScreen(items = shoppingItems, onItemsUpdate = { shoppingItems = it })
                        2 -> IngredientsScreen(records = dietRecords, onRecordsUpdate = { dietRecords = it })
                        3 -> BudgetScreen(
                            shoppingItems = shoppingItems,
                            budgetTotal = budgetTotalStr,
                            onBudgetUpdate = { budgetTotalStr = it },
                            onItemsUpdate = { shoppingItems = it }
                        )
                        4 -> HistoryScreen(shoppingItems = shoppingItems)
                        5 -> AIScreen(
                            shoppingItems = shoppingItems,
                            dietRecords = dietRecords,
                            budgetTotal = budgetTotalStr.toIntOrNull() ?: 0
                        )
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
    val navBg = Color(0xFFFFFFFF)
    val navBorder = Color(0xFFE1F2ED)

    Column {
        Spacer(
            modifier = Modifier
                .fillMaxWidth()
                .height(0.5.dp)
                .background(navBorder)
        )

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(navBg)
                .navigationBarsPadding()
                .padding(vertical = 6.dp),
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
    val navAccent = Color(0xFF059669)
    val navInactive = Color(0xFF94A3B8)
    val navAccentBg = Color(0xFFD1FAE5)

    val tint by animateColorAsState(
        targetValue = if (selected) navAccent else navInactive,
        animationSpec = tween(250),
        label = "nav_tint"
    )
    val scale by animateFloatAsState(
        targetValue = if (selected) 1.08f else 1f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessLow
        ),
        label = "nav_scale"
    )
    val bgAlpha by animateFloatAsState(
        targetValue = if (selected) 1f else 0f,
        animationSpec = tween(250),
        label = "nav_bg"
    )

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = Modifier
            .clickable(
                indication = null,
                interactionSource = remember { MutableInteractionSource() }
            ) { onClick() }
            .clip(RoundedCornerShape(12.dp))
            .background(navAccentBg.copy(alpha = bgAlpha * 0.6f), RoundedCornerShape(12.dp))
            .padding(horizontal = 14.dp, vertical = 6.dp)
    ) {
        Icon(
            icon,
            contentDescription = label,
            tint = tint,
            modifier = Modifier
                .size(22.dp)
                .scale(scale)
        )
        Spacer(modifier = Modifier.height(3.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelSmall,
            color = tint,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal
        )
        // Active indicator dot
        AnimatedVisibility(
            visible = selected,
            enter = scaleIn(animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy)) + fadeIn(),
            exit = scaleOut(animationSpec = tween(150)) + fadeOut()
        ) {
            Box(
                modifier = Modifier
                    .padding(top = 3.dp)
                    .size(4.dp)
                    .background(navAccent, CircleShape)
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

// ── Budget Filter Mode ──────────────────────────────────────

private enum class BudgetViewMode(val label: String) {
    CATEGORY("分類"),
    RECEIPT("收據"),
    DAY("日"),
    MONTH("月")
}

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

    val purchasedItems = remember(shoppingItems) { shoppingItems.filter { it.isChecked } }
    val totalSpent = remember(purchasedItems) { purchasedItems.sumOf { it.price * it.qty } }
    val budgetValue = budgetTotal.toIntOrNull() ?: 0

    var isProcessing by remember { mutableStateOf(false) }
    var showAddDialog by remember { mutableStateOf(false) }
    var showSettingsDialog by remember { mutableStateOf(false) }
    var extractedCount by remember { mutableIntStateOf(0) }
    var showSuccess by remember { mutableStateOf(false) }
    var viewMode by remember { mutableStateOf(BudgetViewMode.CATEGORY) }
    var expandedCategoryId by remember { mutableStateOf<String?>(null) }

    val groqApiKey = stringResource(id = R.string.groq_api_key)
    val paddleOcrApiUrl = stringResource(id = R.string.paddleocr_api_url)
    val paddleOcrToken = stringResource(id = R.string.paddleocr_access_token)
    val tempPhotoFile = remember { File(context.cacheDir, "receipt_photo.jpg") }
    val photoUri = remember { FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", tempPhotoFile) }

    val processImage: (Uri) -> Unit = { uri ->
        isProcessing = true
        scope.launch {
            val bitmap = withContext(Dispatchers.IO) { decodeUriToBitmap(context, uri) }
            if (bitmap != null) {
                try {
                    val rawText = callPaddleOcr(bitmap, paddleOcrApiUrl, paddleOcrToken)

                    if (rawText.isBlank()) {
                        isProcessing = false
                        Toast.makeText(context, "未能從圖片中識別文字", Toast.LENGTH_SHORT).show()
                        return@launch
                    }

                    Log.d("PaddleOCR", "Recognized text: $rawText")

                    // ── Enhanced Groq prompt for 全聯 / 家樂福 templates ──
                    val systemPrompt = """
                        You are a specialized Data Extraction and Budget Logic Engine for the "Smart AI Shopping Assistant" Android app.
                        Your sole responsibility is to process receipt/recipe images and output structured data for the Budget module.
                        The input text is in Traditional Chinese (繁體中文), recognized via OCR.

                        KNOWN RECEIPT TEMPLATES:
                        ───────────────────────
                        A) 全聯福利中心 (PX Mart)
                           • Header contains "全聯福利中心"
                           • Date line: YYYY/MM/DD format
                           • Items: product name, then price on same or next line
                           • Multiplier: qty * unit_price or "數量 × 單價"
                           • "TX" prefix = taxable indicator, strip it
                           • Footer: 合計 or total line, payment method, change

                        B) 家樂福 (Carrefour)
                           • Header may contain "家樂福" or store branch name
                           • "交易明細" = transaction detail section
                           • Date may use Minguo year (民國, e.g. 114 = 2025) or Western
                           • Items listed with qty and price columns
                           • "N" suffix = non-taxable, ignore the letter
                           • Footer: subtotal, tax lines, MASTER CARD / payment info

                        C) Generic receipts
                           • Look for store name at top, date near top
                           • Item lines with prices (rightmost number is usually price)
                           • Total / 合計 / 小計 at bottom

                        1. EXTRACTION RULES
                        - Use only information visible in the provided text.
                        - Date: Convert Minguo year (e.g. 114) to Western (2025) by adding 1911. Format: YYYY/MM/DD. If no date found, use today's date.
                        - Prices: Strip all "TX", "$", "N" symbols. Only store raw numbers.
                        - Multiplier Logic: Look for * or × symbol (e.g., 55 * 3 or 55×3). Calculate total_price = unit_price × qty.
                        - If no multiplier, qty = 1 and total_price = the item price.
                        - Language: Keep all item names in Traditional Chinese exactly as written.
                        - If the text is a recipe (食譜), extract each ingredient as a line item and estimate reasonable prices if not provided.
                        - IMPORTANT: Do NOT include subtotal lines, tax lines, change lines, or payment method lines as items.

                        2. CATEGORIZATION GUIDE
                        Sort every item into these categories ONLY:
                        - Food (食品): Meats, vegetables, fruits, dairy, eggs, snacks, cooking ingredients, frozen foods, bread, rice, noodles.
                        - Beverages (飲品): Milk, tea, coffee, soda, juice, water, alcoholic drinks.
                        - Groceries (生活用品): Cleaning supplies, toiletries, seasonings, kitchen tools, household items, tissue, detergent.
                        - Other (其他): Points, fees, bags, discounts, misc.

                        Return ONLY a JSON object with this structure:
                        {
                          "budget_entry": {
                            "store": "Store Name (e.g. 全聯福利中心 or 家樂福)",
                            "timestamp": "YYYY/MM/DD",
                            "total_amount": 0.0,
                            "line_items": [
                              {"name": "Item Name in Chinese", "cat": "Food/Beverages/Groceries/Other", "qty": 1, "total_price": 0.0}
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
                                    GroqMessage("user", "OCR Text (Traditional Chinese):\n$rawText")
                                ),
                                response_format = GroqResponseFormat()
                            )
                        )
                    }

                    val content = response.choices.firstOrNull()?.message?.content ?: ""
                    val tidied = jsonFormatter.decodeFromString<TidiedReceiptResponse>(content)

                    val receiptId = UUID.randomUUID().toString()
                    val storeName = tidied.budget_entry.store
                    val receiptTimestamp = try {
                        SimpleDateFormat("yyyy/MM/dd", Locale.getDefault()).parse(tidied.budget_entry.timestamp)?.time
                            ?: System.currentTimeMillis()
                    } catch (e: Exception) { System.currentTimeMillis() }

                    val newItems = shoppingItems.toMutableList()
                    tidied.budget_entry.line_items.forEach { line ->
                        newItems.add(
                            ShoppingItem(
                                id = UUID.randomUUID().toString(),
                                name = line.name,
                                price = (line.total_price ?: 0.0).toInt() / line.qty.coerceAtLeast(1),
                                qty = line.qty,
                                location = line.cat,
                                storeName = storeName,
                                isChecked = true,
                                createdAt = receiptTimestamp,
                                purchasedAt = receiptTimestamp,
                                receiptId = receiptId
                            )
                        )
                    }
                    onItemsUpdate(newItems)
                    extractedCount = tidied.budget_entry.line_items.size
                    showSuccess = true
                } catch (e: Exception) {
                    Log.e("PaddleOCR", "Error processing image", e)
                    Toast.makeText(context, "識別錯誤: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                } finally {
                    isProcessing = false
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

    val currentMonth = remember { SimpleDateFormat("yyyy年M月", Locale.getDefault()).format(Date()) }
    val currentMonthEn = remember { SimpleDateFormat("MMMM", Locale.ENGLISH).format(Date()) }

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
                            .clip(RoundedCornerShape(10.dp))
                            .background(Gold.copy(alpha = 0.1f))
                            .clickable { showAddDialog = true }
                            .padding(horizontal = 14.dp, vertical = 8.dp)
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
                                currentMonthEn,
                                style = MaterialTheme.typography.bodySmall,
                                color = TextTertiary
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── View Mode Tabs + Budget Chip ──
            StaggeredItem(index = 2) {
                LazyRow(
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 20.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(BudgetViewMode.entries.size) { idx ->
                        val mode = BudgetViewMode.entries[idx]
                        val isSelected = viewMode == mode
                        FilterChip(
                            selected = isSelected,
                            onClick = { viewMode = mode; expandedCategoryId = null },
                            label = { Text(mode.label) },
                            shape = RoundedCornerShape(20.dp),
                            colors = FilterChipDefaults.filterChipColors(
                                containerColor = SurfaceBase,
                                labelColor = TextSecondary,
                                selectedContainerColor = Gold.copy(alpha = 0.12f),
                                selectedLabelColor = Gold
                            ),
                            border = FilterChipDefaults.filterChipBorder(
                                borderColor = Border,
                                selectedBorderColor = Gold.copy(alpha = 0.3f),
                                enabled = true,
                                selected = isSelected
                            )
                        )
                    }
                    item {
                        FilterChip(
                            selected = false,
                            onClick = { showSettingsDialog = true },
                            label = { Text("預算: $$budgetTotal", color = TextTertiary) },
                            shape = RoundedCornerShape(20.dp),
                            colors = FilterChipDefaults.filterChipColors(containerColor = SurfaceBase),
                            border = FilterChipDefaults.filterChipBorder(
                                borderColor = Border,
                                enabled = true,
                                selected = false
                            )
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // ── Content Area (switches by viewMode) ──
            Crossfade(
                targetState = viewMode,
                animationSpec = tween(300, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                modifier = Modifier.weight(1f),
                label = "budget_view_crossfade"
            ) { mode ->
                when (mode) {
                    BudgetViewMode.CATEGORY -> BudgetCategoryView(
                        purchasedItems = purchasedItems,
                        budgetValue = budgetValue,
                        expandedCategoryId = expandedCategoryId,
                        onToggleCategory = { id ->
                            expandedCategoryId = if (expandedCategoryId == id) null else id
                        }
                    )
                    BudgetViewMode.RECEIPT -> BudgetReceiptView(purchasedItems = purchasedItems)
                    BudgetViewMode.DAY -> BudgetDayView(purchasedItems = purchasedItems)
                    BudgetViewMode.MONTH -> BudgetMonthView(purchasedItems = purchasedItems)
                }
            }

            // ── Processing / Success Feedback ──
            AnimatedVisibility(
                visible = isProcessing,
                enter = fadeIn(tween(300)) + slideInVertically(
                    initialOffsetY = { it / 2 },
                    animationSpec = tween(350, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f))
                ),
                exit = fadeOut(tween(200)) + slideOutVertically(
                    targetOffsetY = { it / 2 },
                    animationSpec = tween(250)
                )
            ) {
                Column {
                    Spacer(modifier = Modifier.height(8.dp))
                    ScanningOverlay()
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }

            AnimatedVisibility(
                visible = showSuccess,
                enter = fadeIn(tween(300)) + scaleIn(
                    initialScale = 0.85f,
                    animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy)
                ),
                exit = fadeOut(tween(200))
            ) {
                Column {
                    Spacer(modifier = Modifier.height(8.dp))
                    SuccessBanner(itemCount = extractedCount, onDismiss = { showSuccess = false })
                    Spacer(modifier = Modifier.height(8.dp))
                }
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
                        modifier = Modifier.fillMaxWidth().height(50.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                        shape = RoundedCornerShape(14.dp)
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
                        modifier = Modifier.fillMaxWidth().height(50.dp),
                        shape = RoundedCornerShape(14.dp),
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
                        shape = RoundedCornerShape(14.dp),
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
                    colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                    shape = RoundedCornerShape(12.dp)
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

// ═══════════════════════════════════════════════════════════════
// ── VIEW MODE: Category (with expandable drill-down) ──────────
// ═══════════════════════════════════════════════════════════════

@Composable
private fun BudgetCategoryView(
    purchasedItems: List<ShoppingItem>,
    budgetValue: Int,
    expandedCategoryId: String?,
    onToggleCategory: (String) -> Unit
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp)
    ) {
        items(dashboardCategories.size) { index ->
            val cat = dashboardCategories[index]
            val catItems = purchasedItems.filter { it.location == cat.catId }
            val catSpent = catItems.sumOf { it.price * it.qty }
            val catBudget = if (budgetValue > 0) budgetValue / dashboardCategories.size else 0
            val catLeft = catBudget - catSpent
            val isExpanded = expandedCategoryId == cat.catId

            StaggeredItem(index = index) {
                CategoryItem(
                    category = cat,
                    spent = catSpent,
                    left = catLeft,
                    budgetPerCategory = catBudget,
                    isExpanded = isExpanded,
                    items = catItems,
                    onToggle = { onToggleCategory(cat.catId) }
                )
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ── VIEW MODE: Receipt ────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

@Composable
private fun BudgetReceiptView(purchasedItems: List<ShoppingItem>) {
    val dateFormat = remember { SimpleDateFormat("yyyy/MM/dd", Locale.getDefault()) }

    // Group by receiptId, fall back to createdAt rounded to minute
    val receiptGroups = remember(purchasedItems) {
        purchasedItems.groupBy { item ->
            item.receiptId ?: "manual_${item.createdAt / 60000}"
        }.entries.sortedByDescending { entry ->
            entry.value.firstOrNull()?.createdAt ?: 0L
        }
    }

    var expandedReceiptId by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp)
    ) {
        items(receiptGroups.size) { index ->
            val (groupId, items) = receiptGroups[index]
            val storeName = items.firstOrNull()?.storeName ?: "手動輸入"
            val receiptDate = dateFormat.format(Date(items.first().createdAt))
            val receiptTotal = items.sumOf { it.price * it.qty }
            val isExpanded = expandedReceiptId == groupId

            StaggeredItem(index = index) {
                PressableSurface(
                    onClick = { expandedReceiptId = if (isExpanded) null else groupId },
                    backgroundColor = SurfaceBase,
                    glowColor = Gold.copy(alpha = 0.04f)
                ) {
                    Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(46.dp)
                                    .background(Gold.copy(alpha = 0.10f), RoundedCornerShape(14.dp)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    Icons.Default.Receipt,
                                    contentDescription = null,
                                    tint = Gold,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(14.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(storeName, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                                Text(receiptDate, style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                            }
                            Column(horizontalAlignment = Alignment.End) {
                                Text(
                                    "$${String.format("%,d", receiptTotal)}",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = TextPrimary
                                )
                                Text(
                                    "${items.size} 項",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextTertiary
                                )
                            }
                            Spacer(modifier = Modifier.width(4.dp))
                            val rotation by animateFloatAsState(
                                targetValue = if (isExpanded) 180f else 0f,
                                animationSpec = tween(300, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                                label = "receipt_chevron"
                            )
                            Icon(
                                Icons.Default.ExpandMore,
                                contentDescription = null,
                                tint = TextTertiary,
                                modifier = Modifier.size(20.dp).graphicsLayer { rotationZ = rotation }
                            )
                        }

                        // ── Expandable item list ──
                        AnimatedVisibility(
                            visible = isExpanded,
                            enter = expandVertically(
                                animationSpec = spring(dampingRatio = Spring.DampingRatioLowBouncy, stiffness = Spring.StiffnessMediumLow)
                            ) + fadeIn(tween(200)),
                            exit = shrinkVertically(tween(250)) + fadeOut(tween(150))
                        ) {
                            Column(modifier = Modifier.padding(top = 12.dp)) {
                                Spacer(
                                    modifier = Modifier.fillMaxWidth().height(1.dp).background(Border)
                                )
                                Spacer(modifier = Modifier.height(10.dp))
                                items.forEach { item ->
                                    ItemDetailRow(item)
                                }
                            }
                        }
                    }
                }
            }
        }

        if (receiptGroups.isEmpty()) {
            item {
                BudgetEmptyState("尚無收據紀錄", "掃描收據以開始追蹤")
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ── VIEW MODE: Day ────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

@Composable
private fun BudgetDayView(purchasedItems: List<ShoppingItem>) {
    val dayFormat = remember { SimpleDateFormat("yyyy/MM/dd", Locale.getDefault()) }
    val displayFormat = remember { SimpleDateFormat("M月d日 (E)", Locale.getDefault()) }

    val dayGroups = remember(purchasedItems) {
        purchasedItems.groupBy { dayFormat.format(Date(it.createdAt)) }
            .entries.sortedByDescending { it.key }
    }

    var expandedDay by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp)
    ) {
        items(dayGroups.size) { index ->
            val (dayKey, items) = dayGroups[index]
            val dayTotal = items.sumOf { it.price * it.qty }
            val displayDate = try {
                displayFormat.format(dayFormat.parse(dayKey)!!)
            } catch (e: Exception) { dayKey }
            val isExpanded = expandedDay == dayKey

            StaggeredItem(index = index) {
                PressableSurface(
                    onClick = { expandedDay = if (isExpanded) null else dayKey },
                    backgroundColor = SurfaceBase,
                    glowColor = ChartBlue.copy(alpha = 0.04f)
                ) {
                    Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(46.dp)
                                    .background(ChartBlue.copy(alpha = 0.10f), RoundedCornerShape(14.dp)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    Icons.Default.CalendarToday,
                                    contentDescription = null,
                                    tint = ChartBlue,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(14.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(displayDate, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                                Text("${items.size} 筆消費", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                            }
                            Text(
                                "$${String.format("%,d", dayTotal)}",
                                style = MaterialTheme.typography.titleMedium,
                                color = TextPrimary
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            val rotation by animateFloatAsState(
                                targetValue = if (isExpanded) 180f else 0f,
                                animationSpec = tween(300, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                                label = "day_chevron"
                            )
                            Icon(
                                Icons.Default.ExpandMore,
                                contentDescription = null,
                                tint = TextTertiary,
                                modifier = Modifier.size(20.dp).graphicsLayer { rotationZ = rotation }
                            )
                        }

                        AnimatedVisibility(
                            visible = isExpanded,
                            enter = expandVertically(
                                animationSpec = spring(dampingRatio = Spring.DampingRatioLowBouncy, stiffness = Spring.StiffnessMediumLow)
                            ) + fadeIn(tween(200)),
                            exit = shrinkVertically(tween(250)) + fadeOut(tween(150))
                        ) {
                            Column(modifier = Modifier.padding(top = 12.dp)) {
                                Spacer(modifier = Modifier.fillMaxWidth().height(1.dp).background(Border))
                                Spacer(modifier = Modifier.height(10.dp))
                                // Group by category within the day
                                val byCategory = items.groupBy { it.location ?: "Other" }
                                byCategory.forEach { (catId, catItems) ->
                                    val catInfo = dashboardCategories.find { it.catId == catId }
                                    val catTotal = catItems.sumOf { it.price * it.qty }
                                    Row(
                                        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .size(8.dp)
                                                .background(catInfo?.color ?: ChartViolet, CircleShape)
                                        )
                                        Spacer(modifier = Modifier.width(10.dp))
                                        Text(
                                            catInfo?.name ?: "其他",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = TextSecondary,
                                            modifier = Modifier.weight(1f)
                                        )
                                        Text(
                                            "$${String.format("%,d", catTotal)}",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = TextPrimary,
                                            fontWeight = FontWeight.Medium
                                        )
                                    }
                                }
                                Spacer(modifier = Modifier.height(8.dp))
                                Spacer(modifier = Modifier.fillMaxWidth().height(1.dp).background(Border))
                                Spacer(modifier = Modifier.height(8.dp))
                                items.forEach { item -> ItemDetailRow(item) }
                            }
                        }
                    }
                }
            }
        }

        if (dayGroups.isEmpty()) {
            item { BudgetEmptyState("尚無消費紀錄", "掃描收據以開始追蹤") }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ── VIEW MODE: Month ──────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

@Composable
private fun BudgetMonthView(purchasedItems: List<ShoppingItem>) {
    val monthKeyFormat = remember { SimpleDateFormat("yyyy-MM", Locale.getDefault()) }
    val displayMonthFormat = remember { SimpleDateFormat("yyyy年M月", Locale.getDefault()) }

    val monthGroups = remember(purchasedItems) {
        purchasedItems.groupBy { monthKeyFormat.format(Date(it.createdAt)) }
            .entries.sortedByDescending { it.key }
    }

    var expandedMonth by remember { mutableStateOf<String?>(null) }

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp)
    ) {
        items(monthGroups.size) { index ->
            val (monthKey, items) = monthGroups[index]
            val monthTotal = items.sumOf { it.price * it.qty }
            val displayMonth = try {
                displayMonthFormat.format(monthKeyFormat.parse(monthKey)!!)
            } catch (e: Exception) { monthKey }
            val isExpanded = expandedMonth == monthKey

            StaggeredItem(index = index) {
                PressableSurface(
                    onClick = { expandedMonth = if (isExpanded) null else monthKey },
                    backgroundColor = SurfaceBase,
                    glowColor = ChartGreen.copy(alpha = 0.04f)
                ) {
                    Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(46.dp)
                                    .background(ChartGreen.copy(alpha = 0.10f), RoundedCornerShape(14.dp)),
                                contentAlignment = Alignment.Center
                            ) {
                                Icon(
                                    Icons.Default.DateRange,
                                    contentDescription = null,
                                    tint = ChartGreen,
                                    modifier = Modifier.size(22.dp)
                                )
                            }
                            Spacer(modifier = Modifier.width(14.dp))
                            Column(modifier = Modifier.weight(1f)) {
                                Text(displayMonth, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                                Text("${items.size} 筆消費", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                            }
                            Text(
                                "$${String.format("%,d", monthTotal)}",
                                style = MaterialTheme.typography.titleMedium,
                                color = TextPrimary
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            val rotation by animateFloatAsState(
                                targetValue = if (isExpanded) 180f else 0f,
                                animationSpec = tween(300, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                                label = "month_chevron"
                            )
                            Icon(
                                Icons.Default.ExpandMore,
                                contentDescription = null,
                                tint = TextTertiary,
                                modifier = Modifier.size(20.dp).graphicsLayer { rotationZ = rotation }
                            )
                        }

                        AnimatedVisibility(
                            visible = isExpanded,
                            enter = expandVertically(
                                animationSpec = spring(dampingRatio = Spring.DampingRatioLowBouncy, stiffness = Spring.StiffnessMediumLow)
                            ) + fadeIn(tween(200)),
                            exit = shrinkVertically(tween(250)) + fadeOut(tween(150))
                        ) {
                            Column(modifier = Modifier.padding(top = 12.dp)) {
                                Spacer(modifier = Modifier.fillMaxWidth().height(1.dp).background(Border))
                                Spacer(modifier = Modifier.height(10.dp))

                                // Category breakdown bar chart
                                dashboardCategories.forEach { cat ->
                                    val catItems = items.filter { it.location == cat.catId }
                                    val catTotal = catItems.sumOf { it.price * it.qty }
                                    if (catTotal > 0) {
                                        val fraction = if (monthTotal > 0) catTotal.toFloat() / monthTotal else 0f
                                        val animatedFraction by animateFloatAsState(
                                            targetValue = fraction,
                                            animationSpec = tween(600, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                                            label = "month_bar_${cat.catId}"
                                        )
                                        Row(
                                            modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Text(
                                                cat.name,
                                                style = MaterialTheme.typography.bodySmall,
                                                color = TextSecondary,
                                                modifier = Modifier.width(56.dp)
                                            )
                                            Box(
                                                modifier = Modifier
                                                    .weight(1f)
                                                    .height(8.dp)
                                                    .clip(RoundedCornerShape(4.dp))
                                                    .background(SurfaceBright)
                                            ) {
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxWidth(animatedFraction)
                                                        .fillMaxHeight()
                                                        .clip(RoundedCornerShape(4.dp))
                                                        .background(cat.color)
                                                )
                                            }
                                            Spacer(modifier = Modifier.width(10.dp))
                                            Text(
                                                "$${String.format("%,d", catTotal)}",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = TextPrimary,
                                                fontWeight = FontWeight.Medium,
                                                modifier = Modifier.width(64.dp)
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (monthGroups.isEmpty()) {
            item { BudgetEmptyState("尚無消費紀錄", "掃描收據以開始追蹤") }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ── Shared: Item Detail Row ───────────────────────────────────
// ═══════════════════════════════════════════════════════════════

@Composable
private fun ItemDetailRow(item: ShoppingItem) {
    val catInfo = dashboardCategories.find { it.catId == item.location }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(6.dp)
                .background(catInfo?.color ?: TextTertiary, CircleShape)
        )
        Spacer(modifier = Modifier.width(10.dp))
        Text(
            item.name,
            style = MaterialTheme.typography.bodyMedium,
            color = TextPrimary,
            modifier = Modifier.weight(1f)
        )
        if (item.qty > 1) {
            Text(
                "×${item.qty}",
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary
            )
            Spacer(modifier = Modifier.width(8.dp))
        }
        Text(
            "$${String.format("%,d", item.price * item.qty)}",
            style = MaterialTheme.typography.bodyMedium,
            color = TextPrimary,
            fontWeight = FontWeight.Medium
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// ── Shared: Empty State ───────────────────────────────────────
// ═══════════════════════════════════════════════════════════════

@Composable
private fun BudgetEmptyState(title: String, subtitle: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 40.dp),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                Icons.Default.ReceiptLong,
                contentDescription = null,
                tint = TextTertiary.copy(alpha = 0.5f),
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(title, style = MaterialTheme.typography.titleSmall, color = TextTertiary)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = TextTertiary.copy(alpha = 0.7f))
        }
    }
}

// ── Category Row Item (with expandable drill-down) ─────────────

@Composable
fun CategoryItem(
    category: CategoryInfo,
    spent: Int,
    left: Int,
    budgetPerCategory: Int = 0,
    isExpanded: Boolean = false,
    items: List<ShoppingItem> = emptyList(),
    onToggle: () -> Unit = {}
) {
    val animatedSpent by animateIntAsState(spent)
    val progressFraction = if (budgetPerCategory > 0) (spent.toFloat() / budgetPerCategory).coerceIn(0f, 1.5f) else 0f
    val animatedProgress by animateFloatAsState(
        targetValue = progressFraction,
        animationSpec = tween(800, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "cat_progress"
    )
    val barColor by animateColorAsState(
        targetValue = if (left >= 0) category.color else Danger,
        animationSpec = tween(400),
        label = "cat_bar_color"
    )

    PressableSurface(
        onClick = onToggle,
        backgroundColor = SurfaceBase,
        glowColor = category.color.copy(alpha = 0.05f)
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(46.dp)
                        .background(category.color.copy(alpha = 0.10f), RoundedCornerShape(14.dp)),
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
                        "$${String.format("%,d", animatedSpent)}",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextPrimary
                    )
                    Text(
                        if (left >= 0) "$${String.format("%,d", left)} 剩餘" else "$${String.format("%,d", -left)} 超支",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (left >= 0) Success else Danger
                    )
                }

                Spacer(modifier = Modifier.width(4.dp))
                val rotation by animateFloatAsState(
                    targetValue = if (isExpanded) 180f else 0f,
                    animationSpec = tween(300, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
                    label = "cat_chevron"
                )
                Icon(
                    Icons.Default.ExpandMore,
                    contentDescription = null,
                    tint = TextTertiary,
                    modifier = Modifier.size(20.dp).graphicsLayer { rotationZ = rotation }
                )
            }

            // Animated spend progress bar
            if (budgetPerCategory > 0) {
                Spacer(modifier = Modifier.height(10.dp))
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp))
                        .background(SurfaceBright)
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth(animatedProgress.coerceAtMost(1f))
                            .fillMaxHeight()
                            .clip(RoundedCornerShape(2.dp))
                            .background(barColor)
                    )
                }
            }

            // ── Expandable Item Drill-Down ──
            AnimatedVisibility(
                visible = isExpanded,
                enter = expandVertically(
                    animationSpec = spring(dampingRatio = Spring.DampingRatioLowBouncy, stiffness = Spring.StiffnessMediumLow)
                ) + fadeIn(tween(200)),
                exit = shrinkVertically(tween(250)) + fadeOut(tween(150))
            ) {
                Column(modifier = Modifier.padding(top = 12.dp)) {
                    Spacer(
                        modifier = Modifier.fillMaxWidth().height(1.dp).background(Border)
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    if (items.isEmpty()) {
                        Text(
                            "尚無此分類的消費",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary,
                            modifier = Modifier.padding(vertical = 8.dp)
                        )
                    } else {
                        items.forEach { item -> ItemDetailRow(item) }
                    }
                }
            }
        }
    }
}

// ── Donut Chart ─────────────────────────────────────────────

@Composable
fun DonutChart(items: List<ShoppingItem>) {
    val total = items.filter { it.isChecked }.sumOf { it.price * it.qty }.toDouble()

    var animationPlayed by remember { mutableStateOf(false) }
    val animationProgress by animateFloatAsState(
        targetValue = if (animationPlayed) 1f else 0f,
        animationSpec = tween(900, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "donut_progress"
    )
    LaunchedEffect(Unit) { animationPlayed = true }

    Canvas(modifier = Modifier.fillMaxSize().padding(10.dp)) {
        if (total == 0.0) {
            drawCircle(color = SurfaceBright, style = Stroke(width = 14.dp.toPx(), cap = StrokeCap.Round))
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
                        style = Stroke(width = 14.dp.toPx(), cap = StrokeCap.Round)
                    )
                    startAngle += (catSpent / total * 360f).toFloat() * animationProgress
                }
            }
        }
    }
}

// ── Scanning Overlay ───────────────────────────────────────
// Cinematic OCR processing indicator with animated scan line.

@Composable
fun ScanningOverlay() {
    val infiniteTransition = rememberInfiniteTransition(label = "scan")

    // Scan line sweeps vertically
    val scanProgress by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1800, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scan_line"
    )

    // Pulse glow on the border
    val glowAlpha by infiniteTransition.animateFloat(
        initialValue = 0.15f,
        targetValue = 0.45f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scan_glow"
    )

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .height(80.dp)
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceBase)
            .drawBehind {
                // Pulsing border glow
                drawRoundRect(
                    brush = Brush.verticalGradient(
                        colors = listOf(
                            Gold.copy(alpha = glowAlpha),
                            Color.Transparent,
                            Gold.copy(alpha = glowAlpha * 0.5f)
                        )
                    ),
                    cornerRadius = CornerRadius(16.dp.toPx())
                )
                // Animated scan line
                val lineY = size.height * scanProgress
                drawLine(
                    brush = Brush.horizontalGradient(
                        colors = listOf(
                            Color.Transparent,
                            Gold.copy(alpha = 0.8f),
                            Gold,
                            Gold.copy(alpha = 0.8f),
                            Color.Transparent
                        )
                    ),
                    start = Offset(0f, lineY),
                    end = Offset(size.width, lineY),
                    strokeWidth = 2.dp.toPx()
                )
            },
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                color = Gold,
                strokeWidth = 2.dp
            )
            Text(
                "正在辨識圖片中...",
                style = MaterialTheme.typography.bodyMedium,
                color = Gold
            )
        }
    }
}

// ── Success Banner ─────────────────────────────────────────
// Animated success feedback after OCR extraction.

@Composable
fun SuccessBanner(itemCount: Int, onDismiss: () -> Unit) {
    var visible by remember { mutableStateOf(false) }
    val scale by animateFloatAsState(
        targetValue = if (visible) 1f else 0.85f,
        animationSpec = spring(
            dampingRatio = Spring.DampingRatioMediumBouncy,
            stiffness = Spring.StiffnessMedium
        ),
        label = "success_scale"
    )
    val alpha by animateFloatAsState(
        targetValue = if (visible) 1f else 0f,
        animationSpec = tween(350, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
        label = "success_alpha"
    )

    LaunchedEffect(Unit) {
        visible = true
        delay(2500)
        visible = false
        delay(350)
        onDismiss()
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .graphicsLayer {
                scaleX = scale
                scaleY = scale
                this.alpha = alpha
            }
            .clip(RoundedCornerShape(16.dp))
            .background(
                Brush.horizontalGradient(
                    colors = listOf(
                        Success.copy(alpha = 0.15f),
                        GoldSurface
                    )
                )
            )
            .padding(horizontal = 20.dp, vertical = 14.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Icon(
                Icons.Default.CheckCircle,
                contentDescription = null,
                tint = Success,
                modifier = Modifier.size(22.dp)
            )
            Text(
                "成功擷取 $itemCount 個項目",
                style = MaterialTheme.typography.titleSmall,
                color = TextPrimary
            )
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
