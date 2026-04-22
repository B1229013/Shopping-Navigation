package com.example.shopping.ui.screens

import android.Manifest
import android.app.DatePickerDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.animateIntAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import coil.request.ImageRequest
import com.example.shopping.model.DietRecord
import com.example.shopping.model.UserProfile
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import com.example.shopping.ui.utils.getFoodIconUrl
import androidx.compose.ui.res.stringResource
import com.example.shopping.R
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

private enum class TimeRange(val label: String) {
    TODAY("當日"), WEEK("一週"), TWO_WEEKS("兩週"), MONTH("當月")
}

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
fun IngredientsScreen(
    records: List<DietRecord>,
    onRecordsUpdate: (List<DietRecord>) -> Unit
) {
    val context = LocalContext.current
    
    val profileFile = remember { File(context.filesDir, "user_profile.json") }
    val currentUser = remember(profileFile.lastModified()) {
        if (profileFile.exists()) {
            try {
                Json.decodeFromString<UserProfile>(profileFile.readText())
            } catch (e: Exception) {
                UserProfile(name = "預設用戶")
            }
        } else {
            UserProfile(name = "預設用戶")
        }
    }

    // ── Allergen Dictionary ──
    val allergenMap = mapOf(
        "牛奶" to listOf("milk", "dairy", "lactose", "乳粉", "乾酪", "奶油", "拿鐵", "拿鐵", "乳清"),
        "花生" to listOf("peanut", "arachis", "花生醬"),
        "雞蛋" to listOf("egg", "albumin", "蛋黃", "蛋白"),
        "小麥" to listOf("wheat", "gluten", "麵粉", "麩質"),
        "大豆" to listOf("soy", "soya", "豆腐", "豆奶", "黃豆"),
        "堅果" to listOf("nut", "almond", "cashew", "核桃", "腰果", "杏仁"),
        "海鮮" to listOf("seafood", "shrimp", "prawn", "crab", "fish", "蝦", "蟹", "魚")
    )

    var currentTimeRange by remember { mutableStateOf(TimeRange.TODAY) }
    var recordDate by remember { mutableStateOf(SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())) }
    
    // Form States
    var editingRecordId by remember { mutableStateOf<String?>(null) }
    var foodName by remember { mutableStateOf("") }
    var ingredientText by remember { mutableStateOf("") }
    var unitCalorie by remember { mutableStateOf("") }
    var portion by remember { mutableStateOf("1.0") }
    var carbs by remember { mutableStateOf("") }
    var sugar by remember { mutableStateOf("") }
    var protein by remember { mutableStateOf("") }
    var fat by remember { mutableStateOf("") }
    var sodium by remember { mutableStateOf("") }
    var cholesterol by remember { mutableStateOf("") }

    var isProcessing by remember { mutableStateOf(false) }

    // ── Recommended Calories Logic ──
    val age = try { currentUser.birthday.split("-")[0].toInt().let { Calendar.getInstance().get(Calendar.YEAR) - it } } catch (e: Exception) { 30 }
    val recommendedCal = when (currentUser.gender) {
        "男" -> when {
            age in 19..30 -> when(currentUser.activityLevel) { "低" -> 1850; "稍低" -> 2150; "適度" -> 2400; else -> 2700 }
            age in 31..50 -> when(currentUser.activityLevel) { "低" -> 1800; "稍低" -> 2100; "適度" -> 2400; else -> 2650 }
            else -> 2150
        }
        else -> when {
            age in 19..30 -> 1950
            else -> 1700
        }
    }

    // ── Date-based Calorie Summing for UI ──
    val dateToCalorieMap = remember(records) {
        records.groupBy { it.date }.mapValues { entry -> entry.value.sumOf { it.totalCalories } }
    }
    val currentSelectedDateTotal = dateToCalorieMap[recordDate] ?: 0

    // ── Filtered Records by Time Range ──
    val filteredRecords = remember(records, recordDate, currentTimeRange) {
        val sdf = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val targetDate = sdf.parse(recordDate) ?: Date()
        val calendar = Calendar.getInstance().apply { time = targetDate }
        
        val startTime = when (currentTimeRange) {
            TimeRange.TODAY -> calendar.apply { 
                set(Calendar.HOUR_OF_DAY, 0); set(Calendar.MINUTE, 0); set(Calendar.SECOND, 0) 
            }.timeInMillis
            TimeRange.WEEK -> calendar.apply { add(Calendar.DAY_OF_YEAR, -7) }.timeInMillis
            TimeRange.TWO_WEEKS -> calendar.apply { add(Calendar.DAY_OF_YEAR, -14) }.timeInMillis
            TimeRange.MONTH -> calendar.apply { add(Calendar.MONTH, -1) }.timeInMillis
        }
        
        records.filter { 
            val d = try { sdf.parse(it.date)?.time ?: 0L } catch(e: Exception) { 0L }
            d >= startTime && d <= targetDate.time
        }.sortedByDescending { it.id } 
    }

    // ── Aggregated Data for Pie Chart ──
    val aggregatedChartData = remember(filteredRecords) {
        filteredRecords.groupBy { it.name }
            .map { (name, list) -> name to list.sumOf { it.totalCalories } }
            .sortedByDescending { it.second }
    }

    val currentTotalCal = ((unitCalorie.toDoubleOrNull() ?: 0.0) * (portion.toDoubleOrNull() ?: 0.0)).toInt()

    // ── Exercise Calculation (METs) ──
    var selectedExercise by remember { mutableStateOf("慢跑 (8km/h)") }
    val exerciseMETs = mapOf(
        "慢跑 (8km/h)" to 8.2,
        "快走 (6km/h)" to 5.0,
        "走路" to 3.5,
        "騎腳踏車" to 4.0,
        "爬樓梯" to 8.0
    )
    val userWeight = currentUser.weight.toDoubleOrNull() ?: 60.0
    val exerciseMinutes = if (currentTotalCal > 0) {
        val met = exerciseMETs[selectedExercise] ?: 1.0
        ((currentTotalCal * 60.0) / (met * userWeight)).toInt()
    } else 0

    // ── Allergen & Disease Logic ──
    val detectedAllergens = remember(ingredientText, currentUser.allergies) {
        if (currentUser.allergies.isBlank()) emptyList<String>()
        else {
            val userAllergies = currentUser.allergies.split(",").map { it.trim() }
            val detected = mutableSetOf<String>()
            userAllergies.forEach { allergy ->
                if (ingredientText.contains(allergy, ignoreCase = true)) detected.add(allergy)
                allergenMap[allergy]?.forEach { alias ->
                    if (ingredientText.contains(alias, ignoreCase = true)) detected.add(allergy)
                }
            }
            detected.toList()
        }
    }

    val diseaseWarnings = remember(currentUser.disease, sugar, fat, sodium, protein, cholesterol) {
        val warnings = mutableListOf<String>()
        val sug = sugar.toDoubleOrNull() ?: 0.0
        val f = fat.toDoubleOrNull() ?: 0.0
        val s = sodium.toDoubleOrNull() ?: 0.0
        val p = protein.toDoubleOrNull() ?: 0.0
        val ch = cholesterol.toDoubleOrNull() ?: 0.0
        when (currentUser.disease) {
            "糖尿病" -> if (sug > 15) warnings.add("糖類量過高 (>15g)")
            "高血壓" -> {
                if (sug > 10) warnings.add("糖類量過高 (>10g)")
                if (s > 200) warnings.add("鈉含量過高 (>200mg)")
                if (ch > 80) warnings.add("膽固醇過高 (>80mg)")
            }
            "三酸甘油脂", "肥胖", "脂肪肝" -> {
                if (sug > 15) warnings.add("糖類量過高 (>15g)")
                if (f > 10) warnings.add("脂肪量過高 (>10g)")
            }
            "痛風", "膽結石" -> if (f > 10) warnings.add("脂肪量過高 (>10g)")
            "腎功能不齊全" -> if (p > 10) warnings.add("蛋白質過高 (>10g)")
            "心臟疾病" -> {
                if (f > 10) warnings.add("脂肪量過高 (>10g)")
                if (ch > 80) warnings.add("膽固醇過高 (>80mg)")
            }
            "膽固醇" -> if (ch > 80) warnings.add("膽固醇過高 (>80mg)")
        }
        warnings
    }

    val scope = rememberCoroutineScope()
    val paddleOcrApiUrl = stringResource(id = R.string.paddleocr_api_url)
    val paddleOcrToken = com.example.shopping.BuildConfig.PADDLEOCR_ACCESS_TOKEN
    val tempFile = remember { File(context.cacheDir, "ocr_scan.jpg") }
    val imageUri = remember { FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", tempFile) }

    val cameraLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        if (success) {
            isProcessing = true
            scope.launch {
                try {
                    val bitmap = withContext(Dispatchers.IO) { decodeUriToBitmap(context, imageUri) }
                    if (bitmap != null) {
                        val recognizedText = callPaddleOcr(bitmap, paddleOcrApiUrl, paddleOcrToken)
                        ingredientText = recognizedText.replace("\n", " ")
                    }
                } catch (e: Exception) {} finally { isProcessing = false }
            }
        }
    }

    val animatedCal by animateIntAsState(currentTotalCal)

    FadeInScreen {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(top = 8.dp, bottom = 80.dp)
        ) {
            item {
                Column(modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp)) {
                    Text("每日推薦攝取量：$recommendedCal kcal", color = Gold, fontWeight = FontWeight.Bold)
                    Text("疾病：${currentUser.disease}  ·  體重：${userWeight}kg", color = TextSecondary, style = MaterialTheme.typography.bodySmall)
                }
            }

            // ── Input Form ──
            item {
                StaggeredItem(index = 0) {
                    Column(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp).clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                            Text(if (editingRecordId == null) "添加紀錄" else "修改紀錄", style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                            IconButton(onClick = { cameraLauncher.launch(imageUri) }) {
                                if (isProcessing) CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Gold)
                                else Icon(Icons.Default.CameraAlt, null, tint = Gold)
                            }
                        }

                        // ── DATE SELECTION WITH TOTAL CALORIE BADGE ──
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
                            OutlinedTextField(
                                value = recordDate,
                                onValueChange = { },
                                label = { Text("日期") },
                                modifier = Modifier.weight(1f),
                                readOnly = true,
                                shape = RoundedCornerShape(12.dp),
                                colors = cinemaTextFieldColors(),
                                trailingIcon = {
                                    IconButton(onClick = {
                                        val cal = Calendar.getInstance()
                                        try {
                                            val parts = recordDate.split("-")
                                            cal.set(parts[0].toInt(), parts[1].toInt() - 1, parts[2].toInt())
                                        } catch(e: Exception) {}
                                        DatePickerDialog(context, { _, y, m, d ->
                                            recordDate = String.format(Locale.getDefault(), "%d-%02d-%02d", y, m + 1, d)
                                        }, cal.get(Calendar.YEAR), cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_MONTH)).show()
                                    }) { Icon(Icons.Default.CalendarToday, null, tint = Gold) }
                                }
                            )
                            
                            Spacer(Modifier.width(10.dp))
                            
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("本日攝取", style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                                Text(
                                    "$currentSelectedDateTotal",
                                    color = if (currentSelectedDateTotal > recommendedCal) Color(0xFFFF5252) else Color(0xFF4CAF50),
                                    fontWeight = FontWeight.Bold,
                                    style = MaterialTheme.typography.titleMedium
                                )
                                Text("kcal", style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                            }
                        }

                        OutlinedTextField(value = foodName, onValueChange = { foodName = it }, label = { Text("商品名稱 *") }, modifier = Modifier.fillMaxWidth(), colors = cinemaTextFieldColors(), singleLine = true)
                        OutlinedTextField(value = ingredientText, onValueChange = { ingredientText = it }, label = { Text("成分表 (自動偵測過敏原)") }, modifier = Modifier.fillMaxWidth(), colors = cinemaTextFieldColors())

                        if (detectedAllergens.isNotEmpty() || diseaseWarnings.isNotEmpty()) {
                            Column(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Danger.copy(alpha = 0.1f)).padding(10.dp)) {
                                detectedAllergens.forEach { Text("⚠️ 過敏原：$it", color = Danger, style = MaterialTheme.typography.bodySmall) }
                                diseaseWarnings.forEach { Text("❌ $it", color = Danger, style = MaterialTheme.typography.bodySmall) }
                            }
                        }

                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(value = unitCalorie, onValueChange = { unitCalorie = it }, label = { Text("熱量") }, modifier = Modifier.weight(1f), colors = cinemaTextFieldColors())
                            OutlinedTextField(value = protein, onValueChange = { protein = it }, label = { Text("蛋白質 (g)") }, modifier = Modifier.weight(1f), colors = cinemaTextFieldColors())
                            OutlinedTextField(value = fat, onValueChange = { fat = it }, label = { Text("脂肪 (g)") }, modifier = Modifier.weight(1f), colors = cinemaTextFieldColors())
                        }
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedTextField(value = carbs, onValueChange = { carbs = it }, label = { Text("碳水化合物 (g)") }, modifier = Modifier.weight(1f), colors = cinemaTextFieldColors())
                            OutlinedTextField(value = sugar, onValueChange = { sugar = it }, label = { Text("糖 (g)") }, modifier = Modifier.weight(1f), colors = cinemaTextFieldColors())
                            OutlinedTextField(value = sodium, onValueChange = { sodium = it }, label = { Text("鈉 (mg)") }, modifier = Modifier.weight(1f), colors = cinemaTextFieldColors())
                        }
                    }
                }
            }

            // ── Calorie Display & Exercise Conversion ──
            item {
                StaggeredItem(index = 1) {
                    Column(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp).clip(RoundedCornerShape(20.dp)).background(GoldSurface).padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Text("計算總熱量: $animatedCal kcal", style = MaterialTheme.typography.titleMedium, color = Gold)
                        HorizontalDivider(color = Gold.copy(alpha = 0.1f))
                        Text("運動消耗換算", style = MaterialTheme.typography.labelLarge, color = Gold, fontWeight = FontWeight.Bold)
                        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(exerciseMETs.keys.toList()) { ex ->
                                FilterChip(
                                    selected = selectedExercise == ex,
                                    onClick = { selectedExercise = ex },
                                    label = { Text(ex, fontSize = 12.sp) },
                                    colors = FilterChipDefaults.filterChipColors(selectedContainerColor = Gold.copy(alpha = 0.2f), selectedLabelColor = Gold)
                                )
                            }
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("約需執行 ", color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
                            Text("$exerciseMinutes", color = Gold, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                            Text(" 分鐘", color = TextSecondary, style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }

            // ── Save Button ──
            item {
                Button(
                    onClick = {
                        if (foodName.isBlank()) return@Button
                        val totalCal = unitCalorie.toIntOrNull() ?: 0
                        val newRecord = DietRecord(
                            id = editingRecordId ?: UUID.randomUUID().toString(),
                            date = recordDate,
                            name = foodName,
                            ingredients = ingredientText,
                            unitCalorie = totalCal,
                            totalCalories = totalCal, // ENSURE THIS IS SET
                            carbs = carbs.toDoubleOrNull() ?: 0.0,
                            sugar = sugar.toDoubleOrNull() ?: 0.0,
                            protein = protein.toDoubleOrNull() ?: 0.0,
                            fat = fat.toDoubleOrNull() ?: 0.0,
                            sodium = sodium.toDoubleOrNull() ?: 0.0
                        )
                        onRecordsUpdate(if (editingRecordId == null) listOf(newRecord) + records else records.map { if (it.id == editingRecordId) newRecord else it })
                        foodName = ""; ingredientText = ""; unitCalorie = ""; carbs = ""; sugar = ""; protein = ""; fat = ""; sodium = ""; editingRecordId = null
                    },
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp).height(54.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Text(if (editingRecordId == null) "儲存紀錄" else "更新紀錄", fontWeight = FontWeight.Bold)
                }
            }

            // ── Time Range Selector & Pie Chart ──
            item {
                Column(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
                    LazyRow(modifier = Modifier.fillMaxWidth(), contentPadding = PaddingValues(horizontal = 20.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        items(TimeRange.entries) { range ->
                            FilterChip(
                                selected = currentTimeRange == range,
                                onClick = { currentTimeRange = range },
                                label = { Text(range.label) },
                                colors = FilterChipDefaults.filterChipColors(selectedContainerColor = Gold.copy(alpha = 0.2f), selectedLabelColor = Gold)
                            )
                        }
                    }
                    
                    if (aggregatedChartData.isNotEmpty()) {
                        Spacer(Modifier.height(16.dp))
                        DietDetailedPieChart(aggregatedChartData)
                    }
                }
            }

            // ── History List ──
            itemsIndexed(filteredRecords) { index, record ->
                StaggeredItem(index = index + 5) {
                    DietRecordCard(record, onEdit = {
                        editingRecordId = record.id
                        recordDate = record.date
                        foodName = record.name
                        ingredientText = record.ingredients
                        unitCalorie = record.unitCalorie.toString()
                        carbs = record.carbs.toString()
                        sugar = record.sugar.toString()
                        protein = record.protein.toString()
                        fat = record.fat.toString()
                        sodium = record.sodium.toString()
                    }, onDelete = { onRecordsUpdate(records.filter { it.id != record.id }) })
                }
            }
        }
    }
}

@Composable
fun DietDetailedPieChart(data: List<Pair<String, Int>>) {
    val total = data.sumOf { it.second }.toFloat()
    val colors = listOf(ChartBlue, ChartGreen, ChartAmber, ChartViolet, Color(0xFFF43F5E), Color(0xFF10B981), Color(0xFFF59E0B), Color(0xFF8B5CF6))

    CinemaCard(modifier = Modifier.padding(horizontal = 20.dp)) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("熱量攝取比例", style = MaterialTheme.typography.titleMedium, color = TextPrimary)
            Spacer(Modifier.height(16.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(modifier = Modifier.size(150.dp), contentAlignment = Alignment.Center) {
                    Canvas(modifier = Modifier.size(140.dp)) {
                        var startAngle = -90f
                        data.forEachIndexed { i, pair ->
                            val sweepAngle = (pair.second / total) * 360f
                            drawArc(
                                color = colors[i % colors.size],
                                startAngle = startAngle,
                                sweepAngle = sweepAngle,
                                useCenter = false,
                                style = Stroke(24.dp.toPx(), cap = StrokeCap.Butt)
                            )
                            startAngle += sweepAngle
                        }
                    }
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("總計", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                        Text("${total.toInt()}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = TextPrimary)
                        Text("kcal", style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                    }
                }
                
                Spacer(Modifier.width(20.dp))
                
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    data.take(5).forEachIndexed { i, pair ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(10.dp).background(colors[i % colors.size], CircleShape))
                            Spacer(Modifier.width(8.dp))
                            Text("${pair.first}: ${pair.second} kcal", style = MaterialTheme.typography.bodySmall, color = TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                    if (data.size > 5) {
                        Text("...以及其他 ${data.size - 5} 項", style = MaterialTheme.typography.labelSmall, color = TextTertiary)
                    }
                }
            }
        }
    }
}

@Composable
fun DietRecordCard(record: DietRecord, onEdit: () -> Unit, onDelete: () -> Unit) {
    val context = LocalContext.current
    val iconUrl = remember(record.name) { getFoodIconUrl(record.name) }
    val initial = remember(record.name) { record.name.firstOrNull()?.toString() ?: "?" }
    val avatarTint = remember(record.name) {
        val palette = listOf(
            Color(0xFFEF4444), Color(0xFFF59E0B), Color(0xFF10B981),
            Color(0xFF3B82F6), Color(0xFF8B5CF6), Color(0xFFEC4899),
            Color(0xFF06B6D4), Color(0xFF84CC16)
        )
        palette[(record.name.hashCode().ushr(1)) % palette.size]
    }
    PressableSurface(onClick = onEdit, modifier = Modifier.padding(horizontal = 20.dp), backgroundColor = SurfaceBase) {
        Row(modifier = Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(44.dp)
                    .clip(CircleShape)
                    .background(if (iconUrl != null) Gold.copy(alpha = 0.12f) else avatarTint.copy(alpha = 0.18f)),
                contentAlignment = Alignment.Center
            ) {
                if (iconUrl != null) {
                    AsyncImage(
                        model = ImageRequest.Builder(context).data(iconUrl).crossfade(true).build(),
                        contentDescription = record.name,
                        contentScale = androidx.compose.ui.layout.ContentScale.Crop,
                        modifier = Modifier.fillMaxSize().clip(CircleShape)
                    )
                } else {
                    Text(
                        text = initial,
                        style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold),
                        color = avatarTint
                    )
                }
            }
            Spacer(Modifier.width(12.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(record.name, style = MaterialTheme.typography.titleMedium, color = TextPrimary)
                Text("${record.date}  ·  ${record.totalCalories} kcal", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                Text("糖:${record.sugar}g 鈉:${record.sodium}mg 蛋:${record.protein}g 脂:${record.fat}g", style = MaterialTheme.typography.bodySmall, color = Gold.copy(alpha = 0.7f))
            }
            IconButton(onClick = onDelete) { Icon(Icons.Default.Close, null, tint = TextDisabled, modifier = Modifier.size(16.dp)) }
        }
    }
}
