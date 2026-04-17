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
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.shopping.model.DietRecord
import com.example.shopping.model.UserProfile
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import androidx.compose.ui.res.stringResource
import com.example.shopping.R
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
fun IngredientsScreen(
    records: List<DietRecord>,
    onRecordsUpdate: (List<DietRecord>) -> Unit
) {
    val context = LocalContext.current
    val currentUser = UserProfile(name = "預設用戶", allergies = "")

    var recordDate by remember { mutableStateOf(SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())) }
    var foodName by remember { mutableStateOf("") }
    var ingredientText by remember { mutableStateOf("") }
    var expiryDate by remember { mutableStateOf("") }
    var unitCalorie by remember { mutableStateOf("") }
    var portion by remember { mutableStateOf("1.0") }
    var selectedCategory by remember { mutableStateOf("未分類") }
    var isProcessing by remember { mutableStateOf(false) }

    val categories = listOf("全榖雜糧", "豆魚蛋肉", "乳品類", "蔬菜類", "水果類", "油脂類")
    val filteredRecords = remember(records, recordDate) { records.filter { it.date == recordDate } }

    val currentTotalCal = ((unitCalorie.toDoubleOrNull() ?: 0.0) * (portion.toDoubleOrNull() ?: 0.0)).toInt()

    val detectedAllergens = remember(ingredientText, currentUser.allergies) {
        if (currentUser.allergies.isBlank()) emptyList<String>()
        else {
            val userAllergens = currentUser.allergies.split(",").map { it.trim() }
            userAllergens.filter { it.isNotBlank() && ingredientText.contains(it) }
        }
    }

    var selectedExercise by remember { mutableStateOf("慢跑") }
    val exercises = mapOf(
        "慢跑" to 8.1, "走路" to 3.5, "騎腳踏車" to 5.5, "健走" to 5.5, "爬樓梯" to 8.5
    )
    val exerciseMinutes = if (currentTotalCal > 0) (currentTotalCal / (exercises[selectedExercise] ?: 1.0)).toInt() else 0

    val scope = rememberCoroutineScope()
    val paddleOcrApiUrl = stringResource(id = R.string.paddleocr_api_url)
    val paddleOcrToken = stringResource(id = R.string.paddleocr_access_token)

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
                    } else {
                        Toast.makeText(context, "無法讀取圖片", Toast.LENGTH_SHORT).show()
                    }
                } catch (e: Exception) {
                    Log.e("OCR", "PaddleOCR 辨識失敗", e)
                    Toast.makeText(context, "辨識失敗: ${e.localizedMessage}", Toast.LENGTH_SHORT).show()
                } finally {
                    isProcessing = false
                }
            }
        }
    }

    val cameraPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) cameraLauncher.launch(imageUri)
    }

    val animatedCal by animateIntAsState(currentTotalCal)

    FadeInScreen {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(top = 8.dp, bottom = 80.dp)
        ) {
            // ── Input Form Card ──
            item {
                StaggeredItem(index = 0) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(SurfaceBase)
                            .padding(18.dp),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                "添加飲食紀錄",
                                style = MaterialTheme.typography.titleLarge,
                                color = TextPrimary
                            )
                            IconButton(onClick = {
                                if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                                    cameraLauncher.launch(imageUri)
                                } else {
                                    cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                                }
                            }) {
                                if (isProcessing) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(22.dp),
                                        strokeWidth = 2.dp,
                                        color = Gold
                                    )
                                } else {
                                    Icon(Icons.Default.CameraAlt, contentDescription = "OCR 辨識", tint = Gold)
                                }
                            }
                        }

                        OutlinedTextField(
                            value = foodName,
                            onValueChange = { foodName = it },
                            label = { Text("商品名稱 *", color = TextTertiary) },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(14.dp),
                            colors = cinemaTextFieldColors(),
                            singleLine = true
                        )

                        OutlinedTextField(
                            value = ingredientText,
                            onValueChange = { ingredientText = it },
                            label = { Text("成分表", color = TextTertiary) },
                            modifier = Modifier.fillMaxWidth(),
                            minLines = 2,
                            shape = RoundedCornerShape(14.dp),
                            colors = cinemaTextFieldColors(),
                            placeholder = { Text("可點擊上方相機圖標辨識包裝成分", color = TextDisabled) }
                        )

                        AnimatedVisibility(
                            visible = detectedAllergens.isNotEmpty(),
                            enter = fadeIn() + slideInVertically(),
                            exit = fadeOut()
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(DangerDim)
                                    .padding(12.dp)
                            ) {
                                Icon(Icons.Default.Warning, null, tint = Danger, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(8.dp))
                                Text(
                                    "警告：包含過敏原 (${detectedAllergens.joinToString(", ")})",
                                    color = Danger,
                                    style = MaterialTheme.typography.bodySmall,
                                    fontWeight = FontWeight.SemiBold
                                )
                            }
                        }

                        OutlinedTextField(
                            value = expiryDate,
                            onValueChange = { },
                            label = { Text("有效期限", color = TextTertiary) },
                            modifier = Modifier.fillMaxWidth(),
                            readOnly = true,
                            shape = RoundedCornerShape(14.dp),
                            colors = cinemaTextFieldColors(),
                            trailingIcon = {
                                IconButton(onClick = {
                                    val cal = Calendar.getInstance()
                                    DatePickerDialog(context, { _, y, m, d ->
                                        expiryDate = String.format(Locale.getDefault(), "%d-%02d-%02d", y, m + 1, d)
                                    }, cal.get(Calendar.YEAR), cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_MONTH)).show()
                                }) { Icon(Icons.Default.CalendarToday, null, tint = Gold) }
                            }
                        )

                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            OutlinedTextField(
                                value = unitCalorie,
                                onValueChange = { unitCalorie = it },
                                label = { Text("單份熱量", color = TextTertiary) },
                                modifier = Modifier.weight(1f),
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                shape = RoundedCornerShape(14.dp),
                                colors = cinemaTextFieldColors(),
                                singleLine = true
                            )
                            OutlinedTextField(
                                value = portion,
                                onValueChange = { portion = it },
                                label = { Text("份量", color = TextTertiary) },
                                modifier = Modifier.weight(1f),
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                                shape = RoundedCornerShape(14.dp),
                                colors = cinemaTextFieldColors(),
                                singleLine = true
                            )
                        }
                    }
                }
            }

            // ── Calorie Display ──
            item {
                StaggeredItem(index = 1) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(GoldSurface)
                            .padding(20.dp)
                    ) {
                        Column {
                            Text(
                                "計算總熱量",
                                style = MaterialTheme.typography.labelMedium,
                                color = GoldDim
                            )
                            Spacer(Modifier.height(4.dp))
                            Row(verticalAlignment = Alignment.Bottom) {
                                Text(
                                    "$animatedCal",
                                    style = MaterialTheme.typography.displayLarge,
                                    color = Gold
                                )
                                Spacer(Modifier.width(6.dp))
                                Text(
                                    "cal",
                                    style = MaterialTheme.typography.titleLarge,
                                    color = GoldDim,
                                    modifier = Modifier.padding(bottom = 6.dp)
                                )
                            }
                        }
                    }
                }
            }

            // ── Exercise Conversion ──
            item {
                StaggeredItem(index = 2) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .clip(RoundedCornerShape(20.dp))
                            .background(SuccessDim)
                            .padding(18.dp)
                    ) {
                        Text(
                            "運動消耗換算",
                            style = MaterialTheme.typography.titleMedium,
                            color = Success,
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(Modifier.height(12.dp))
                        LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            items(exercises.keys.toList()) { ex ->
                                FilterChip(
                                    selected = selectedExercise == ex,
                                    onClick = { selectedExercise = ex },
                                    label = { Text(ex, style = MaterialTheme.typography.labelMedium) },
                                    shape = RoundedCornerShape(20.dp),
                                    colors = FilterChipDefaults.filterChipColors(
                                        selectedContainerColor = Success.copy(alpha = 0.15f),
                                        selectedLabelColor = Success,
                                        containerColor = SurfaceBase,
                                        labelColor = TextSecondary
                                    )
                                )
                            }
                        }
                        Spacer(Modifier.height(16.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.Center,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("約需 $selectedExercise ", style = MaterialTheme.typography.bodyMedium, color = Success)
                            Text("$exerciseMinutes", style = MaterialTheme.typography.displaySmall, color = Success, fontWeight = FontWeight.Bold)
                            Text(" 分鐘", style = MaterialTheme.typography.bodyMedium, color = Success)
                        }
                    }
                }
            }

            // ── Save Button ──
            item {
                StaggeredItem(index = 3) {
                    Button(
                        onClick = {
                            if (foodName.isBlank()) return@Button
                            val newRecord = DietRecord(
                                date = recordDate,
                                name = foodName,
                                ingredients = ingredientText,
                                expiryDate = expiryDate,
                                unitCalorie = unitCalorie.toIntOrNull() ?: 0,
                                portion = portion.toDoubleOrNull() ?: 1.0,
                                totalCalories = currentTotalCal,
                                foodCategory = selectedCategory
                            )
                            onRecordsUpdate(listOf(newRecord) + records)
                            foodName = ""; ingredientText = ""; expiryDate = ""; unitCalorie = ""; portion = "1.0"
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .height(54.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                        shape = RoundedCornerShape(16.dp)
                    ) {
                        Text("儲存紀錄", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                    }
                }
            }

            // ── Today's Records ──
            itemsIndexed(filteredRecords) { index, record ->
                StaggeredItem(index = index + 4) {
                    DietRecordCard(record, onEdit = {}, onDelete = { onRecordsUpdate(records.filter { it.id != record.id }) })
                }
            }
        }
    }
}


@Composable
fun DietRecordCard(record: DietRecord, onEdit: () -> Unit, onDelete: () -> Unit) {
    PressableSurface(
        onClick = onEdit,
        modifier = Modifier.padding(horizontal = 20.dp),
        backgroundColor = SurfaceBase
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    record.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "${record.totalCalories} kcal  ·  ${record.date}",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }
            IconButton(onClick = onDelete, modifier = Modifier.size(36.dp)) {
                Icon(Icons.Default.Close, null, tint = TextDisabled, modifier = Modifier.size(16.dp))
            }
        }
    }
}
