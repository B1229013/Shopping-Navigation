package com.example.shopping.ui.screens

import android.app.DatePickerDialog
import android.content.Context
import android.text.format.Formatter
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.shopping.model.UserProfile
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(navController: NavController) {
    val context = LocalContext.current
    var cacheSize by remember { mutableStateOf(getCacheSize(context)) }
    var appSize by remember { mutableStateOf(getAppSize(context)) }

    // Load initial profile
    val profileFile = File(context.filesDir, "user_profile.json")
    val initialProfile = remember {
        if (profileFile.exists()) {
            try {
                Json.decodeFromString<UserProfile>(profileFile.readText())
            } catch (e: Exception) {
                UserProfile()
            }
        } else {
            UserProfile()
        }
    }

    var gender by remember { mutableStateOf(initialProfile.gender) }
    var birthday by remember { mutableStateOf(initialProfile.birthday) }
    var height by remember { mutableStateOf(initialProfile.height) }
    var weight by remember { mutableStateOf(initialProfile.weight) }
    var allergies by remember { mutableStateOf(initialProfile.allergies) }
    var disease by remember { mutableStateOf(initialProfile.disease) }
    var activityLevel by remember { mutableStateOf(initialProfile.activityLevel) }

    Scaffold(
        containerColor = Noir,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "設定",
                        style = MaterialTheme.typography.titleLarge,
                        color = TextPrimary
                    )
                },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回", tint = TextSecondary)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Noir)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(Modifier.height(4.dp))

            // ── Personal Profile ──
            StaggeredItem(index = 0) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(SurfaceBase)
                        .padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(14.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(38.dp)
                                .background(Gold.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Person, null, tint = Gold, modifier = Modifier.size(20.dp))
                        }
                        Spacer(Modifier.width(12.dp))
                        Text(
                            "個人資料",
                            style = MaterialTheme.typography.titleLarge,
                            color = TextPrimary
                        )
                    }

                    Text("性別", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        listOf("男", "女", "其他").forEach { option ->
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                RadioButton(
                                    selected = gender == option,
                                    onClick = { gender = option },
                                    colors = RadioButtonDefaults.colors(
                                        selectedColor = Gold,
                                        unselectedColor = TextTertiary
                                    )
                                )
                                Text(option, style = MaterialTheme.typography.bodyMedium, color = TextPrimary)
                                Spacer(Modifier.width(4.dp))
                            }
                        }
                    }

                    OutlinedTextField(
                        value = birthday,
                        onValueChange = { },
                        label = { Text("生日", color = TextTertiary) },
                        modifier = Modifier.fillMaxWidth(),
                        readOnly = true,
                        shape = RoundedCornerShape(12.dp),
                        colors = cinemaTextFieldColors(),
                        trailingIcon = {
                            IconButton(onClick = {
                                val cal = Calendar.getInstance()
                                // Parse current birthday if possible
                                try {
                                    val parts = birthday.split("-")
                                    if (parts.size == 3) {
                                        cal.set(Calendar.YEAR, parts[0].toInt())
                                        cal.set(Calendar.MONTH, parts[1].toInt() - 1)
                                        cal.set(Calendar.DAY_OF_MONTH, parts[2].toInt())
                                    }
                                } catch (e: Exception) {}

                                DatePickerDialog(context, { _, y, m, d ->
                                    birthday = "$y-${m + 1}-$d"
                                }, cal.get(Calendar.YEAR), cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_MONTH)).show()
                            }) { Icon(Icons.Default.CalendarMonth, null, tint = Gold) }
                        }
                    )

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        OutlinedTextField(
                            value = height,
                            onValueChange = { height = it },
                            label = { Text("身高 (cm)", color = TextTertiary) },
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp),
                            colors = cinemaTextFieldColors(),
                            singleLine = true
                        )
                        OutlinedTextField(
                            value = weight,
                            onValueChange = { weight = it },
                            label = { Text("體重 (kg)", color = TextTertiary) },
                            modifier = Modifier.weight(1f),
                            shape = RoundedCornerShape(12.dp),
                            colors = cinemaTextFieldColors(),
                            singleLine = true
                        )
                    }

                    Text("生活活動強度", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        listOf("低", "稍低", "適度", "高").forEach { level ->
                            FilterChip(
                                selected = activityLevel == level,
                                onClick = { activityLevel = level },
                                label = { Text(level) },
                                colors = FilterChipDefaults.filterChipColors(
                                    selectedContainerColor = Gold.copy(alpha = 0.2f),
                                    selectedLabelColor = Gold
                                )
                            )
                        }
                    }

                    var expandedDisease by remember { mutableStateOf(false) }
                    val diseases = listOf("無", "三酸甘油脂", "肥胖", "脂肪肝", "高血壓", "痛風", "腎功能不齊全", "糖尿病", "心臟疾病", "膽固醇", "膽結石")
                    
                    ExposedDropdownMenuBox(
                        expanded = expandedDisease,
                        onExpandedChange = { expandedDisease = it }
                    ) {
                        OutlinedTextField(
                            value = disease,
                            onValueChange = {},
                            readOnly = true,
                            label = { Text("疾病狀態", color = TextTertiary) },
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedDisease) },
                            modifier = Modifier.fillMaxWidth().menuAnchor(),
                            shape = RoundedCornerShape(12.dp),
                            colors = cinemaTextFieldColors()
                        )
                        ExposedDropdownMenu(
                            expanded = expandedDisease,
                            onDismissRequest = { expandedDisease = false }
                        ) {
                            diseases.forEach { d ->
                                DropdownMenuItem(
                                    text = { Text(d) },
                                    onClick = {
                                        disease = d
                                        expandedDisease = false
                                    }
                                )
                            }
                        }
                    }

                    OutlinedTextField(
                        value = allergies,
                        onValueChange = { allergies = it },
                        label = { Text("過敏原", color = TextTertiary) },
                        modifier = Modifier.fillMaxWidth(),
                        minLines = 2,
                        shape = RoundedCornerShape(12.dp),
                        colors = cinemaTextFieldColors()
                    )

                    Button(
                        onClick = {
                            val profile = UserProfile(
                                gender = gender,
                                birthday = birthday,
                                height = height,
                                weight = weight,
                                allergies = allergies,
                                disease = disease,
                                activityLevel = activityLevel
                            )
                            try {
                                profileFile.writeText(Json.encodeToString(profile))
                                Toast.makeText(context, "資料已儲存", Toast.LENGTH_SHORT).show()
                                appSize = getAppSize(context)
                            } catch (e: Exception) {
                                Toast.makeText(context, "儲存失敗: ${e.message}", Toast.LENGTH_SHORT).show()
                            }
                        },
                        modifier = Modifier.fillMaxWidth().height(48.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir)
                    ) {
                        Text("儲存修改", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                    }
                }
            }

            // ── Storage Info ──
            StaggeredItem(index = 1) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(SurfaceBase)
                        .padding(20.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(38.dp)
                                .background(Info.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(Icons.Default.Storage, null, tint = Info, modifier = Modifier.size(20.dp))
                        }
                        Spacer(Modifier.width(12.dp))
                        Text(
                            "儲存空間",
                            style = MaterialTheme.typography.titleLarge,
                            color = TextPrimary
                        )
                    }

                    StorageInfoRow(label = "應用程式資料", size = appSize)
                    CinemaDivider()
                    StorageInfoRow(label = "快取檔案", size = cacheSize)

                    Spacer(Modifier.height(4.dp))

                    Button(
                        onClick = {
                            clearCache(context)
                            cacheSize = getCacheSize(context)
                        },
                        modifier = Modifier.fillMaxWidth().height(48.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Danger.copy(alpha = 0.12f),
                            contentColor = Danger
                        ),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Icon(Icons.Default.DeleteSweep, null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("清除快取資料", style = MaterialTheme.typography.labelLarge)
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
fun StorageInfoRow(label: String, size: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
        Text(size, style = MaterialTheme.typography.titleSmall, color = TextPrimary)
    }
}

private fun getCacheSize(context: Context): String {
    val size = getFolderSize(context.cacheDir) + (context.externalCacheDir?.let { getFolderSize(it) } ?: 0L)
    return Formatter.formatFileSize(context, size)
}

private fun getAppSize(context: Context): String {
    val size = getFolderSize(context.filesDir)
    return Formatter.formatFileSize(context, size)
}

private fun getFolderSize(file: File): Long {
    var size = 0L
    if (file.isDirectory) {
        file.listFiles()?.forEach { size += getFolderSize(it) }
    } else {
        size = file.length()
    }
    return size
}

private fun clearCache(context: Context) {
    context.cacheDir.deleteRecursively()
    context.externalCacheDir?.deleteRecursively()
}
