package com.example.shopping.ui.screens

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.os.Build
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.shopping.ui.components.StaggeredItem
import com.example.shopping.ui.theme.*
import kotlinx.coroutines.delay
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

// ── Sensor data record ──

private data class SensorRecord(
    val timestampMs: Long,
    val elapsedSec: Double,
    val totalSteps: Int,
    val stepDetected: Int,
    val accelX: Float, val accelY: Float, val accelZ: Float,
    val gyroX: Float, val gyroY: Float, val gyroZ: Float,
    val magX: Float, val magY: Float, val magZ: Float,
    val heading: Float,
    val rotX: Float, val rotY: Float, val rotZ: Float, val rotW: Float,
    val pressure: Float,
    val light: Float,
)

// ── CSV export ──

private fun exportCsv(context: Context, records: List<SensorRecord>, groundTruth: GroundTruth): File? {
    if (records.isEmpty()) return null
    val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val dir = File(context.getExternalFilesDir(null), "sensor_lab")
    dir.mkdirs()
    val file = File(dir, "sensor_log_$ts.csv")

    file.bufferedWriter().use { w ->
        w.appendLine("# SensorLab Export — $ts")
        w.appendLine("# ground_truth_steps=${groundTruth.steps},ground_truth_distance_m=${groundTruth.distanceM},ground_truth_turns=${groundTruth.turns}")
        w.appendLine("timestamp_ms,elapsed_sec,total_steps,step_detected,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,heading_deg,rot_x,rot_y,rot_z,rot_w,pressure_hpa,light_lux")
        for (r in records) {
            w.appendLine(
                "${r.timestampMs},%.3f,%d,%d,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.1f,%.6f,%.6f,%.6f,%.6f,%.2f,%.1f".format(
                    r.elapsedSec, r.totalSteps, r.stepDetected,
                    r.accelX, r.accelY, r.accelZ,
                    r.gyroX, r.gyroY, r.gyroZ,
                    r.magX, r.magY, r.magZ,
                    r.heading,
                    r.rotX, r.rotY, r.rotZ, r.rotW,
                    r.pressure, r.light,
                )
            )
        }
    }
    return file
}

private data class GroundTruth(
    val steps: String = "",
    val distanceM: String = "",
    val turns: String = "",
)

// ── Saved log metadata ──

private data class LogFileInfo(
    val file: File,
    val name: String,
    val date: String,
    val sizeKb: String,
    val dataPoints: Int,
    val groundTruthSteps: String,
    val groundTruthDistance: String,
)

private fun listSavedLogs(context: Context): List<LogFileInfo> {
    val dir = File(context.getExternalFilesDir(null), "sensor_lab")
    if (!dir.exists()) return emptyList()
    return dir.listFiles { f -> f.extension == "csv" }
        ?.sortedByDescending { it.lastModified() }
        ?.map { f ->
            val lines = f.readLines()
            val gtLine = lines.getOrNull(1) ?: ""
            val gtSteps = Regex("ground_truth_steps=([^,]*)").find(gtLine)?.groupValues?.get(1) ?: ""
            val gtDist = Regex("ground_truth_distance_m=([^,]*)").find(gtLine)?.groupValues?.get(1) ?: ""
            val dataLines = lines.count { !it.startsWith("#") && it.contains(",") } - 1
            LogFileInfo(
                file = f,
                name = f.nameWithoutExtension.removePrefix("sensor_log_"),
                date = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.US).format(Date(f.lastModified())),
                sizeKb = "%.1f KB".format(f.length() / 1024.0),
                dataPoints = dataLines.coerceAtLeast(0),
                groundTruthSteps = gtSteps,
                groundTruthDistance = gtDist,
            )
        } ?: emptyList()
}

private fun shareFile(context: Context, file: File) {
    val uri = FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        file
    )
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "text/csv"
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_SUBJECT, "SensorLab — ${file.nameWithoutExtension}")
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "分享感測器記錄"))
}

// ── Main composable ──

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SensorLabScreen(navController: NavController) {
    val context = LocalContext.current
    val sensorManager = remember { context.getSystemService(Context.SENSOR_SERVICE) as SensorManager }

    // Request ACTIVITY_RECOGNITION for step sensors on Android 10+
    var activityPermissionGranted by remember {
        mutableStateOf(
            Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            context.checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        activityPermissionGranted = granted
    }
    LaunchedEffect(Unit) {
        if (!activityPermissionGranted) {
            permissionLauncher.launch(Manifest.permission.ACTIVITY_RECOGNITION)
        }
    }

    // Sensor availability
    val availability = remember(activityPermissionGranted) { checkSensorAvailability(sensorManager) }

    // Recording state
    var isRecording by remember { mutableStateOf(false) }
    var recordStartTime by remember { mutableLongStateOf(0L) }
    var elapsedSeconds by remember { mutableDoubleStateOf(0.0) }
    val records = remember { mutableStateListOf<SensorRecord>() }

    // Live values
    var stepCount by remember { mutableIntStateOf(0) }
    var stepBaseline by remember { mutableIntStateOf(-1) }
    var accel by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var gyro by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var mag by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var heading by remember { mutableFloatStateOf(0f) }
    var stepDetectedCount by remember { mutableIntStateOf(0) }
    var rotationVec by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f, 1f)) }
    var pressure by remember { mutableFloatStateOf(0f) }
    var light by remember { mutableFloatStateOf(0f) }

    // Ground truth inputs
    var groundTruth by remember { mutableStateOf(GroundTruth()) }

    // Saved logs
    var savedLogs by remember { mutableStateOf(listSavedLogs(context)) }

    // Elapsed timer
    LaunchedEffect(isRecording) {
        while (isRecording) {
            elapsedSeconds = (SystemClock.elapsedRealtime() - recordStartTime) / 1000.0
            delay(100)
        }
    }

    // Sensor listener — re-register when permission changes
    DisposableEffect(activityPermissionGranted) {
        val gravity = floatArrayOf(0f, 0f, 0f)
        val geomagnetic = floatArrayOf(0f, 0f, 0f)
        val rotationMatrix = FloatArray(9)
        val orientation = FloatArray(3)

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        accel = event.values.copyOf()
                        gravity[0] = event.values[0]
                        gravity[1] = event.values[1]
                        gravity[2] = event.values[2]
                    }
                    Sensor.TYPE_GYROSCOPE -> {
                        gyro = event.values.copyOf()
                    }
                    Sensor.TYPE_MAGNETIC_FIELD -> {
                        mag = event.values.copyOf()
                        geomagnetic[0] = event.values[0]
                        geomagnetic[1] = event.values[1]
                        geomagnetic[2] = event.values[2]
                    }
                    Sensor.TYPE_STEP_COUNTER -> {
                        val raw = event.values[0].toInt()
                        if (stepBaseline < 0) stepBaseline = raw
                        stepCount = raw - stepBaseline
                    }
                    Sensor.TYPE_PRESSURE -> {
                        pressure = event.values[0]
                    }
                    Sensor.TYPE_LIGHT -> {
                        light = event.values[0]
                    }
                    Sensor.TYPE_STEP_DETECTOR -> {
                        if (isRecording) stepDetectedCount++
                    }
                    Sensor.TYPE_ROTATION_VECTOR -> {
                        val v = event.values
                        val w = if (v.size > 3) v[3]
                                else kotlin.math.sqrt(
                                    (1f - v[0]*v[0] - v[1]*v[1] - v[2]*v[2]).coerceAtLeast(0f)
                                )
                        rotationVec = floatArrayOf(v[0], v[1], v[2], w)
                    }
                }

                // Compute heading from accelerometer + magnetometer
                if (SensorManager.getRotationMatrix(rotationMatrix, null, gravity, geomagnetic)) {
                    SensorManager.getOrientation(rotationMatrix, orientation)
                    heading = Math.toDegrees(orientation[0].toDouble()).toFloat()
                    if (heading < 0) heading += 360f
                }

                // Record at ~10 Hz when recording
                if (isRecording && event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                    val now = SystemClock.elapsedRealtime()
                    val elapsed = (now - recordStartTime) / 1000.0
                    if (records.isEmpty() || elapsed - records.last().elapsedSec >= 0.1) {
                        records.add(
                            SensorRecord(
                                timestampMs = System.currentTimeMillis(),
                                elapsedSec = elapsed,
                                totalSteps = stepCount,
                                stepDetected = stepDetectedCount,
                                accelX = accel[0], accelY = accel[1], accelZ = accel[2],
                                gyroX = gyro[0], gyroY = gyro[1], gyroZ = gyro[2],
                                magX = mag[0], magY = mag[1], magZ = mag[2],
                                heading = heading,
                                rotX = rotationVec[0], rotY = rotationVec[1],
                                rotZ = rotationVec[2], rotW = rotationVec[3],
                                pressure = pressure,
                                light = light,
                            )
                        )
                    }
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        val rate = SensorManager.SENSOR_DELAY_GAME // ~50Hz
        listOf(
            Sensor.TYPE_ACCELEROMETER, Sensor.TYPE_GYROSCOPE,
            Sensor.TYPE_MAGNETIC_FIELD, Sensor.TYPE_STEP_COUNTER,
            Sensor.TYPE_STEP_DETECTOR, Sensor.TYPE_ROTATION_VECTOR,
            Sensor.TYPE_PRESSURE, Sensor.TYPE_LIGHT,
        ).forEach { type ->
            sensorManager.getDefaultSensor(type)?.let {
                sensorManager.registerListener(listener, it, rate)
            }
        }

        onDispose {
            sensorManager.unregisterListener(listener)
        }
    }

    Scaffold(
        containerColor = Noir,
        topBar = {
            TopAppBar(
                title = {
                    Text("SensorLab", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
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

            // ── Sensor availability ──
            StaggeredItem(index = 0) {
                SensorAvailabilityCard(availability)
            }

            // ── Live readings ──
            StaggeredItem(index = 1) {
                LiveReadingsCard(
                    stepCount = stepCount,
                    stepDetected = stepDetectedCount,
                    heading = heading,
                    accel = accel,
                    gyro = gyro,
                    mag = mag,
                    rotationVec = rotationVec,
                    pressure = pressure,
                    light = light,
                )
            }

            // ── Recording control ──
            StaggeredItem(index = 2) {
                RecordingCard(
                    isRecording = isRecording,
                    elapsedSeconds = elapsedSeconds,
                    recordCount = records.size,
                    stepCount = stepCount,
                    onStart = {
                        records.clear()
                        stepBaseline = -1
                        stepCount = 0
                        stepDetectedCount = 0
                        recordStartTime = SystemClock.elapsedRealtime()
                        isRecording = true
                    },
                    onStop = {
                        isRecording = false
                    },
                )
            }

            // ── Ground truth + export ──
            StaggeredItem(index = 3) {
                GroundTruthCard(
                    groundTruth = groundTruth,
                    onUpdate = { groundTruth = it },
                    recordCount = records.size,
                    onExport = {
                        val file = exportCsv(context, records.toList(), groundTruth)
                        if (file != null) {
                            savedLogs = listSavedLogs(context)
                            Toast.makeText(context, "已匯出: ${file.name}", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "無記錄可匯出", Toast.LENGTH_SHORT).show()
                        }
                    }
                )
            }

            // ── Summary stats (after recording) ──
            if (!isRecording && records.isNotEmpty()) {
                StaggeredItem(index = 4) {
                    SummaryCard(records.toList())
                }
            }

            // ── History ──
            if (savedLogs.isNotEmpty()) {
                StaggeredItem(index = 5) {
                    HistoryCard(
                        logs = savedLogs,
                        onShare = { shareFile(context, it.file) },
                        onDelete = { info ->
                            info.file.delete()
                            savedLogs = listSavedLogs(context)
                            Toast.makeText(context, "已刪除 ${info.name}", Toast.LENGTH_SHORT).show()
                        },
                    )
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

// ── Sensor availability check ──

private data class SensorStatus(val name: String, val type: Int, val available: Boolean)

private fun checkSensorAvailability(sm: SensorManager): List<SensorStatus> {
    return listOf(
        SensorStatus("加速度計", Sensor.TYPE_ACCELEROMETER, sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) != null),
        SensorStatus("陀螺儀", Sensor.TYPE_GYROSCOPE, sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE) != null),
        SensorStatus("磁力計", Sensor.TYPE_MAGNETIC_FIELD, sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD) != null),
        SensorStatus("計步器", Sensor.TYPE_STEP_COUNTER, sm.getDefaultSensor(Sensor.TYPE_STEP_COUNTER) != null),
        SensorStatus("步伐偵測器", Sensor.TYPE_STEP_DETECTOR, sm.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR) != null),
        SensorStatus("旋轉向量", Sensor.TYPE_ROTATION_VECTOR, sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR) != null),
        SensorStatus("氣壓計", Sensor.TYPE_PRESSURE, sm.getDefaultSensor(Sensor.TYPE_PRESSURE) != null),
        SensorStatus("光線感測器", Sensor.TYPE_LIGHT, sm.getDefaultSensor(Sensor.TYPE_LIGHT) != null),
    )
}

// ── UI Cards ──

@Composable
private fun SensorAvailabilityCard(sensors: List<SensorStatus>) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceBase)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(38.dp)
                    .background(Info.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Sensors, null, tint = Info, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Text("感測器狀態", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        sensors.forEach { s ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(s.name, style = MaterialTheme.typography.bodyMedium, color = TextPrimary)
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(if (s.available) Success else Danger)
                    )
                    Text(
                        if (s.available) "可用" else "不可用",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (s.available) Success else Danger,
                    )
                }
            }
        }
    }
}

@Composable
private fun LiveReadingsCard(
    stepCount: Int,
    stepDetected: Int,
    heading: Float,
    accel: FloatArray,
    gyro: FloatArray,
    mag: FloatArray,
    rotationVec: FloatArray,
    pressure: Float,
    light: Float,
) {
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
                    .background(Gold.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Speed, null, tint = Gold, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Text("即時讀數", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        // Big stats row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
        ) {
            BigStat(value = "$stepCount", label = "步數")
            BigStat(value = "%.0f°".format(heading), label = "朝向")
            BigStat(value = if (pressure > 0) "%.1f".format(pressure) else "—", label = "氣壓 hPa")
        }

        Divider(color = Border)

        // Detailed readings
        SensorRow("加速度 (m/s²)", "X=%.2f  Y=%.2f  Z=%.2f".format(accel[0], accel[1], accel[2]))
        SensorRow("陀螺儀 (rad/s)", "X=%.4f  Y=%.4f  Z=%.4f".format(gyro[0], gyro[1], gyro[2]))
        SensorRow("磁力 (μT)", "X=%.1f  Y=%.1f  Z=%.1f".format(mag[0], mag[1], mag[2]))
        SensorRow("步伐偵測", "$stepDetected 次")
        SensorRow("旋轉向量", "X=%.3f Y=%.3f Z=%.3f W=%.3f".format(rotationVec[0], rotationVec[1], rotationVec[2], rotationVec[3]))
        SensorRow("光線 (lux)", "%.0f".format(light))
    }
}

@Composable
private fun BigStat(value: String, label: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            value,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            color = TextPrimary,
        )
        Text(label, style = MaterialTheme.typography.labelSmall, color = TextTertiary)
    }
}

@Composable
private fun SensorRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        Text(
            value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            color = TextPrimary,
        )
    }
}

@Composable
private fun RecordingCard(
    isRecording: Boolean,
    elapsedSeconds: Double,
    recordCount: Int,
    stepCount: Int,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(if (isRecording) Danger.copy(alpha = 0.05f) else SurfaceBase)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(38.dp)
                    .background(
                        (if (isRecording) Danger else ChartAmber).copy(alpha = 0.10f),
                        RoundedCornerShape(12.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    if (isRecording) Icons.Default.Stop else Icons.Default.FiberManualRecord,
                    null,
                    tint = if (isRecording) Danger else ChartAmber,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("記錄控制", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
                if (isRecording) {
                    Text("記錄中 — 請開始行走", style = MaterialTheme.typography.bodySmall, color = Danger)
                }
            }
        }

        if (isRecording || recordCount > 0) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                BigStat(
                    value = "%d:%02d".format((elapsedSeconds / 60).toInt(), (elapsedSeconds % 60).toInt()),
                    label = "時間"
                )
                BigStat(value = "$recordCount", label = "資料點")
                BigStat(value = "$stepCount", label = "步數")
            }
        }

        Button(
            onClick = { if (isRecording) onStop() else onStart() },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isRecording) Danger else Gold,
            ),
            shape = RoundedCornerShape(14.dp),
        ) {
            Icon(
                if (isRecording) Icons.Default.Stop else Icons.Default.PlayArrow,
                null,
                modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(
                if (isRecording) "停止記錄" else "開始記錄",
                fontWeight = FontWeight.SemiBold,
            )
        }

        if (!isRecording && recordCount == 0) {
            Text(
                "按下「開始記錄」後走一段已知距離的路線，\n停止後輸入實際步數和距離，匯出 CSV 分析準確率。",
                style = MaterialTheme.typography.bodySmall,
                color = TextTertiary,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
private fun GroundTruthCard(
    groundTruth: GroundTruth,
    onUpdate: (GroundTruth) -> Unit,
    recordCount: Int,
    onExport: () -> Unit,
) {
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
                    .background(ChartViolet.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Straighten, null, tint = ChartViolet, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Text("Ground Truth", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        Text(
            "走完後填入實際數值，會一併寫入 CSV 方便比對",
            style = MaterialTheme.typography.bodySmall,
            color = TextTertiary,
        )

        OutlinedTextField(
            value = groundTruth.steps,
            onValueChange = { onUpdate(groundTruth.copy(steps = it)) },
            label = { Text("實際步數") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Gold,
                cursorColor = Gold,
                focusedLabelColor = Gold,
            ),
        )
        OutlinedTextField(
            value = groundTruth.distanceM,
            onValueChange = { onUpdate(groundTruth.copy(distanceM = it)) },
            label = { Text("實際距離 (公尺)") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Gold,
                cursorColor = Gold,
                focusedLabelColor = Gold,
            ),
        )
        OutlinedTextField(
            value = groundTruth.turns,
            onValueChange = { onUpdate(groundTruth.copy(turns = it)) },
            label = { Text("轉彎次數") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Gold,
                cursorColor = Gold,
                focusedLabelColor = Gold,
            ),
        )

        Button(
            onClick = onExport,
            modifier = Modifier.fillMaxWidth(),
            enabled = recordCount > 0,
            colors = ButtonDefaults.buttonColors(containerColor = Info),
            shape = RoundedCornerShape(14.dp),
        ) {
            Icon(Icons.Default.FileDownload, null, modifier = Modifier.size(20.dp))
            Spacer(Modifier.width(8.dp))
            Text("匯出 CSV ($recordCount 筆)", fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun SummaryCard(records: List<SensorRecord>) {
    val first = records.first()
    val last = records.last()
    val duration = last.elapsedSec - first.elapsedSec
    val steps = last.totalSteps - first.totalSteps

    // Heading change analysis
    val headingChanges = records.zipWithNext().sumOf { (a, b) ->
        var diff = (b.heading - a.heading).toDouble()
        if (diff > 180) diff -= 360
        if (diff < -180) diff += 360
        kotlin.math.abs(diff)
    }

    // Gyro magnitude average
    val avgGyro = records.map {
        kotlin.math.sqrt((it.gyroX * it.gyroX + it.gyroY * it.gyroY + it.gyroZ * it.gyroZ).toDouble())
    }.average()

    // Pressure change (floor detection)
    val pressureChange = if (first.pressure > 0 && last.pressure > 0) last.pressure - first.pressure else 0f

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(SurfaceBase)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(38.dp)
                    .background(Success.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Analytics, null, tint = Success, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Text("記錄摘要", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        Divider(color = Border)

        SummaryRow("記錄時長", "%.1f 秒".format(duration))
        SummaryRow("計步器步數", "$steps 步")
        SummaryRow("步伐偵測器", "${last.stepDetected} 次")
        SummaryRow("估計步頻", if (duration > 0) "%.1f 步/秒".format(steps / duration) else "—")
        SummaryRow("估計距離 (0.65m/步)", "%.1f m".format(steps * 0.65))
        SummaryRow("累計朝向變化", "%.0f°".format(headingChanges))
        SummaryRow("平均陀螺儀幅值", "%.4f rad/s".format(avgGyro))
        SummaryRow("起始朝向", "%.0f°".format(first.heading))
        SummaryRow("結束朝向", "%.0f°".format(last.heading))
        if (first.pressure > 0) {
            SummaryRow("氣壓變化", "%.2f hPa".format(pressureChange))
            SummaryRow("估計樓層變化", "%.1f 層".format(pressureChange / -0.4f))
        }
    }
}

@Composable
private fun HistoryCard(
    logs: List<LogFileInfo>,
    onShare: (LogFileInfo) -> Unit,
    onDelete: (LogFileInfo) -> Unit,
) {
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
                    .background(ChartBlue.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.History, null, tint = ChartBlue, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("歷史記錄", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
                Text("${logs.size} 筆記錄", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
            }
        }

        logs.forEach { info ->
            var showConfirmDelete by remember { mutableStateOf(false) }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(12.dp))
                    .background(Noir)
                    .padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            info.date,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary,
                        )
                        Text(
                            "${info.dataPoints} 筆 · ${info.sizeKb}",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextTertiary,
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        IconButton(
                            onClick = { onShare(info) },
                            modifier = Modifier.size(36.dp),
                        ) {
                            Icon(Icons.Default.Share, "分享", tint = Info, modifier = Modifier.size(18.dp))
                        }
                        IconButton(
                            onClick = { showConfirmDelete = true },
                            modifier = Modifier.size(36.dp),
                        ) {
                            Icon(Icons.Default.Delete, "刪除", tint = Danger, modifier = Modifier.size(18.dp))
                        }
                    }
                }

                if (info.groundTruthSteps.isNotBlank() || info.groundTruthDistance.isNotBlank()) {
                    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                        if (info.groundTruthSteps.isNotBlank()) {
                            Text(
                                "實際步數: ${info.groundTruthSteps}",
                                style = MaterialTheme.typography.labelSmall,
                                color = Gold,
                            )
                        }
                        if (info.groundTruthDistance.isNotBlank()) {
                            Text(
                                "實際距離: ${info.groundTruthDistance}m",
                                style = MaterialTheme.typography.labelSmall,
                                color = Gold,
                            )
                        }
                    }
                }

                if (showConfirmDelete) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End),
                    ) {
                        TextButton(onClick = { showConfirmDelete = false }) {
                            Text("取消", color = TextSecondary)
                        }
                        Button(
                            onClick = {
                                showConfirmDelete = false
                                onDelete(info)
                            },
                            colors = ButtonDefaults.buttonColors(containerColor = Danger),
                            shape = RoundedCornerShape(10.dp),
                        ) {
                            Text("確認刪除")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SummaryRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            fontFamily = FontFamily.Monospace,
            color = TextPrimary,
        )
    }
}
