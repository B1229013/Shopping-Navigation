package com.example.shopping.ui.screens

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorManager
import android.os.Build
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import android.graphics.Paint as NativePaint
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.NavController
import com.example.shopping.LocalVolumeKeyHandler
import com.example.shopping.sensor.PdrPoint
import com.example.shopping.sensor.PdrTracker
import com.example.shopping.ui.components.StaggeredItem
import com.example.shopping.ui.theme.*
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

// ── GPS convergence status ──

private enum class GpsConvergeStatus { WAITING, CONVERGING, READY }

// ── Posture enum ──

private enum class Posture(val label: String, val tag: String) {
    HAND_HELD("手持", "hand_held"),
    POCKET("口袋", "pocket"),
    SWINGING("擺動", "swinging"),
}

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
    val gameRotX: Float, val gameRotY: Float, val gameRotZ: Float, val gameRotW: Float,
    val gravX: Float, val gravY: Float, val gravZ: Float,
    val linAccX: Float, val linAccY: Float, val linAccZ: Float,
    val pressure: Float,
    val light: Float,
    val pdrX: Float = 0f,
    val pdrY: Float = 0f,
    val grvYawDeg: Float = 0f,
    val rvYawDeg: Float = 0f,
    val compYawDeg: Float = 0f,
    val gyroYawDeg: Float = 0f,
    val calYawDeg: Float = 0f,
    val gpsLat: Double? = null,
    val gpsLng: Double? = null,
    val gpsAccM: Float? = null,
    val waypoint: String = "",
)

private val PHOTO_DIR_TAGS = listOf("front", "right", "back", "left")
private val PHOTO_DIR_LABELS = listOf("前方", "右方", "後方", "左方")

private data class WaypointMarker(
    val x: Float, val y: Float,
    val name: String,
    val stepIndex: Int,
    val elapsedSec: Double,
    val photoFiles: List<String> = emptyList(),
)

// ── CSV export ──

private fun exportZip(
    context: Context,
    records: List<SensorRecord>,
    waypoints: List<WaypointMarker>,
    groundTruth: GroundTruth,
    posture: Posture,
): File? {
    if (records.isEmpty()) return null
    val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    val dir = File(context.getExternalFilesDir(null), "sensor_lab")
    dir.mkdirs()
    val zipFile = File(dir, "sensor_log_${posture.tag}_$ts.zip")

    val csvName = "sensor_log.csv"
    ZipOutputStream(BufferedOutputStream(FileOutputStream(zipFile))).use { zos ->
        // CSV entry — stream row-by-row to avoid large StringBuilder in memory
        zos.putNextEntry(ZipEntry(csvName))
        val writer = zos.bufferedWriter(Charsets.UTF_8)
        writer.appendLine("# SensorLab Export — $ts")
        writer.appendLine("# posture=${posture.tag}")
        writer.appendLine("# ground_truth_steps=${groundTruth.steps},ground_truth_distance_m=${groundTruth.distanceM},ground_truth_turns=${groundTruth.turns},weinberg_k=${groundTruth.weinbergK}")
        writer.appendLine("timestamp_ms,elapsed_sec,total_steps,step_detected,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,heading_deg,rot_x,rot_y,rot_z,rot_w,game_rot_x,game_rot_y,game_rot_z,game_rot_w,grav_x,grav_y,grav_z,lin_acc_x,lin_acc_y,lin_acc_z,pressure_hpa,light_lux,pdr_x,pdr_y,grv_yaw_deg,rv_yaw_deg,comp_yaw_deg,gyro_yaw_deg,cal_yaw_deg,gps_lat,gps_lng,gps_acc_m,waypoint")
        for (r in records) {
            val wpField = if (r.waypoint.isNotBlank()) "\"${r.waypoint}\"" else ""
            val gpsLat = r.gpsLat?.let { "%.6f".format(it) } ?: ""
            val gpsLng = r.gpsLng?.let { "%.6f".format(it) } ?: ""
            val gpsAcc = r.gpsAccM?.let { "%.1f".format(it) } ?: ""
            writer.appendLine(
                "${r.timestampMs},%.3f,%d,%d,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.1f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f,%.1f,%s,%s,%s,%s".format(
                    r.elapsedSec, r.totalSteps, r.stepDetected,
                    r.accelX, r.accelY, r.accelZ,
                    r.gyroX, r.gyroY, r.gyroZ,
                    r.magX, r.magY, r.magZ,
                    r.heading,
                    r.rotX, r.rotY, r.rotZ, r.rotW,
                    r.gameRotX, r.gameRotY, r.gameRotZ, r.gameRotW,
                    r.gravX, r.gravY, r.gravZ,
                    r.linAccX, r.linAccY, r.linAccZ,
                    r.pressure, r.light,
                    r.pdrX, r.pdrY,
                    r.grvYawDeg, r.rvYawDeg, r.compYawDeg, r.gyroYawDeg, r.calYawDeg,
                    gpsLat, gpsLng, gpsAcc, wpField,
                )
            )
        }
        writer.flush()
        zos.closeEntry()

        // Photo entries
        val photoDir = File(context.getExternalFilesDir(null), "sensor_lab/photos")
        for (wp in waypoints) {
            for (pf in wp.photoFiles) {
                val photo = File(photoDir, pf)
                if (!photo.exists()) continue
                zos.putNextEntry(ZipEntry("photos/$pf"))
                FileInputStream(photo).use { it.copyTo(zos) }
                zos.closeEntry()
            }
        }
    }
    return zipFile
}

private data class GroundTruth(
    val steps: String = "",
    val distanceM: String = "",
    val turns: String = "",
    val weinbergK: String = "0.4667",
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
    val posture: String,
    val photoCount: Int,
)

private fun listSavedLogs(context: Context): List<LogFileInfo> {
    val dir = File(context.getExternalFilesDir(null), "sensor_lab")
    if (!dir.exists()) return emptyList()

    val zipLogs = dir.listFiles { f -> f.extension == "zip" }
        ?.sortedByDescending { it.lastModified() }
        ?.mapNotNull { f ->
            try {
                ZipFile(f).use { zf ->
                    val csvEntry = zf.entries().asSequence().firstOrNull { it.name.endsWith(".csv") } ?: return@mapNotNull null
                    val lines = zf.getInputStream(csvEntry).bufferedReader().readLines()
                    val commentLines = lines.filter { it.startsWith("#") }
                    val posture = commentLines.firstNotNullOfOrNull { Regex("posture=([^,\\s]*)").find(it)?.groupValues?.get(1) } ?: ""
                    val gtSteps = commentLines.firstNotNullOfOrNull { Regex("ground_truth_steps=([^,]*)").find(it)?.groupValues?.get(1) } ?: ""
                    val gtDist = commentLines.firstNotNullOfOrNull { Regex("ground_truth_distance_m=([^,]*)").find(it)?.groupValues?.get(1) } ?: ""
                    val dataLines = lines.count { !it.startsWith("#") && it.contains(",") } - 1
                    val photos = zf.entries().asSequence().count { it.name.startsWith("photos/") && !it.isDirectory }
                    LogFileInfo(
                        file = f,
                        name = f.nameWithoutExtension.removePrefix("sensor_log_"),
                        date = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.US).format(Date(f.lastModified())),
                        sizeKb = "%.1f KB".format(f.length() / 1024.0),
                        dataPoints = dataLines.coerceAtLeast(0),
                        groundTruthSteps = gtSteps,
                        groundTruthDistance = gtDist,
                        posture = posture,
                        photoCount = photos,
                    )
                }
            } catch (_: Exception) { null }
        } ?: emptyList()

    // Also list legacy CSV files
    val csvLogs = dir.listFiles { f -> f.extension == "csv" }
        ?.sortedByDescending { it.lastModified() }
        ?.map { f ->
            val lines = f.readLines()
            val commentLines = lines.filter { it.startsWith("#") }
            val posture = commentLines.firstNotNullOfOrNull { Regex("posture=([^,\\s]*)").find(it)?.groupValues?.get(1) } ?: ""
            val gtSteps = commentLines.firstNotNullOfOrNull { Regex("ground_truth_steps=([^,]*)").find(it)?.groupValues?.get(1) } ?: ""
            val gtDist = commentLines.firstNotNullOfOrNull { Regex("ground_truth_distance_m=([^,]*)").find(it)?.groupValues?.get(1) } ?: ""
            val dataLines = lines.count { !it.startsWith("#") && it.contains(",") } - 1
            LogFileInfo(
                file = f,
                name = f.nameWithoutExtension.removePrefix("sensor_log_"),
                date = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.US).format(Date(f.lastModified())),
                sizeKb = "%.1f KB".format(f.length() / 1024.0),
                dataPoints = dataLines.coerceAtLeast(0),
                groundTruthSteps = gtSteps,
                groundTruthDistance = gtDist,
                posture = posture,
                photoCount = 0,
            )
        } ?: emptyList()

    return (zipLogs + csvLogs).sortedByDescending { it.file.lastModified() }
}

private fun shareFile(context: Context, file: File) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    val mimeType = if (file.extension == "zip") "application/zip" else "text/csv"
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = mimeType
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
    val lifecycleOwner = LocalLifecycleOwner.current
    val sensorManager = remember { context.getSystemService(Context.SENSOR_SERVICE) as SensorManager }

    // ── Permissions ──
    var activityPermissionGranted by remember {
        mutableStateOf(
            Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            context.checkSelfPermission(Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
        )
    }
    val activityPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { activityPermissionGranted = it }

    var cameraPermissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val cameraPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { cameraPermissionGranted = it }

    var locationPermissionGranted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }
    val locationPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { locationPermissionGranted = it }

    LaunchedEffect(Unit) {
        if (!activityPermissionGranted) activityPermLauncher.launch(Manifest.permission.ACTIVITY_RECOGNITION)
        if (!cameraPermissionGranted) cameraPermLauncher.launch(Manifest.permission.CAMERA)
        if (!locationPermissionGranted) locationPermLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
    }


    // ── PdrTracker ──
    val pdrTracker = remember { PdrTracker(sensorManager) }

    // ── Camera ──
    val imageCapture = remember { ImageCapture.Builder().build() }
    val cameraPreview = remember { Preview.Builder().build() }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    var showCameraPreview by remember { mutableStateOf(false) }

    LaunchedEffect(cameraPermissionGranted) {
        if (!cameraPermissionGranted) return@LaunchedEffect
        val cameraProvider = withContext(Dispatchers.IO) {
            ProcessCameraProvider.getInstance(context).get()
        }
        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, cameraPreview, imageCapture)
        } catch (e: Exception) {
            Log.e("SensorLab", "Camera bind failed", e)
        }
    }

    DisposableEffect(Unit) {
        onDispose { captureExecutor.shutdown() }
    }

    // ── Recording state ──
    var isRecording by remember { mutableStateOf(false) }
    var isPaused by remember { mutableStateOf(false) }
    var pausedElapsedMs by remember { mutableLongStateOf(0L) }
    var recordStartTime by remember { mutableLongStateOf(0L) }
    var elapsedSeconds by remember { mutableDoubleStateOf(0.0) }
    val records = remember { mutableStateListOf<SensorRecord>() }

    // ── Posture ──
    var posture by remember { mutableStateOf(Posture.HAND_HELD) }

    // ── Display values (updated via polling) ──
    var displayStepCount by remember { mutableIntStateOf(0) }
    var displayStepDetected by remember { mutableIntStateOf(0) }
    var displayHeading by remember { mutableFloatStateOf(0f) }
    var displayAccel by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var displayGyro by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var displayMag by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var displayRotVec by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f, 1f)) }
    var displayPressure by remember { mutableFloatStateOf(0f) }
    var displayLight by remember { mutableFloatStateOf(0f) }
    var displayPdrX by remember { mutableFloatStateOf(0f) }
    var displayPdrY by remember { mutableFloatStateOf(0f) }
    var displayGrvYaw by remember { mutableFloatStateOf(0f) }
    var displayRvYaw by remember { mutableFloatStateOf(0f) }
    var displayCompYaw by remember { mutableFloatStateOf(0f) }
    var displayGyroYaw by remember { mutableFloatStateOf(0f) }
    var displayCalYaw by remember { mutableFloatStateOf(0f) }
    var displayGpsLat by remember { mutableStateOf<Double?>(null) }
    var displayGpsLng by remember { mutableStateOf<Double?>(null) }
    var displayGpsAcc by remember { mutableStateOf<Float?>(null) }

    // ── GPS convergence ──
    var gpsStatus by remember { mutableStateOf(GpsConvergeStatus.WAITING) }
    val gpsConvergenceWindow = remember { mutableStateListOf<Triple<Double, Double, Float>>() }
    val GPS_ACCURACY_THRESHOLD = 25f
    val GPS_CONVERGE_COUNT = 3
    val GPS_CONVERGE_RADIUS = 10.0

    var magCalStatus by remember { mutableIntStateOf(SensorManager.SENSOR_STATUS_UNRELIABLE) }
    var gyroCalState by remember { mutableStateOf(PdrTracker.GyroCalState.UNCALIBRATED) }
    var gyroCalProgress by remember { mutableFloatStateOf(0f) }
    var gyroBias by remember { mutableStateOf(floatArrayOf(0f, 0f, 0f)) }
    var gyroCalNoise by remember { mutableFloatStateOf(0f) }
    var totalPdrDistance by remember { mutableFloatStateOf(0f) }
    var lastStride by remember { mutableFloatStateOf(0f) }
    var northCalibrated by remember { mutableStateOf(false) }
    var magQuality by remember { mutableStateOf(PdrTracker.MagQuality.UNKNOWN) }
    var magFieldStrength by remember { mutableFloatStateOf(0f) }

    // ── PDR waypoints ──
    val waypoints = remember { mutableStateListOf<WaypointMarker>() }
    var pendingWaypoint by remember { mutableStateOf("") }
    var showWaypointDialog by remember { mutableStateOf(false) }
    var waypointName by remember { mutableStateOf("") }
    var isTakingPhoto by remember { mutableStateOf(false) }
    var isMultiCapturing by remember { mutableStateOf(false) }
    var multiCaptureDir by remember { mutableIntStateOf(0) }
    val multiCapturePhotos = remember { mutableStateListOf<String>() }
    var multiCaptureX by remember { mutableFloatStateOf(0f) }
    var multiCaptureY by remember { mutableFloatStateOf(0f) }
    var multiCaptureStepIndex by remember { mutableIntStateOf(0) }
    var multiCaptureElapsed by remember { mutableStateOf(0.0) }
    var isExporting by remember { mutableStateOf(false) }
    val exportScope = rememberCoroutineScope()

    // ── Ground truth + history ──
    var groundTruth by remember { mutableStateOf(GroundTruth()) }
    var savedLogs by remember { mutableStateOf(listSavedLogs(context)) }

    // ── Start/stop PdrTracker ──
    DisposableEffect(activityPermissionGranted) {
        if (activityPermissionGranted) {
            val k = groundTruth.weinbergK.toFloatOrNull() ?: 0.4667f
            pdrTracker.onMagAccuracyChanged = { magCalStatus = it }
            pdrTracker.onNorthCalibrated = { northCalibrated = true }
            pdrTracker.onMagQualityChanged = { quality, strength ->
                magQuality = quality
                magFieldStrength = strength
            }
            pdrTracker.onGyroCalStateChanged = { state ->
                gyroCalState = state
                gyroCalProgress = pdrTracker.gyroCalProgress
                gyroBias = pdrTracker.gyroBias.copyOf()
                gyroCalNoise = pdrTracker.gyroCalNoise
            }
            pdrTracker.onSampleReady = {
                if (isRecording) {
                    val now = SystemClock.elapsedRealtime()
                    val elapsed = (now - recordStartTime) / 1000.0
                    if (records.isEmpty() || elapsed - records.last().elapsedSec >= 0.02) {
                        val rv = pdrTracker.currentRotVec
                        val grv = pdrTracker.currentGameRotVec
                        records.add(SensorRecord(
                            timestampMs = System.currentTimeMillis(),
                            elapsedSec = elapsed,
                            totalSteps = pdrTracker.totalSteps,
                            stepDetected = pdrTracker.stepDetectorCount,
                            accelX = pdrTracker.accelValues[0], accelY = pdrTracker.accelValues[1], accelZ = pdrTracker.accelValues[2],
                            gyroX = pdrTracker.gyroValues[0], gyroY = pdrTracker.gyroValues[1], gyroZ = pdrTracker.gyroValues[2],
                            magX = pdrTracker.magValues[0], magY = pdrTracker.magValues[1], magZ = pdrTracker.magValues[2],
                            heading = pdrTracker.headingDeg,
                            rotX = rv[0], rotY = rv[1], rotZ = rv[2], rotW = rv[3],
                            gameRotX = grv[0], gameRotY = grv[1], gameRotZ = grv[2], gameRotW = grv[3],
                            gravX = pdrTracker.gravityValues[0], gravY = pdrTracker.gravityValues[1], gravZ = pdrTracker.gravityValues[2],
                            linAccX = pdrTracker.linAccValues[0], linAccY = pdrTracker.linAccValues[1], linAccZ = pdrTracker.linAccValues[2],
                            pressure = pdrTracker.pressureValue,
                            light = pdrTracker.lightValue,
                            pdrX = pdrTracker.x, pdrY = pdrTracker.y,
                            grvYawDeg = pdrTracker.getGameRotVecYawDeg(),
                            rvYawDeg = pdrTracker.getRotVecYawDeg(),
                            compYawDeg = pdrTracker.getComplementaryYawDeg(),
                            gyroYawDeg = pdrTracker.getGyroOnlyYawDeg(),
                            calYawDeg = pdrTracker.getCalibratedYawDeg(),
                            gpsLat = pdrTracker.gpsLat, gpsLng = pdrTracker.gpsLng, gpsAccM = pdrTracker.gpsAccuracy,
                            waypoint = pendingWaypoint,
                        ))
                        if (pendingWaypoint.isNotBlank()) pendingWaypoint = ""
                    }
                }
            }
            pdrTracker.start(k)
        }
        onDispose {
            pdrTracker.onSampleReady = null
            pdrTracker.onNorthCalibrated = null
            pdrTracker.onMagQualityChanged = null
            pdrTracker.stop()
        }
    }

    // ── GPS with convergence detection ──
    fun haversineMetres(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val r = 6_371_000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2)
        return r * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
    }

    DisposableEffect(locationPermissionGranted) {
        if (!locationPermissionGranted) return@DisposableEffect onDispose {}
        val client = LocationServices.getFusedLocationProviderClient(context)
        val callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return

                if (loc.accuracy > GPS_ACCURACY_THRESHOLD) {
                    Log.d("SensorLab", "GPS fix dropped: accuracy=${loc.accuracy}m")
                    return
                }

                if (gpsStatus != GpsConvergeStatus.READY) {
                    gpsConvergenceWindow.add(Triple(loc.latitude, loc.longitude, loc.accuracy))
                    if (gpsConvergenceWindow.size > GPS_CONVERGE_COUNT)
                        gpsConvergenceWindow.removeAt(0)

                    if (gpsConvergenceWindow.size >= GPS_CONVERGE_COUNT) {
                        val anchor = gpsConvergenceWindow.last()
                        val converged = gpsConvergenceWindow.all { fix ->
                            haversineMetres(anchor.first, anchor.second, fix.first, fix.second) < GPS_CONVERGE_RADIUS
                        }
                        if (converged) {
                            gpsStatus = GpsConvergeStatus.READY
                            Log.i("SensorLab", "GPS converged, accuracy=${loc.accuracy}m")
                        } else {
                            gpsStatus = GpsConvergeStatus.CONVERGING
                            return
                        }
                    } else {
                        gpsStatus = GpsConvergeStatus.CONVERGING
                        return
                    }
                }

                pdrTracker.updateLocation(loc.latitude, loc.longitude, loc.accuracy, loc.altitude)
            }
        }
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(500L)
            .build()
        try {
            client.requestLocationUpdates(request, callback, Looper.getMainLooper())
        } catch (_: SecurityException) {}
        onDispose {
            client.removeLocationUpdates(callback)
            gpsConvergenceWindow.clear()
            gpsStatus = GpsConvergeStatus.WAITING
        }
    }

    // ── Display update polling (10Hz) ──
    LaunchedEffect(Unit) {
        while (true) {
            displayStepCount = pdrTracker.totalSteps
            displayStepDetected = pdrTracker.stepDetectorCount
            displayHeading = pdrTracker.headingDeg
            displayAccel = pdrTracker.accelValues
            displayGyro = pdrTracker.gyroValues
            displayMag = pdrTracker.magValues
            displayRotVec = pdrTracker.currentRotVec
            displayPressure = pdrTracker.pressureValue
            displayLight = pdrTracker.lightValue
            displayPdrX = pdrTracker.x
            displayPdrY = pdrTracker.y
            totalPdrDistance = pdrTracker.totalDistance
            lastStride = pdrTracker.lastStride
            displayGrvYaw = pdrTracker.getGameRotVecYawDeg()
            displayRvYaw = pdrTracker.getRotVecYawDeg()
            displayCompYaw = pdrTracker.getComplementaryYawDeg()
            displayGyroYaw = pdrTracker.getGyroOnlyYawDeg()
            displayCalYaw = pdrTracker.getCalibratedYawDeg()
            displayGpsLat = pdrTracker.gpsLat
            displayGpsLng = pdrTracker.gpsLng
            displayGpsAcc = pdrTracker.gpsAccuracy
            magQuality = pdrTracker.magQuality
            magFieldStrength = pdrTracker.magFieldStrength
            if (pdrTracker.gyroCalState == PdrTracker.GyroCalState.CALIBRATING) {
                gyroCalProgress = pdrTracker.gyroCalProgress
            }
            delay(100)
        }
    }

    // ── Elapsed timer ──
    LaunchedEffect(isRecording) {
        while (isRecording) {
            elapsedSeconds = (SystemClock.elapsedRealtime() - recordStartTime) / 1000.0
            delay(100)
        }
    }

    // ── Weinberg K sync ──
    LaunchedEffect(groundTruth.weinbergK) {
        groundTruth.weinbergK.toFloatOrNull()?.let { if (it > 0) pdrTracker.setWeinbergK(it) }
    }

    // ── Multi-directional photo waypoint capture ──
    fun startMultiCapture() {
        if (isMultiCapturing || !cameraPermissionGranted) return
        isMultiCapturing = true
        multiCaptureDir = 0
        multiCapturePhotos.clear()
        multiCaptureX = displayPdrX
        multiCaptureY = displayPdrY
        multiCaptureStepIndex = displayStepCount
        multiCaptureElapsed = (SystemClock.elapsedRealtime() - recordStartTime) / 1000.0
        if (!showCameraPreview) showCameraPreview = true
    }

    fun finishMultiCapture() {
        if (!isMultiCapturing) return
        if (multiCapturePhotos.isNotEmpty()) {
            val name = "照片${waypoints.size + 1}"
            waypoints.add(WaypointMarker(multiCaptureX, multiCaptureY, name, multiCaptureStepIndex, multiCaptureElapsed, multiCapturePhotos.toList()))
            pendingWaypoint = name
        }
        isMultiCapturing = false
        multiCaptureDir = 0
        multiCapturePhotos.clear()
    }

    fun captureCurrentDirection() {
        if (isTakingPhoto || !isMultiCapturing || !cameraPermissionGranted) return
        isTakingPhoto = true
        val photoDir = File(context.getExternalFilesDir(null), "sensor_lab/photos")
        photoDir.mkdirs()
        val wpNum = waypoints.size + 1
        val dirTag = PHOTO_DIR_TAGS[multiCaptureDir]
        val photoName = "wp_${wpNum}_${dirTag}_${System.currentTimeMillis()}.jpg"
        val photoFile = File(photoDir, photoName)
        val options = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        imageCapture.takePicture(options, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                multiCapturePhotos.add(photoName)
                multiCaptureDir++
                isTakingPhoto = false
                if (multiCaptureDir >= 4) finishMultiCapture()
            }
            override fun onError(e: ImageCaptureException) {
                Log.e("SensorLab", "Photo capture failed", e)
                isTakingPhoto = false
            }
        })
    }

    // ── Volume key waypoint recording ──
    val volumeKeyHandler = LocalVolumeKeyHandler.current
    var volumeWaypointCount by remember { mutableIntStateOf(0) }

    DisposableEffect(isRecording) {
        if (isRecording) {
            volumeKeyHandler.onVolumeDown = {
                volumeWaypointCount++
                val name = "按鍵$volumeWaypointCount"
                val elapsed = (SystemClock.elapsedRealtime() - recordStartTime) / 1000.0
                waypoints.add(WaypointMarker(displayPdrX, displayPdrY, name, displayStepCount, elapsed))
                pendingWaypoint = name
            }
            volumeKeyHandler.onVolumeUp = {
                if (!isMultiCapturing) {
                    startMultiCapture()
                } else {
                    captureCurrentDirection()
                }
            }
        } else {
            volumeKeyHandler.onVolumeDown = null
            volumeKeyHandler.onVolumeUp = null
        }
        onDispose {
            volumeKeyHandler.onVolumeDown = null
            volumeKeyHandler.onVolumeUp = null
        }
    }

    Scaffold(
        containerColor = Noir,
        topBar = {
            TopAppBar(
                title = { Text("SensorLab", style = MaterialTheme.typography.titleLarge, color = TextPrimary) },
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

            // ── Posture selection ──
            StaggeredItem(index = 1) { PostureSelector(posture) { posture = it } }

            // ── Mag calibration status ──
            StaggeredItem(index = 2) { MagCalibrationCard(magCalStatus) }

            // ── Mag field quality ──
            StaggeredItem(index = 2) { MagFieldQualityCard(magQuality, magFieldStrength) }

            // ── Gyro calibration ──
            StaggeredItem(index = 2) {
                GyroCalibrationCard(
                    state = gyroCalState,
                    progress = gyroCalProgress,
                    bias = gyroBias,
                    noise = gyroCalNoise,
                    onCalibrate = { pdrTracker.startGyroCalibration() },
                )
            }

            // ── GPS status ──
            StaggeredItem(index = 3) {
                GpsStatusCard(displayGpsLat, displayGpsLng, displayGpsAcc, locationPermissionGranted, gpsStatus)
            }

            // ── Live readings ──
            StaggeredItem(index = 4) {
                LiveReadingsCard(
                    stepCount = displayStepCount,
                    stepDetected = displayStepDetected,
                    heading = displayHeading,
                    accel = displayAccel, gyro = displayGyro, mag = displayMag,
                    rotationVec = displayRotVec,
                    pressure = displayPressure, light = displayLight,
                    grvYaw = displayGrvYaw, rvYaw = displayRvYaw,
                    compYaw = displayCompYaw, gyroYaw = displayGyroYaw, calYaw = displayCalYaw,
                )
            }

            // ── Recording control ──
            StaggeredItem(index = 5) {
                RecordingCard(
                    isRecording = isRecording,
                    isPaused = isPaused,
                    elapsedSeconds = elapsedSeconds,
                    recordCount = records.size,
                    stepCount = displayStepCount,
                    northCalibrated = northCalibrated,
                    magAccuracy = magCalStatus,
                    onStart = {
                        records.clear()
                        pdrTracker.reset()
                        northCalibrated = pdrTracker.isNorthCalibrated
                        waypoints.clear()
                        pendingWaypoint = ""
                        pausedElapsedMs = 0L
                        recordStartTime = SystemClock.elapsedRealtime()
                        isPaused = false
                        isRecording = true
                    },
                    onPause = {
                        pausedElapsedMs = SystemClock.elapsedRealtime() - recordStartTime
                        isRecording = false
                        isPaused = true
                    },
                    onResume = {
                        recordStartTime = SystemClock.elapsedRealtime() - pausedElapsedMs
                        isPaused = false
                        isRecording = true
                    },
                    onStop = {
                        isRecording = false
                        isPaused = false
                    },
                )
            }

            // ── Camera preview (during recording) ──
            if (isRecording && showCameraPreview && cameraPermissionGranted) {
                StaggeredItem(index = 6) {
                    Column(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
                            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase),
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            if (isMultiCapturing) {
                                Text(
                                    "拍攝 ${PHOTO_DIR_LABELS[multiCaptureDir]} (${multiCaptureDir + 1}/4)",
                                    style = MaterialTheme.typography.titleSmall, color = ChartBlue,
                                )
                            } else {
                                Text("相機預覽", style = MaterialTheme.typography.titleSmall, color = TextPrimary)
                            }
                            if (!isMultiCapturing) {
                                IconButton(onClick = { showCameraPreview = false }, modifier = Modifier.size(32.dp)) {
                                    Icon(Icons.Default.Close, contentDescription = "關閉預覽", tint = TextSecondary, modifier = Modifier.size(18.dp))
                                }
                            }
                        }
                        if (isMultiCapturing) {
                            Row(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                                horizontalArrangement = Arrangement.spacedBy(6.dp),
                            ) {
                                PHOTO_DIR_LABELS.forEachIndexed { i, label ->
                                    val done = i < multiCaptureDir
                                    val current = i == multiCaptureDir
                                    Box(
                                        modifier = Modifier.weight(1f).height(32.dp)
                                            .background(
                                                when {
                                                    done -> Success.copy(alpha = 0.15f)
                                                    current -> ChartBlue.copy(alpha = 0.15f)
                                                    else -> Border.copy(alpha = 0.3f)
                                                },
                                                RoundedCornerShape(8.dp)
                                            ),
                                        contentAlignment = Alignment.Center,
                                    ) {
                                        Text(
                                            if (done) "$label ✓" else label,
                                            fontSize = 13.sp,
                                            fontWeight = if (current) FontWeight.Bold else FontWeight.Normal,
                                            color = when {
                                                done -> Success
                                                current -> ChartBlue
                                                else -> TextTertiary
                                            },
                                        )
                                    }
                                }
                            }
                        }
                        Box {
                            AndroidView(
                                factory = { ctx ->
                                    PreviewView(ctx).also {
                                        it.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                                        cameraPreview.setSurfaceProvider(it.surfaceProvider)
                                    }
                                },
                                modifier = Modifier.fillMaxWidth().aspectRatio(3f / 4f)
                                    .clip(RoundedCornerShape(bottomStart = 20.dp, bottomEnd = 20.dp)),
                            )
                            if (isMultiCapturing) {
                                Box(
                                    modifier = Modifier.align(Alignment.TopCenter).padding(top = 12.dp)
                                        .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(20.dp))
                                        .padding(horizontal = 16.dp, vertical = 6.dp),
                                ) {
                                    Text(
                                        "請拍攝${PHOTO_DIR_LABELS[multiCaptureDir]}",
                                        color = Color.White, fontWeight = FontWeight.Bold, fontSize = 16.sp,
                                    )
                                }
                            }
                        }
                    }
                }
            }

            // ── Waypoint + photo buttons (during recording) ──
            if (isRecording) {
                StaggeredItem(index = 6) {
                    if (isMultiCapturing) {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Button(
                                onClick = { captureCurrentDirection() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                enabled = !isTakingPhoto,
                                colors = ButtonDefaults.buttonColors(containerColor = ChartBlue),
                                shape = RoundedCornerShape(14.dp),
                            ) {
                                if (isTakingPhoto) {
                                    CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Color.White, strokeWidth = 2.dp)
                                } else {
                                    Icon(Icons.Default.CameraAlt, null, modifier = Modifier.size(20.dp))
                                }
                                Spacer(Modifier.width(6.dp))
                                Text("拍攝${PHOTO_DIR_LABELS[multiCaptureDir]}", fontWeight = FontWeight.SemiBold)
                            }
                            Button(
                                onClick = { finishMultiCapture() },
                                modifier = Modifier.height(48.dp),
                                enabled = multiCapturePhotos.isNotEmpty() && !isTakingPhoto,
                                colors = ButtonDefaults.buttonColors(containerColor = Success),
                                shape = RoundedCornerShape(14.dp),
                            ) {
                                Icon(Icons.Default.Check, null, modifier = Modifier.size(20.dp))
                                Spacer(Modifier.width(4.dp))
                                Text("完成", fontWeight = FontWeight.SemiBold)
                            }
                        }
                    } else {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                        ) {
                            Button(
                                onClick = { waypointName = ""; showWaypointDialog = true },
                                modifier = Modifier.weight(1f).height(48.dp),
                                colors = ButtonDefaults.buttonColors(containerColor = ChartAmber),
                                shape = RoundedCornerShape(14.dp),
                            ) {
                                Icon(Icons.Default.AddLocation, null, modifier = Modifier.size(20.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("標記點位", fontWeight = FontWeight.SemiBold)
                            }
                            Button(
                                onClick = { startMultiCapture() },
                                modifier = Modifier.weight(1f).height(48.dp),
                                enabled = cameraPermissionGranted && !isTakingPhoto,
                                colors = ButtonDefaults.buttonColors(containerColor = ChartBlue),
                                shape = RoundedCornerShape(14.dp),
                            ) {
                                Icon(Icons.Default.CameraAlt, null, modifier = Modifier.size(20.dp))
                                Spacer(Modifier.width(6.dp))
                                Text("拍照標記", fontWeight = FontWeight.SemiBold)
                            }
                            IconButton(
                                onClick = { showCameraPreview = !showCameraPreview },
                                modifier = Modifier.size(48.dp)
                                    .background(
                                        if (showCameraPreview) ChartBlue.copy(alpha = 0.15f) else ChartBlue.copy(alpha = 0.10f),
                                        RoundedCornerShape(14.dp)
                                    ),
                            ) {
                                Icon(
                                    if (showCameraPreview) Icons.Default.VideocamOff else Icons.Default.Videocam,
                                    contentDescription = "切換相機預覽",
                                    tint = ChartBlue,
                                    modifier = Modifier.size(22.dp),
                                )
                            }
                            if (waypoints.isNotEmpty()) {
                                Box(
                                    modifier = Modifier.size(48.dp).background(ChartAmber.copy(alpha = 0.10f), RoundedCornerShape(14.dp)),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Text("${waypoints.size}", fontWeight = FontWeight.Bold, color = ChartAmber, fontSize = 18.sp)
                                }
                            }
                        }
                    }
                }
            }

            // ── Path trail ──
            if (pdrTracker.pathPoints.size > 1 || waypoints.isNotEmpty()) {
                StaggeredItem(index = 7) {
                    PathTrailCard(
                        pathPoints = pdrTracker.pathPoints,
                        waypoints = waypoints,
                        currentX = displayPdrX, currentY = displayPdrY,
                        currentHeading = displayHeading,
                        isRecording = isRecording,
                        stepCount = displayStepCount,
                        totalDistance = totalPdrDistance, lastStride = lastStride,
                    )
                }
            }

            // ── Ground truth + export ──
            StaggeredItem(index = 8) {
                GroundTruthCard(
                    groundTruth = groundTruth,
                    onUpdate = { groundTruth = it },
                    recordCount = records.size,
                    posture = posture,
                    onExport = {
                        if (!isExporting) {
                            isExporting = true
                            val snapshot = records.toList()
                            val wpSnapshot = waypoints.toList()
                            val gt = groundTruth
                            val pos = posture
                            exportScope.launch {
                                val file = withContext(Dispatchers.IO) {
                                    exportZip(context, snapshot, wpSnapshot, gt, pos)
                                }
                                if (file != null) {
                                    savedLogs = listSavedLogs(context)
                                    val photoCount = wpSnapshot.sumOf { it.photoFiles.size }
                                    val msg = if (photoCount > 0) "已匯出: ${file.name} ($photoCount 張照片)" else "已匯出: ${file.name}"
                                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(context, "無記錄可匯出", Toast.LENGTH_SHORT).show()
                                }
                                isExporting = false
                            }
                        }
                    }
                )
            }

            // ── Summary stats (after recording) ──
            if (!isRecording && records.isNotEmpty()) {
                StaggeredItem(index = 9) { SummaryCard(records) }
            }

            // ── History ──
            if (savedLogs.isNotEmpty()) {
                StaggeredItem(index = 10) {
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

    // Waypoint dialog
    if (showWaypointDialog) {
        AlertDialog(
            onDismissRequest = { showWaypointDialog = false },
            title = { Text("標記點位") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("為目前位置命名", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                    OutlinedTextField(
                        value = waypointName,
                        onValueChange = { waypointName = it },
                        label = { Text("點位名稱") },
                        placeholder = { Text("例如：轉角、入口、電梯") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = ChartAmber,
                            cursorColor = ChartAmber,
                            focusedLabelColor = ChartAmber,
                        ),
                    )
                    Text(
                        "步數: $displayStepCount · 距離: %.1fm · 步幅: %.2fm · 朝向: %.0f°".format(totalPdrDistance, lastStride, displayHeading),
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        color = TextTertiary,
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        val name = waypointName.ifBlank { "點位${waypoints.size + 1}" }
                        val elapsed = (SystemClock.elapsedRealtime() - recordStartTime) / 1000.0
                        waypoints.add(WaypointMarker(displayPdrX, displayPdrY, name, displayStepCount, elapsed))
                        pendingWaypoint = name
                        showWaypointDialog = false
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = ChartAmber),
                ) { Text("標記") }
            },
            dismissButton = {
                TextButton(onClick = { showWaypointDialog = false }) {
                    Text("取消", color = TextSecondary)
                }
            },
        )
    }
}


// ── UI Cards ──

@Composable
private fun PostureSelector(selected: Posture, onSelect: (Posture) -> Unit) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(ChartViolet.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.Accessibility, null, tint = ChartViolet, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Text("測試姿勢", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Posture.entries.forEach { p ->
                FilterChip(
                    selected = selected == p,
                    onClick = { onSelect(p) },
                    label = { Text(p.label, fontWeight = if (selected == p) FontWeight.Bold else FontWeight.Normal) },
                    modifier = Modifier.weight(1f),
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = ChartViolet.copy(alpha = 0.15f),
                        selectedLabelColor = ChartViolet,
                    ),
                )
            }
        }
    }
}

@Composable
private fun MagCalibrationCard(accuracy: Int) {
    val statusText: String
    val statusColor: Color
    val barWidth: Float
    val bgColor: Color
    when (accuracy) {
        SensorManager.SENSOR_STATUS_UNRELIABLE -> {
            statusText = "未校正"; statusColor = Color(0xFFDC2626); barWidth = 0.1f; bgColor = Color(0xFFFFF7ED)
        }
        SensorManager.SENSOR_STATUS_ACCURACY_LOW -> {
            statusText = "精度低"; statusColor = Color(0xFFD97706); barWidth = 0.35f; bgColor = Color(0xFFFFF7ED)
        }
        SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> {
            statusText = "中等精度"; statusColor = Color(0xFF16A34A); barWidth = 0.7f; bgColor = Color(0xFFF0FDF4)
        }
        else -> {
            statusText = "高精度"; statusColor = Color(0xFF059669); barWidth = 1f; bgColor = Color(0xFFF0FDF4)
        }
    }
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(bgColor)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                if (accuracy >= SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM) Icons.Default.CheckCircle else Icons.Default.Warning,
                null, tint = statusColor, modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(8.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text("磁力計校正：$statusText", fontWeight = FontWeight.Medium, fontSize = 13.sp, color = Color(0xFF92400E))
                if (accuracy < SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM) {
                    Text("請將手機在空中畫 8 字形", fontSize = 12.sp, color = Color(0xFFB45309))
                }
            }
        }
        Box(modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)).background(Color(0xFFE5E7EB))) {
            Box(modifier = Modifier.fillMaxHeight().fillMaxWidth(barWidth).clip(RoundedCornerShape(2.dp)).background(statusColor))
        }
    }
}

@Composable
private fun MagFieldQualityCard(quality: PdrTracker.MagQuality, strengthUt: Float) {
    val (statusText, statusColor, bgColor) = when (quality) {
        PdrTracker.MagQuality.UNKNOWN -> Triple("等待數據…", Color(0xFF6B7280), SurfaceBase)
        PdrTracker.MagQuality.GOOD -> Triple("正常", Color(0xFF16A34A), Color(0xFFF0FDF4))
        PdrTracker.MagQuality.SUSPECT -> Triple("可疑", Color(0xFFD97706), Color(0xFFFFF7ED))
        PdrTracker.MagQuality.BAD -> Triple("嚴重干擾", Color(0xFFDC2626), Color(0xFFFEF2F2))
    }
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(bgColor).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                when (quality) {
                    PdrTracker.MagQuality.GOOD -> Icons.Default.CheckCircle
                    PdrTracker.MagQuality.BAD -> Icons.Default.ErrorOutline
                    PdrTracker.MagQuality.SUSPECT -> Icons.Default.Warning
                    PdrTracker.MagQuality.UNKNOWN -> Icons.Default.Sensors
                },
                null, tint = statusColor, modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(8.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text("磁場品質：$statusText", fontWeight = FontWeight.Medium, fontSize = 13.sp, color = TextPrimary)
                if (quality != PdrTracker.MagQuality.UNKNOWN) {
                    Text(
                        "場強 %.0f μT（正常範圍 25-65 μT）".format(strengthUt),
                        fontSize = 12.sp, fontFamily = FontFamily.Monospace, color = TextSecondary,
                    )
                }
                if (quality == PdrTracker.MagQuality.BAD) {
                    Text("磁場干擾嚴重，磁力計方向不可信。建議遠離金屬物體。", fontSize = 12.sp, color = Color(0xFFDC2626))
                } else if (quality == PdrTracker.MagQuality.SUSPECT) {
                    Text("磁場偏離正常範圍，方向可能有誤差。", fontSize = 12.sp, color = Color(0xFFB45309))
                }
            }
        }

        if (quality != PdrTracker.MagQuality.UNKNOWN) {
            val barFraction = (strengthUt / 130f).coerceIn(0f, 1f)
            Box(modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)).background(Color(0xFFE5E7EB))) {
                Box(modifier = Modifier.fillMaxHeight().fillMaxWidth(barFraction).clip(RoundedCornerShape(3.dp)).background(statusColor))
            }
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("0", fontSize = 10.sp, color = TextTertiary)
                Text("|25", fontSize = 10.sp, color = Color(0xFF16A34A))
                Text("65|", fontSize = 10.sp, color = Color(0xFF16A34A))
                Text("130+ μT", fontSize = 10.sp, color = TextTertiary)
            }
        }
    }
}

@Composable
private fun GyroCalibrationCard(
    state: PdrTracker.GyroCalState,
    progress: Float,
    bias: FloatArray,
    noise: Float,
    onCalibrate: () -> Unit,
) {
    val (statusText, statusColor, bgColor) = when (state) {
        PdrTracker.GyroCalState.UNCALIBRATED -> Triple("未校正", Color(0xFFD97706), Color(0xFFFFF7ED))
        PdrTracker.GyroCalState.CALIBRATING -> Triple("校正中…", Color(0xFF2563EB), Color(0xFFEFF6FF))
        PdrTracker.GyroCalState.CALIBRATED -> Triple("已校正", Color(0xFF16A34A), Color(0xFFF0FDF4))
        PdrTracker.GyroCalState.FAILED -> Triple("校正失敗", Color(0xFFDC2626), Color(0xFFFEF2F2))
    }
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(bgColor).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                when (state) {
                    PdrTracker.GyroCalState.CALIBRATED -> Icons.Default.CheckCircle
                    PdrTracker.GyroCalState.CALIBRATING -> Icons.Default.Sync
                    PdrTracker.GyroCalState.FAILED -> Icons.Default.ErrorOutline
                    else -> Icons.Default.Warning
                },
                null, tint = statusColor, modifier = Modifier.size(20.dp),
            )
            Spacer(Modifier.width(8.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text("陀螺儀校正：$statusText", fontWeight = FontWeight.Medium, fontSize = 13.sp, color = TextPrimary)
                when (state) {
                    PdrTracker.GyroCalState.UNCALIBRATED -> Text("保持手機靜止 3 秒（任何姿勢皆可）", fontSize = 12.sp, color = TextTertiary)
                    PdrTracker.GyroCalState.CALIBRATING -> Text("請保持手機靜止不動…", fontSize = 12.sp, color = Color(0xFF2563EB))
                    PdrTracker.GyroCalState.FAILED -> {
                        Text("手機晃動過大，請握穩後重試", fontSize = 12.sp, color = Color(0xFFDC2626))
                        Text("Noise: %.5f rad/s（閾值: 0.05）".format(noise), fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = TextTertiary)
                    }
                    PdrTracker.GyroCalState.CALIBRATED -> {
                        Text(
                            "Bias: (%.5f, %.5f, %.5f) rad/s".format(bias[0], bias[1], bias[2]),
                            fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = TextSecondary,
                        )
                        Text("Noise: %.5f rad/s".format(noise), fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = TextSecondary)
                    }
                }
            }
        }

        if (state == PdrTracker.GyroCalState.CALIBRATING) {
            LinearProgressIndicator(
                progress = { progress },
                modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                color = statusColor,
                trackColor = Color(0xFFE5E7EB),
            )
        }

        if (state != PdrTracker.GyroCalState.CALIBRATING) {
            Button(
                onClick = onCalibrate,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = statusColor),
                shape = RoundedCornerShape(10.dp),
            ) {
                Text(
                    when (state) {
                        PdrTracker.GyroCalState.CALIBRATED -> "重新校正"
                        PdrTracker.GyroCalState.FAILED -> "重試校正"
                        else -> "開始校正"
                    },
                    fontWeight = FontWeight.SemiBold,
                )
            }
        }
    }
}

@Composable
private fun GpsStatusCard(lat: Double?, lng: Double?, accuracy: Float?, hasPermission: Boolean, status: GpsConvergeStatus) {
    val tint: Color
    val bg: Color
    val statusLabel: String
    val bar: Float
    when {
        !hasPermission -> { tint = TextTertiary; bg = SurfaceBase; statusLabel = "未授權定位權限"; bar = 0f }
        status == GpsConvergeStatus.WAITING -> { tint = Color(0xFFDC2626); bg = Color(0xFFFFF7ED); statusLabel = "等待衛星訊號…"; bar = 0.1f }
        status == GpsConvergeStatus.CONVERGING -> { tint = Color(0xFFD97706); bg = Color(0xFFFFF7ED); statusLabel = "GPS 收斂中…"; bar = 0.5f }
        else -> { tint = Success; bg = Color(0xFFF0FDF4); statusLabel = "已定位"; bar = 1f }
    }

    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(bg).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(tint.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) {
                Icon(Icons.Default.LocationOn, null, tint = tint, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("GPS", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
                Text(statusLabel, style = MaterialTheme.typography.bodySmall, color = if (status == GpsConvergeStatus.READY) TextSecondary else tint)
                if (status == GpsConvergeStatus.READY && lat != null) {
                    Text(
                        "%.6f, %.6f (±%.0fm)".format(lat, lng, accuracy ?: 0f),
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                    )
                }
                if (status != GpsConvergeStatus.READY && hasPermission) {
                    Text("請在空曠處等待 GPS 定位穩定", fontSize = 12.sp, color = Color(0xFFB45309))
                }
            }
        }
        if (hasPermission) {
            Box(modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)).background(Color(0xFFE5E7EB))) {
                Box(modifier = Modifier.fillMaxHeight().fillMaxWidth(bar).clip(RoundedCornerShape(2.dp)).background(tint))
            }
        }
    }
}

@Composable
private fun LiveReadingsCard(
    stepCount: Int, stepDetected: Int, heading: Float,
    accel: FloatArray, gyro: FloatArray, mag: FloatArray,
    rotationVec: FloatArray, pressure: Float, light: Float,
    grvYaw: Float, rvYaw: Float, compYaw: Float, gyroYaw: Float, calYaw: Float,
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(modifier = Modifier.size(38.dp).background(Gold.copy(alpha = 0.10f), RoundedCornerShape(12.dp)), contentAlignment = Alignment.Center) {
                Icon(Icons.Default.Speed, null, tint = Gold, modifier = Modifier.size(20.dp))
            }
            Spacer(Modifier.width(12.dp))
            Text("即時讀數", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
            BigStat(value = "$stepCount", label = "步數")
            BigStat(value = "%.0f°".format(heading), label = "朝向")
            BigStat(value = if (pressure > 0) "%.1f".format(pressure) else "—", label = "氣壓 hPa")
        }

        HorizontalDivider(color = Border)

        SensorRow("加速度 (m/s²)", "X=%.2f  Y=%.2f  Z=%.2f".format(accel[0], accel[1], accel[2]))
        SensorRow("陀螺儀 (rad/s)", "X=%.4f  Y=%.4f  Z=%.4f".format(gyro[0], gyro[1], gyro[2]))
        SensorRow("磁力 (μT)", "X=%.1f  Y=%.1f  Z=%.1f".format(mag[0], mag[1], mag[2]))
        SensorRow("步伐偵測", "$stepDetected 次")
        SensorRow("旋轉向量", "X=%.3f Y=%.3f Z=%.3f W=%.3f".format(rotationVec[0], rotationVec[1], rotationVec[2], rotationVec[3]))
        SensorRow("光線 (lux)", "%.0f".format(light))

        HorizontalDivider(color = Border)
        Text("Heading 融合", style = MaterialTheme.typography.labelMedium, color = TextTertiary, fontWeight = FontWeight.SemiBold)
        SensorRow("Game RotVec Yaw", "%.1f°".format(grvYaw))
        SensorRow("RotVec Yaw", "%.1f°".format(rvYaw))
        SensorRow("互補濾波 Yaw", "%.1f°".format(compYaw))
        SensorRow("純 Gyro Yaw", "%.1f°".format(gyroYaw))
        SensorRow("校正 Yaw (導航用)", "%.1f°".format(calYaw))
    }
}

@Composable
private fun BigStat(value: String, label: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, fontSize = 28.sp, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, color = TextPrimary)
        Text(label, style = MaterialTheme.typography.labelSmall, color = TextTertiary)
    }
}

@Composable
private fun SensorRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
        Text(value, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace, color = TextPrimary)
    }
}

@Composable
private fun RecordingCard(
    isRecording: Boolean, isPaused: Boolean,
    elapsedSeconds: Double, recordCount: Int, stepCount: Int,
    northCalibrated: Boolean, magAccuracy: Int,
    onStart: () -> Unit, onPause: () -> Unit, onResume: () -> Unit, onStop: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp))
            .background(
                when {
                    isRecording -> Danger.copy(alpha = 0.05f)
                    isPaused -> ChartAmber.copy(alpha = 0.05f)
                    else -> SurfaceBase
                }
            )
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(
                    when {
                        isRecording -> Danger
                        isPaused -> ChartAmber
                        else -> ChartAmber
                    }.copy(alpha = 0.10f), RoundedCornerShape(12.dp)
                ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    when {
                        isRecording -> Icons.Default.Stop
                        isPaused -> Icons.Default.Pause
                        else -> Icons.Default.FiberManualRecord
                    }, null,
                    tint = when {
                        isRecording -> Danger
                        isPaused -> ChartAmber
                        else -> ChartAmber
                    }, modifier = Modifier.size(20.dp),
                )
            }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("記錄控制", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
                when {
                    isRecording -> Text("記錄中 — 請開始行走", style = MaterialTheme.typography.bodySmall, color = Danger)
                    isPaused -> Text("已暫停", style = MaterialTheme.typography.bodySmall, color = ChartAmber)
                }
            }
        }

        if (isRecording || isPaused || recordCount > 0) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                BigStat(value = "%d:%02d".format((elapsedSeconds / 60).toInt(), (elapsedSeconds % 60).toInt()), label = "時間")
                BigStat(value = "$recordCount", label = "資料點")
                BigStat(value = "$stepCount", label = "步數")
            }
        }

        if (isRecording && !northCalibrated) {
            Row(
                modifier = Modifier.fillMaxWidth()
                    .background(ChartAmber.copy(alpha = 0.10f), RoundedCornerShape(12.dp))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(Icons.Default.Warning, null, tint = ChartAmber, modifier = Modifier.size(18.dp))
                Text(
                    if (magAccuracy < SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM)
                        "磁力計精度不足，方向校正尚未完成。請緩慢畫 8 字形校準。"
                    else
                        "等待北方校正中…",
                    style = MaterialTheme.typography.bodySmall,
                    color = ChartAmber,
                )
            }
        }

        if (isRecording && northCalibrated && elapsedSeconds < 3.0) {
            Row(
                modifier = Modifier.fillMaxWidth()
                    .background(Gold.copy(alpha = 0.10f), RoundedCornerShape(12.dp))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(Icons.Default.CheckCircle, null, tint = Gold, modifier = Modifier.size(18.dp))
                Text("北方校正完成", style = MaterialTheme.typography.bodySmall, color = Gold)
            }
        }

        when {
            isRecording -> {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(
                        onClick = onPause,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = ChartAmber),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Icon(Icons.Default.Pause, null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("暫停", fontWeight = FontWeight.SemiBold)
                    }
                    Button(
                        onClick = onStop,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Danger),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Icon(Icons.Default.Stop, null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("停止記錄", fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            isPaused -> {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Button(
                        onClick = onResume,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Gold),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Icon(Icons.Default.PlayArrow, null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("繼續記錄", fontWeight = FontWeight.SemiBold)
                    }
                    Button(
                        onClick = onStart,
                        modifier = Modifier.weight(1f),
                        colors = ButtonDefaults.buttonColors(containerColor = Danger),
                        shape = RoundedCornerShape(14.dp),
                    ) {
                        Icon(Icons.Default.Refresh, null, modifier = Modifier.size(20.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("重新記錄", fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            else -> {
                Button(
                    onClick = onStart,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(containerColor = Gold),
                    shape = RoundedCornerShape(14.dp),
                ) {
                    Icon(Icons.Default.PlayArrow, null, modifier = Modifier.size(20.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("開始記錄", fontWeight = FontWeight.SemiBold)
                }
            }
        }

        if (!isRecording && !isPaused && recordCount == 0) {
            Text(
                "按下「開始記錄」後走一段已知距離的路線，\n停止後輸入實際步數和距離，匯出分析準確率。\n\n記錄中：音量↓ = 標記點位 · 音量↑ = 拍照標記",
                style = MaterialTheme.typography.bodySmall, color = TextTertiary,
                textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
private fun GroundTruthCard(
    groundTruth: GroundTruth, onUpdate: (GroundTruth) -> Unit,
    recordCount: Int, posture: Posture, onExport: () -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(ChartViolet.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) { Icon(Icons.Default.Straighten, null, tint = ChartViolet, modifier = Modifier.size(20.dp)) }
            Spacer(Modifier.width(12.dp))
            Text("Ground Truth", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        Text("走完後填入實際數值，會一併寫入 CSV 方便比對", style = MaterialTheme.typography.bodySmall, color = TextTertiary)

        OutlinedTextField(
            value = groundTruth.steps, onValueChange = { onUpdate(groundTruth.copy(steps = it)) },
            label = { Text("實際步數") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Gold, cursorColor = Gold, focusedLabelColor = Gold),
        )
        OutlinedTextField(
            value = groundTruth.distanceM, onValueChange = { onUpdate(groundTruth.copy(distanceM = it)) },
            label = { Text("實際距離 (公尺)") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Gold, cursorColor = Gold, focusedLabelColor = Gold),
        )
        OutlinedTextField(
            value = groundTruth.weinbergK, onValueChange = { onUpdate(groundTruth.copy(weinbergK = it)) },
            label = { Text("Weinberg K 值（預設 0.4667）") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Gold, cursorColor = Gold, focusedLabelColor = Gold),
        )
        OutlinedTextField(
            value = groundTruth.turns, onValueChange = { onUpdate(groundTruth.copy(turns = it)) },
            label = { Text("轉彎次數") }, modifier = Modifier.fillMaxWidth(), singleLine = true,
            colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Gold, cursorColor = Gold, focusedLabelColor = Gold),
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
            Text("匯出 ZIP ($recordCount 筆 · ${posture.label})", fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun SummaryCard(records: List<SensorRecord>) {
    val first = records.first()
    val last = records.last()
    val duration = last.elapsedSec - first.elapsedSec
    val steps = last.totalSteps - first.totalSteps

    val stats = remember(records.size) {
        val headingChanges = records.asSequence().zipWithNext().sumOf { (a, b) ->
            var diff = (b.heading - a.heading).toDouble()
            if (diff > 180) diff -= 360
            if (diff < -180) diff += 360
            kotlin.math.abs(diff)
        }
        val avgGyro = records.asSequence().map {
            kotlin.math.sqrt((it.gyroX * it.gyroX + it.gyroY * it.gyroY + it.gyroZ * it.gyroZ).toDouble())
        }.average()
        val pdrDist = if (last.pdrX != 0f || last.pdrY != 0f) {
            records.asSequence().zipWithNext().sumOf { (a, b) ->
                kotlin.math.sqrt(((b.pdrX - a.pdrX) * (b.pdrX - a.pdrX) + (b.pdrY - a.pdrY) * (b.pdrY - a.pdrY)).toDouble())
            }
        } else steps * 0.68
        Triple(headingChanges, avgGyro, pdrDist)
    }
    val headingChanges = stats.first
    val avgGyro = stats.second
    val pdrDist = stats.third

    val pressureChange = if (first.pressure > 0 && last.pressure > 0) last.pressure - first.pressure else 0f

    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(Success.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) { Icon(Icons.Default.Analytics, null, tint = Success, modifier = Modifier.size(20.dp)) }
            Spacer(Modifier.width(12.dp))
            Text("記錄摘要", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
        }

        HorizontalDivider(color = Border)

        SummaryRow("記錄時長", "%.1f 秒".format(duration))
        SummaryRow("計步器步數", "$steps 步")
        SummaryRow("步伐偵測器", "${last.stepDetected} 次")
        SummaryRow("估計步頻", if (duration > 0) "%.1f 步/秒".format(steps / duration) else "—")
        SummaryRow("PDR 距離 (Weinberg)", "%.1f m".format(pdrDist))
        if (steps > 0) {
            SummaryRow("平均步幅", "%.2f m/步".format(pdrDist / steps))
        }
        SummaryRow("累計朝向變化", "%.0f°".format(headingChanges))
        SummaryRow("平均陀螺儀幅值", "%.4f rad/s".format(avgGyro))
        SummaryRow("起始朝向", "%.0f°".format(first.heading))
        SummaryRow("結束朝向", "%.0f°".format(last.heading))
        if (first.pressure > 0) {
            SummaryRow("氣壓變化", "%.2f hPa".format(pressureChange))
            SummaryRow("估計樓層變化", "%.1f 層".format(pressureChange / -0.4f))
        }

        HorizontalDivider(color = Border)
        Text("Heading 比較 (終點)", style = MaterialTheme.typography.labelMedium, color = TextTertiary, fontWeight = FontWeight.SemiBold)
        SummaryRow("Azimuth (Accel+Mag)", "%.1f°".format(last.heading))
        SummaryRow("Game RotVec Yaw", "%.1f°".format(last.grvYawDeg))
        SummaryRow("RotVec Yaw", "%.1f°".format(last.rvYawDeg))
        SummaryRow("互補濾波 Yaw", "%.1f°".format(last.compYawDeg))
        SummaryRow("純 Gyro Yaw", "%.1f°".format(last.gyroYawDeg))
        SummaryRow("校正 Yaw", "%.1f°".format(last.calYawDeg))

        if (last.gpsLat != null) {
            HorizontalDivider(color = Border)
            SummaryRow("GPS 終點", "%.6f, %.6f".format(last.gpsLat, last.gpsLng))
            SummaryRow("GPS 精度", "±%.0fm".format(last.gpsAccM ?: 0f))
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
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(ChartBlue.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) { Icon(Icons.Default.History, null, tint = ChartBlue, modifier = Modifier.size(20.dp)) }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("歷史記錄", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
                Text("${logs.size} 筆記錄", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
            }
        }

        logs.forEach { info ->
            var showConfirmDelete by remember { mutableStateOf(false) }

            Column(
                modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Noir).padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Column(modifier = Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(info.date, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                            if (info.posture.isNotBlank()) {
                                Text(
                                    info.posture.replace("_", " "),
                                    style = MaterialTheme.typography.labelSmall, color = ChartViolet,
                                    modifier = Modifier.background(ChartViolet.copy(alpha = 0.10f), RoundedCornerShape(4.dp)).padding(horizontal = 6.dp, vertical = 2.dp),
                                )
                            }
                        }
                        val photoLabel = if (info.photoCount > 0) " · ${info.photoCount} 張照片" else ""
                        Text("${info.dataPoints} 筆 · ${info.sizeKb}$photoLabel", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        IconButton(onClick = { onShare(info) }, modifier = Modifier.size(36.dp)) {
                            Icon(Icons.Default.Share, "分享", tint = Info, modifier = Modifier.size(18.dp))
                        }
                        IconButton(onClick = { showConfirmDelete = true }, modifier = Modifier.size(36.dp)) {
                            Icon(Icons.Default.Delete, "刪除", tint = Danger, modifier = Modifier.size(18.dp))
                        }
                    }
                }

                if (info.groundTruthSteps.isNotBlank() || info.groundTruthDistance.isNotBlank()) {
                    Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                        if (info.groundTruthSteps.isNotBlank()) {
                            Text("實際步數: ${info.groundTruthSteps}", style = MaterialTheme.typography.labelSmall, color = Gold)
                        }
                        if (info.groundTruthDistance.isNotBlank()) {
                            Text("實際距離: ${info.groundTruthDistance}m", style = MaterialTheme.typography.labelSmall, color = Gold)
                        }
                    }
                }

                if (showConfirmDelete) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.End)) {
                        TextButton(onClick = { showConfirmDelete = false }) { Text("取消", color = TextSecondary) }
                        Button(
                            onClick = { showConfirmDelete = false; onDelete(info) },
                            colors = ButtonDefaults.buttonColors(containerColor = Danger),
                            shape = RoundedCornerShape(10.dp),
                        ) { Text("確認刪除") }
                    }
                }
            }
        }
    }
}

@Composable
private fun PathTrailCard(
    pathPoints: List<PdrPoint>,
    waypoints: List<WaypointMarker>,
    currentX: Float, currentY: Float, currentHeading: Float,
    isRecording: Boolean, stepCount: Int,
    totalDistance: Float, lastStride: Float,
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 20.dp)
            .clip(RoundedCornerShape(20.dp)).background(SurfaceBase).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier.size(38.dp).background(ChartBlue.copy(alpha = 0.10f), RoundedCornerShape(12.dp)),
                contentAlignment = Alignment.Center
            ) { Icon(Icons.Default.Route, null, tint = ChartBlue, modifier = Modifier.size(20.dp)) }
            Spacer(Modifier.width(12.dp))
            Column {
                Text("行走軌跡", style = MaterialTheme.typography.titleLarge, color = TextPrimary)
                Text(
                    "%.1fm · %d 步 · %d 標記".format(totalDistance, stepCount, waypoints.size),
                    style = MaterialTheme.typography.bodySmall, color = TextTertiary,
                )
            }
        }

        if (pathPoints.size > 2) {
            val startToEnd = kotlin.math.sqrt(
                ((pathPoints.last().x - pathPoints.first().x).let { it * it } +
                 (pathPoints.last().y - pathPoints.first().y).let { it * it }).toDouble()
            )
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                BigStat(value = "%.1fm".format(totalDistance), label = "總距離")
                BigStat(value = "%.1fm".format(startToEnd), label = "直線距離")
                BigStat(value = "%d".format(waypoints.size), label = "標記點")
            }
        }

        // Canvas
        val trailColor = ChartBlue
        val startColor = Success
        val currentColor = Danger
        val waypointColor = ChartAmber
        val photoWpColor = ChartBlue
        val gridColor = Border
        val textColor = TextSecondary

        Canvas(
            modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(12.dp)).background(Noir)
        ) {
            if (pathPoints.size < 2) {
                drawContext.canvas.nativeCanvas.drawText(
                    "等待步行數據...", size.width / 2 - 100f, size.height / 2,
                    NativePaint().apply { color = textColor.toArgb(); textSize = 32f; isAntiAlias = true }
                )
                return@Canvas
            }

            val allX = pathPoints.map { it.x }
            val allY = pathPoints.map { it.y }
            val minX = allX.min(); val maxX = allX.max()
            val minY = allY.min(); val maxY = allY.max()
            val rangeX = (maxX - minX).coerceAtLeast(5f)
            val rangeY = (maxY - minY).coerceAtLeast(5f)
            val padding = 50f

            val scaleX = (size.width - 2 * padding) / rangeX
            val scaleY = (size.height - 2 * padding) / rangeY
            val scale = minOf(scaleX, scaleY)
            val centerX = (minX + maxX) / 2
            val centerY = (minY + maxY) / 2

            fun toCanvas(px: Float, py: Float): Offset {
                return Offset(size.width / 2 + (px - centerX) * scale, size.height / 2 - (py - centerY) * scale)
            }

            // Grid
            val gridSpacing = when {
                rangeX / scale > 100 -> 50f; rangeX / scale > 20 -> 10f; else -> 5f
            }
            val dashEffect = PathEffect.dashPathEffect(floatArrayOf(4f, 8f))
            var gx = (minX / gridSpacing).toInt() * gridSpacing - gridSpacing
            while (gx <= maxX + gridSpacing) {
                val p = toCanvas(gx, 0f)
                drawLine(gridColor.copy(alpha = 0.3f), Offset(p.x, 0f), Offset(p.x, size.height), strokeWidth = 1f, pathEffect = dashEffect)
                gx += gridSpacing
            }
            var gy = (minY / gridSpacing).toInt() * gridSpacing - gridSpacing
            while (gy <= maxY + gridSpacing) {
                val p = toCanvas(0f, gy)
                drawLine(gridColor.copy(alpha = 0.3f), Offset(0f, p.y), Offset(size.width, p.y), strokeWidth = 1f, pathEffect = dashEffect)
                gy += gridSpacing
            }

            // Path trail
            for (i in 0 until pathPoints.size - 1) {
                val from = toCanvas(pathPoints[i].x, pathPoints[i].y)
                val to = toCanvas(pathPoints[i + 1].x, pathPoints[i + 1].y)
                val progress = i.toFloat() / pathPoints.size
                drawLine(trailColor.copy(alpha = 0.4f + 0.6f * progress), from, to, strokeWidth = 3f, cap = StrokeCap.Round)
            }

            // Waypoint markers
            val labelPaint = NativePaint().apply { color = waypointColor.toArgb(); textSize = 28f; isAntiAlias = true }
            val photoLabelPaint = NativePaint().apply { color = photoWpColor.toArgb(); textSize = 28f; isAntiAlias = true }
            for (wp in waypoints) {
                val pos = toCanvas(wp.x, wp.y)
                val isPhoto = wp.photoFiles.isNotEmpty()
                val wpColor = if (isPhoto) photoWpColor else waypointColor
                drawCircle(color = wpColor, center = pos, radius = 8f)
                drawCircle(color = wpColor.copy(alpha = 0.2f), center = pos, radius = 16f)
                drawContext.canvas.nativeCanvas.drawText(
                    wp.name, pos.x + 14f, pos.y - 14f,
                    if (isPhoto) photoLabelPaint else labelPaint,
                )
            }

            // Start point
            val startPos = toCanvas(pathPoints.first().x, pathPoints.first().y)
            drawCircle(color = startColor, center = startPos, radius = 10f)
            drawCircle(color = startColor.copy(alpha = 0.3f), center = startPos, radius = 18f)
            drawContext.canvas.nativeCanvas.drawText(
                "起點", startPos.x + 14f, startPos.y - 14f,
                NativePaint().apply { color = startColor.toArgb(); textSize = 26f; isAntiAlias = true },
            )

            // Current position
            val curPos = toCanvas(pathPoints.last().x, pathPoints.last().y)
            drawCircle(color = currentColor, center = curPos, radius = 10f)
            drawCircle(color = currentColor.copy(alpha = 0.3f), center = curPos, radius = 18f)
            val arrowLen = 30f
            val headingRad = Math.toRadians(currentHeading.toDouble())
            val arrowEnd = Offset(
                curPos.x + (arrowLen * kotlin.math.sin(headingRad)).toFloat(),
                curPos.y - (arrowLen * kotlin.math.cos(headingRad)).toFloat(),
            )
            drawLine(currentColor, curPos, arrowEnd, strokeWidth = 3f, cap = StrokeCap.Round)

            // Scale bar
            val scaleBarMeters = gridSpacing
            val scaleBarPx = scaleBarMeters * scale
            val barY = size.height - 20f; val barX = 20f
            drawLine(textColor, Offset(barX, barY), Offset(barX + scaleBarPx, barY), strokeWidth = 2f)
            drawLine(textColor, Offset(barX, barY - 5f), Offset(barX, barY + 5f), strokeWidth = 2f)
            drawLine(textColor, Offset(barX + scaleBarPx, barY - 5f), Offset(barX + scaleBarPx, barY + 5f), strokeWidth = 2f)
            drawContext.canvas.nativeCanvas.drawText(
                "%.0fm".format(scaleBarMeters), barX + scaleBarPx / 2 - 20f, barY - 10f,
                NativePaint().apply { color = textColor.toArgb(); textSize = 22f; isAntiAlias = true },
            )

            // North arrow
            val northX = size.width - 40f; val northY = 40f
            drawLine(textColor, Offset(northX, northY + 20f), Offset(northX, northY - 10f), strokeWidth = 2f)
            drawContext.canvas.nativeCanvas.drawText(
                "N", northX - 8f, northY - 14f,
                NativePaint().apply { color = textColor.toArgb(); textSize = 24f; isAntiAlias = true; isFakeBoldText = true },
            )
        }

        // Waypoint list
        if (waypoints.isNotEmpty()) {
            HorizontalDivider(color = Border)
            waypoints.forEachIndexed { i, wp ->
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Box(
                            modifier = Modifier.size(22.dp).background(
                                (if (wp.photoFiles.isNotEmpty()) ChartBlue else ChartAmber).copy(alpha = 0.15f), CircleShape
                            ),
                            contentAlignment = Alignment.Center,
                        ) {
                            if (wp.photoFiles.isNotEmpty()) {
                                Icon(Icons.Default.CameraAlt, null, modifier = Modifier.size(12.dp), tint = ChartBlue)
                            } else {
                                Text("${i + 1}", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = ChartAmber)
                            }
                        }
                        Text(wp.name, style = MaterialTheme.typography.bodyMedium, color = TextPrimary)
                        if (wp.photoFiles.isNotEmpty()) {
                            Text("${wp.photoFiles.size}張", fontSize = 11.sp, color = ChartBlue)
                        }
                    }
                    Text(
                        "%.0fs · %d步 · (%.1f, %.1f)".format(wp.elapsedSec, wp.stepIndex, wp.x, wp.y),
                        style = MaterialTheme.typography.labelSmall, fontFamily = FontFamily.Monospace, color = TextTertiary,
                    )
                }
            }
        }
    }
}

@Composable
private fun SummaryRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = TextSecondary)
        Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, fontFamily = FontFamily.Monospace, color = TextPrimary)
    }
}
