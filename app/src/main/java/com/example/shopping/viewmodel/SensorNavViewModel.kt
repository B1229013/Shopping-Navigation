package com.example.shopping.viewmodel

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.SensorManager
import android.net.Uri
import android.os.Looper
import android.util.Log
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.example.shopping.network.AnswerRequest
import com.example.shopping.network.ConfirmArrivalRequest
import com.example.shopping.network.ConfirmLocationRequest
import com.example.shopping.network.Neo4jPlaceInfo
import com.example.shopping.network.SNavMapInfo
import com.example.shopping.network.SNavStartRequest
import com.example.shopping.network.SNavSubGoalInfo
import com.example.shopping.network.SNavTurnResponse
import com.example.shopping.network.SaveMapRequest
import com.example.shopping.network.SensorNavApi
import com.example.shopping.sensor.LocalSession
import com.example.shopping.sensor.LocalSessionStore
import com.example.shopping.sensor.PdrPoint
import com.example.shopping.sensor.PdrTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

enum class LocalizationPhase { READY, PROCESSING, CONFIRMING, NAVIGATING }

class SensorNavViewModel : ViewModel() {

    // ── Mode ──
    var isOfflineMode: Boolean by mutableStateOf(false)
    private var localStore: LocalSessionStore? = null
    private var localSessionId: String? = null

    // ── Navigation state ──
    var sessionId: String? by mutableStateOf(null)
    var goal: String by mutableStateOf("")
    var guidance: String by mutableStateOf("")
    var currentAction: String by mutableStateOf("TAKE_PHOTO")
    var pendingQuestion: String? by mutableStateOf(null)
    var answerText: String by mutableStateOf("")
    var isUploading: Boolean by mutableStateOf(false)
    var isAnswering: Boolean by mutableStateOf(false)
    var hasArrived: Boolean by mutableStateOf(false)
    var pendingArrival: Boolean by mutableStateOf(false)
    var isConfirming: Boolean by mutableStateOf(false)
    var annotatedPhotoUrl: String? by mutableStateOf(null)
    var errorMessage: String? by mutableStateOf(null)
    var photoCount: Int by mutableIntStateOf(0)

    // ── PDR state ──
    var pdrTracker: PdrTracker? = null; private set
    var pdrX: Float by mutableFloatStateOf(0f)
    var pdrY: Float by mutableFloatStateOf(0f)
    var pdrTotalSteps: Int by mutableIntStateOf(0)
    var pdrTotalDistance: Float by mutableFloatStateOf(0f)
    val pdrPathPoints: List<PdrPoint> get() = pdrTracker?.pathPoints ?: emptyList()
    var magCalibrationStatus: Int by mutableIntStateOf(SensorManager.SENSOR_STATUS_UNRELIABLE)
    var northCalibrated: Boolean by mutableStateOf(false)
    var gyroCalState: PdrTracker.GyroCalState by mutableStateOf(PdrTracker.GyroCalState.UNCALIBRATED)
    var gyroCalProgress: Float by mutableFloatStateOf(0f)
    var gyroBias: FloatArray by mutableStateOf(floatArrayOf(0f, 0f, 0f))
    var gyroCalNoise: Float by mutableFloatStateOf(0f)

    // Photo capture nodes (for topomap graph)
    data class PhotoNode(val index: Int, val x: Float, val y: Float, val totalSteps: Int, val segmentSteps: Int, val segmentDistanceM: Float)
    private val _photoNodes = mutableListOf<PhotoNode>()
    val photoNodes: List<PhotoNode> get() = _photoNodes.toList()

    // ── Step guidance ──
    var stepGuidance: String? by mutableStateOf(null)
    var estimatedRemainingSteps: Int? by mutableStateOf(null)

    // ── Maps ──
    var availableMaps: List<SNavMapInfo> by mutableStateOf(emptyList())
    var isLoadingMaps: Boolean by mutableStateOf(false)
    var isSavingMap: Boolean by mutableStateOf(false)
    var savedMapId: String? by mutableStateOf(null)

    // ── Neo4j places ──
    var neo4jPlaces: List<Neo4jPlaceInfo> by mutableStateOf(emptyList())
    var isLoadingPlaces: Boolean by mutableStateOf(false)
    var navMode: String by mutableStateOf("explore")
    var goalPhotoIds: List<Int> by mutableStateOf(emptyList())

    // ── Localization ──
    var localizationPhase: LocalizationPhase by mutableStateOf(LocalizationPhase.READY)
    var isConfirmingLocation: Boolean by mutableStateOf(false)
    var localizedPhotoId: Int? by mutableStateOf(null)
    var localizationScore: Double? by mutableStateOf(null)
    var localizationCandidates: List<Map<String, Any>> by mutableStateOf(emptyList())

    // ── Route planning ──
    var plannedRoute: List<Int> by mutableStateOf(emptyList())
    var routeTargetPhotoId: Int? by mutableStateOf(null)
    var routeDistance: Double? by mutableStateOf(null)
    var routeHops: Int? by mutableStateOf(null)
    var routeGuidance: String? by mutableStateOf(null)
    var routeWaypoints: List<Map<String, Any>> by mutableStateOf(emptyList())

    // ── Multi-goal progress ──
    var subGoals: List<SNavSubGoalInfo> by mutableStateOf(emptyList())
    var currentGoalIdx: Int by mutableIntStateOf(0)
    var currentGoalName: String by mutableStateOf("")
    var totalGoals: Int by mutableIntStateOf(1)

    // ── Local sessions ──
    var localSessions: List<LocalSession> by mutableStateOf(emptyList())

    // ── GPS convergence filter ──
    enum class GpsStatus { WAITING, CONVERGING, READY }
    var gpsStatus: GpsStatus by mutableStateOf(GpsStatus.WAITING)

    private var fusedLocationClient: com.google.android.gms.location.FusedLocationProviderClient? = null
    private val gpsConvergenceWindow = mutableListOf<Triple<Double, Double, Float>>() // lat, lng, accuracy
    private companion object {
        const val GPS_ACCURACY_THRESHOLD = 25f   // metres
        const val GPS_CONVERGE_COUNT = 3         // consecutive good fixes needed
        const val GPS_CONVERGE_RADIUS = 10.0     // metres between fixes
    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            val loc = result.lastLocation ?: return

            if (loc.accuracy > GPS_ACCURACY_THRESHOLD) {
                Log.d("SensorNav", "GPS fix dropped: accuracy=${loc.accuracy}m > ${GPS_ACCURACY_THRESHOLD}m")
                return
            }

            if (gpsStatus != GpsStatus.READY) {
                gpsConvergenceWindow.add(Triple(loc.latitude, loc.longitude, loc.accuracy))
                if (gpsConvergenceWindow.size > GPS_CONVERGE_COUNT)
                    gpsConvergenceWindow.removeAt(0)

                if (gpsConvergenceWindow.size >= GPS_CONVERGE_COUNT && isConverged()) {
                    gpsStatus = GpsStatus.READY
                    Log.i("SensorNav", "GPS converged, accuracy=${loc.accuracy}m")
                } else {
                    gpsStatus = GpsStatus.CONVERGING
                    Log.d("SensorNav", "GPS converging: ${gpsConvergenceWindow.size}/$GPS_CONVERGE_COUNT fixes")
                    return
                }
            }

            pdrTracker?.updateLocation(loc.latitude, loc.longitude, loc.accuracy, loc.altitude)
        }
    }

    private fun isConverged(): Boolean {
        if (gpsConvergenceWindow.size < GPS_CONVERGE_COUNT) return false
        val anchor = gpsConvergenceWindow.last()
        return gpsConvergenceWindow.all { fix ->
            haversineMetres(anchor.first, anchor.second, fix.first, fix.second) < GPS_CONVERGE_RADIUS
        }
    }

    private fun haversineMetres(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val r = 6_371_000.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                Math.sin(dLon / 2) * Math.sin(dLon / 2)
        return r * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
    }

    // ── Camera ──
    val imageCapture = ImageCapture.Builder().build()
    val captureExecutor = Executors.newSingleThreadExecutor()!!

    // ── Init ──

    fun startSession(context: Context, goalText: String, mapId: String? = null, placeName: String? = null) {
        goal = goalText
        errorMessage = null

        if (isOfflineMode) {
            startOfflineSession(context, goalText)
            return
        }

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    SensorNavApi.service.startSession(
                        SNavStartRequest(goal = goalText, map_id = mapId, place_name = placeName)
                    )
                }
                sessionId = response.session_id
                guidance = response.guidance
                currentAction = response.action
                navMode = response.nav_mode
                goalPhotoIds = response.goal_photo_ids
                subGoals = response.sub_goals
                currentGoalIdx = response.current_goal_idx
                currentGoalName = response.current_goal_name
                totalGoals = response.total_goals
                startPdr(context)
            } catch (e: Exception) {
                Log.e("SensorNav", "Start session failed", e)
                errorMessage = "建立導航失敗: ${e.message}"
            }
        }
    }

    private fun startOfflineSession(context: Context, goalText: String) {
        val store = LocalSessionStore(context).also { localStore = it }
        val id = store.createSession(goalText)
        localSessionId = id
        sessionId = id
        guidance = "離線採集模式：走路拍照記錄路線，照片與感測器資料會存在手機裡。"
        currentAction = "TAKE_PHOTO"
        startPdr(context)
    }

    fun startPdr(context: Context) {
        if (pdrTracker != null) return
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        pdrTracker = PdrTracker(sm).also {
            it.onMagAccuracyChanged = { accuracy -> magCalibrationStatus = accuracy }
            it.onNorthCalibrated = { northCalibrated = true }
            it.onGyroCalStateChanged = { state ->
                gyroCalState = state
                gyroCalProgress = it.gyroCalProgress
                gyroBias = it.gyroBias.copyOf()
                gyroCalNoise = it.gyroCalNoise
            }
            it.start()
        }
        startGps(context)
    }

    fun startGyroCalibration() {
        pdrTracker?.startGyroCalibration()
    }

    private fun startGps(context: Context) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED) {
            Log.w("SensorNav", "No location permission, GPS not recorded")
            return
        }
        val client = LocationServices.getFusedLocationProviderClient(context)
        fusedLocationClient = client
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L)
            .setMinUpdateIntervalMillis(500L)
            .build()
        client.requestLocationUpdates(request, locationCallback, Looper.getMainLooper())
        Log.i("SensorNav", "GPS recording started")
    }

    fun stopPdr() {
        pdrTracker?.stop()
        pdrTracker = null
        fusedLocationClient?.removeLocationUpdates(locationCallback)
        fusedLocationClient = null
        gpsConvergenceWindow.clear()
        gpsStatus = GpsStatus.WAITING
    }

    // ── Photo capture ──

    fun captureAndUpload(context: Context) {
        val sid = sessionId ?: return
        if (isUploading || hasArrived || pendingArrival || pendingQuestion != null) return
        isUploading = true
        errorMessage = null
        localizationPhase = LocalizationPhase.PROCESSING

        val photoFile = File(context.cacheDir, "snav_photo_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                viewModelScope.launch {
                    if (isOfflineMode) {
                        savePhotoLocally(photoFile)
                    } else {
                        uploadPhotoFile(sid, photoFile)
                    }
                }
            }

            override fun onError(e: ImageCaptureException) {
                Log.e("SensorNav", "Capture failed", e)
                isUploading = false
                errorMessage = "拍照失敗: ${e.message}"
            }
        })
    }

    fun uploadFromGallery(context: Context, uri: Uri) {
        val sid = sessionId ?: return
        if (isUploading || hasArrived || pendingArrival || pendingQuestion != null) return
        isUploading = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val photoFile = withContext(Dispatchers.IO) {
                    val f = File(context.cacheDir, "snav_gallery_${System.currentTimeMillis()}.jpg")
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(f).use { output -> input.copyTo(output) }
                    } ?: throw Exception("無法讀取照片")
                    f
                }
                if (isOfflineMode) {
                    savePhotoLocally(photoFile)
                } else {
                    uploadPhotoFile(sid, photoFile)
                }
            } catch (e: Exception) {
                Log.e("SensorNav", "Gallery upload failed", e)
                isUploading = false
                errorMessage = "上傳失敗: ${e.message}"
            }
        }
    }

    // ── Offline: save locally ──

    private suspend fun savePhotoLocally(photoFile: File) {
        try {
            val pdrSnap = pdrTracker?.snapshot()
            updatePdrState()

            val index = withContext(Dispatchers.IO) {
                localStore?.saveEntry(localSessionId!!, photoFile, pdrSnap) ?: -1
            }
            if (index > 0) {
                photoCount = index
                _photoNodes.add(PhotoNode(
                    index = index,
                    x = pdrX, y = pdrY,
                    totalSteps = pdrTotalSteps,
                    segmentSteps = pdrSnap?.segmentSteps ?: 0,
                    segmentDistanceM = pdrSnap?.segmentDistanceM ?: 0f,
                ))
                guidance = "已儲存第 $index 張照片（${pdrTotalSteps} 步 · ${"%.1f".format(pdrTotalDistance)}m）"
            }
        } catch (e: Exception) {
            Log.e("SensorNav", "Local save failed", e)
            errorMessage = "本地儲存失敗: ${e.message}"
        } finally {
            isUploading = false
            photoFile.delete()
        }
    }

    // ── Online: upload to server ──

    private suspend fun uploadPhotoFile(sessionId: String, photoFile: File) {
        try {
            val pdrSnap = pdrTracker?.snapshot()
            updatePdrState()

            val response = withContext(Dispatchers.IO) {
                val requestBody = photoFile.asRequestBody("image/jpeg".toMediaType())
                val part = MultipartBody.Part.createFormData("photo", photoFile.name, requestBody)

                val pdrBody = if (pdrSnap != null) {
                    val json = JSONObject().apply {
                        put("steps", pdrSnap.segmentSteps)
                        put("distance_m", pdrSnap.segmentDistanceM)
                        put("heading_deg", pdrSnap.avgHeadingDeg)
                        put("pdr_x", pdrSnap.currentX)
                        put("pdr_y", pdrSnap.currentY)
                        put("total_steps", pdrSnap.totalSteps)
                        put("total_distance_m", pdrSnap.totalDistanceM)
                        put("raw_sensors", pdrSnap.rawSensorJson)
                        put("game_rot_vec_yaw_deg", pdrSnap.gameRotVecYawDeg)
                        put("rot_vec_yaw_deg", pdrSnap.rotVecYawDeg)
                        put("azimuth_deg", pdrSnap.azimuthDeg)
                        put("complementary_yaw_deg", pdrSnap.complementaryYawDeg)
                        put("gyro_only_yaw_deg", pdrSnap.gyroOnlyYawDeg)
                        put("calibrated_yaw_deg", pdrSnap.calibratedYawDeg)
                        put("gps_lat", pdrSnap.gpsLat ?: JSONObject.NULL)
                        put("gps_lng", pdrSnap.gpsLng ?: JSONObject.NULL)
                        put("gps_accuracy", pdrSnap.gpsAccuracy ?: JSONObject.NULL)
                    }.toString()
                    json.toRequestBody("text/plain".toMediaType())
                } else null

                SensorNavApi.service.uploadPhoto(sessionId, part, pdrBody)
            }
            handleTurnResponse(response)
            photoCount++
            _photoNodes.add(PhotoNode(
                index = photoCount,
                x = pdrX, y = pdrY,
                totalSteps = pdrTotalSteps,
                segmentSteps = pdrSnap?.segmentSteps ?: 0,
                segmentDistanceM = pdrSnap?.segmentDistanceM ?: 0f,
            ))
        } catch (e: Exception) {
            Log.e("SensorNav", "Upload failed", e)
            errorMessage = "上傳失敗: ${e.message}"
        } finally {
            isUploading = false
            photoFile.delete()
        }
    }

    // ── Answer ──

    fun submitAnswer() {
        val sid = sessionId ?: return
        if (answerText.isBlank() || isAnswering) return
        isAnswering = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    SensorNavApi.service.postAnswer(sid, AnswerRequest(answer = answerText))
                }
                handleTurnResponse(response)
                answerText = ""
            } catch (e: Exception) {
                Log.e("SensorNav", "Answer failed", e)
                errorMessage = "回答提交失敗: ${e.message}"
            } finally {
                isAnswering = false
            }
        }
    }

    // ── Confirm arrival ──

    fun confirmArrival(kind: String) {
        val sid = sessionId ?: return
        if (isConfirming) return
        isConfirming = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    SensorNavApi.service.confirmArrival(sid, ConfirmArrivalRequest(kind = kind))
                }
                pendingArrival = false
                handleTurnResponse(response)
                if (response.action == "ARRIVED" && response.sub_goals.all { it.arrived }) {
                    hasArrived = true
                    stopPdr()
                }
            } catch (e: Exception) {
                Log.e("SensorNav", "Confirm failed", e)
                errorMessage = "確認失敗: ${e.message}"
            } finally {
                isConfirming = false
            }
        }
    }

    // ── Map management ──

    fun loadMaps() {
        isLoadingMaps = true
        viewModelScope.launch {
            try {
                availableMaps = withContext(Dispatchers.IO) {
                    SensorNavApi.service.listMaps()
                }
            } catch (e: Exception) {
                Log.e("SensorNav", "Load maps failed", e)
                errorMessage = "載入地圖列表失敗: ${e.message}"
            } finally {
                isLoadingMaps = false
            }
        }
    }

    fun loadNeo4jPlaces() {
        isLoadingPlaces = true
        viewModelScope.launch {
            try {
                neo4jPlaces = withContext(Dispatchers.IO) {
                    SensorNavApi.service.listNeo4jPlaces()
                }
            } catch (e: Exception) {
                Log.e("SensorNav", "Load Neo4j places failed", e)
            } finally {
                isLoadingPlaces = false
            }
        }
    }

    fun saveMap(name: String) {
        val sid = sessionId ?: return
        isSavingMap = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    SensorNavApi.service.saveMap(SaveMapRequest(session_id = sid, name = name))
                }
                savedMapId = response.map_id
            } catch (e: Exception) {
                Log.e("SensorNav", "Save map failed", e)
                errorMessage = "儲存地圖失敗: ${e.message}"
            } finally {
                isSavingMap = false
            }
        }
    }

    // ── Local session management ──

    fun loadLocalSessions(context: Context) {
        viewModelScope.launch {
            localSessions = withContext(Dispatchers.IO) {
                LocalSessionStore(context).listSessions()
            }
        }
    }

    fun deleteLocalSession(context: Context, id: String) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                LocalSessionStore(context).deleteSession(id)
            }
            loadLocalSessions(context)
        }
    }

    fun shareLocalSession(context: Context, id: String) {
        viewModelScope.launch {
            val intent = withContext(Dispatchers.IO) {
                LocalSessionStore(context).shareSession(id)
            }
            if (intent != null) {
                context.startActivity(Intent.createChooser(intent, "匯出採集資料"))
            }
        }
    }

    fun finishOfflineSession() {
        val path = pdrTracker?.pathPoints ?: emptyList()
        viewModelScope.launch(Dispatchers.IO) {
            localStore?.savePdrPath(localSessionId ?: return@launch, path)
        }
        hasArrived = true
        guidance = "採集完成！共 $photoCount 張照片，${pdrTotalSteps} 步，${"%.1f".format(pdrTotalDistance)}m。資料已存於手機。"
        stopPdr()
    }

    // ── Confirm localization ──

    fun confirmLocation(photoId: Int) {
        val sid = sessionId ?: return
        if (isConfirmingLocation) return
        isConfirmingLocation = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    SensorNavApi.service.confirmLocation(sid, ConfirmLocationRequest(photo_id = photoId))
                }
                localizedPhotoId = response.localized_photo_id
                goalPhotoIds = response.goal_photo_ids
                routeGuidance = response.route_guidance
                routeWaypoints = response.route_waypoints
                routeDistance = response.route_distance
                routeHops = response.route_hops
                guidance = response.guidance
                if (response.sub_goals.isNotEmpty()) {
                    subGoals = response.sub_goals
                }
                currentGoalIdx = response.current_goal_idx
                currentGoalName = response.current_goal_name
                totalGoals = response.total_goals
                localizationPhase = LocalizationPhase.NAVIGATING
            } catch (e: Exception) {
                Log.e("SensorNav", "Confirm location failed", e)
                errorMessage = "位置確認失敗: ${e.message}"
            } finally {
                isConfirmingLocation = false
            }
        }
    }

    fun retakePhoto() {
        localizationPhase = LocalizationPhase.READY
        localizationCandidates = emptyList()
        localizedPhotoId = null
        localizationScore = null
        guidance = "請重新拍一張照片進行定位"
    }

    // ── Internal ──

    private fun handleTurnResponse(response: SNavTurnResponse) {
        guidance = response.guidance
        currentAction = response.action
        annotatedPhotoUrl = response.annotated_photo_url
        stepGuidance = response.step_guidance
        estimatedRemainingSteps = response.estimated_remaining_steps

        // Update localization
        localizedPhotoId = response.localized_photo_id
        localizationScore = response.localization_score
        localizationCandidates = response.localization_candidates
        if (response.goal_photo_ids.isNotEmpty()) {
            goalPhotoIds = response.goal_photo_ids
        }

        // Enter CONFIRMING phase when candidates arrive
        if (response.localization_candidates.isNotEmpty() &&
            localizationPhase != LocalizationPhase.NAVIGATING) {
            localizationPhase = LocalizationPhase.CONFIRMING
        }

        // Update route planning
        plannedRoute = response.planned_route
        routeTargetPhotoId = response.route_target_photo_id
        routeDistance = response.route_distance
        routeHops = response.route_hops
        routeGuidance = response.route_guidance
        routeWaypoints = response.route_waypoints

        // Update multi-goal progress
        if (response.sub_goals.isNotEmpty()) {
            subGoals = response.sub_goals
        }
        currentGoalIdx = response.current_goal_idx
        currentGoalName = response.current_goal_name
        totalGoals = response.total_goals

        when (response.action) {
            "ARRIVED" -> {
                pendingArrival = true
                pendingQuestion = null
            }
            "ASK" -> {
                pendingQuestion = response.question
            }
            "MOVE" -> {
                pendingQuestion = null
                pendingArrival = false
            }
        }
    }

    private fun updatePdrState() {
        pdrTracker?.let {
            pdrX = it.x
            pdrY = it.y
            pdrTotalSteps = it.totalSteps
            pdrTotalDistance = it.totalDistance
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopPdr()
        captureExecutor.shutdown()
    }
}
