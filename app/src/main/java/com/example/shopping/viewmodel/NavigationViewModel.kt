package com.example.shopping.viewmodel

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.shopping.network.AnswerRequest
import com.example.shopping.network.ConfirmArrivalRequest
import com.example.shopping.network.NavConfirmLocationRequest
import com.example.shopping.network.NavSubGoalInfo
import com.example.shopping.network.NavigationApi
import com.example.shopping.network.TurnResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

class NavigationViewModel : ViewModel() {

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

    // Localization
    var localizationPhase: LocalizationPhase by mutableStateOf(LocalizationPhase.READY)
    var isConfirmingLocation: Boolean by mutableStateOf(false)
    var localizedPhotoId: Int? by mutableStateOf(null)
    var localizationScore: Double? by mutableStateOf(null)
    var localizationCandidates: List<Map<String, Any>> by mutableStateOf(emptyList())
    var goalPhotoIds: List<Int> by mutableStateOf(emptyList())
    var routeGuidance: String? by mutableStateOf(null)
    var routeWaypoints: List<Map<String, Any>> by mutableStateOf(emptyList())

    // Multi-goal progress
    var subGoals: List<NavSubGoalInfo> by mutableStateOf(emptyList())
    var currentGoalIdx: Int by mutableIntStateOf(0)
    var currentGoalName: String by mutableStateOf("")
    var totalGoals: Int by mutableIntStateOf(1)

    val imageCapture = ImageCapture.Builder().build()
    val captureExecutor = Executors.newSingleThreadExecutor()!!

    fun init(sessionId: String?, goal: String, initialGuidance: String, initialGoalPhotoIds: List<Int> = emptyList()) {
        this.sessionId = sessionId
        this.goal = goal
        this.guidance = initialGuidance
        this.goalPhotoIds = initialGoalPhotoIds
    }

    private fun handleTurnResponse(response: TurnResponse) {
        guidance = response.guidance
        currentAction = response.action
        annotatedPhotoUrl = response.annotated_photo_url
        localizedPhotoId = response.localized_photo_id
        localizationScore = response.localization_score
        localizationCandidates = response.localization_candidates
        goalPhotoIds = response.goal_photo_ids
        routeGuidance = response.route_guidance
        routeWaypoints = response.route_waypoints

        if (response.localization_candidates.isNotEmpty() &&
            localizationPhase != LocalizationPhase.NAVIGATING) {
            val bestCandidate = response.localization_candidates.maxByOrNull {
                (it["score"] as? Number)?.toDouble() ?: 0.0
            }
            val bestNid = (bestCandidate?.get("nid") as? Number)?.toInt()
            if (bestNid != null) {
                confirmLocation(bestNid)
            }
        }

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

    fun uploadFromGallery(context: Context, uri: Uri) {
        val sid = sessionId ?: return
        if (isUploading || hasArrived || pendingArrival || pendingQuestion != null) return
        isUploading = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    val photoFile = File(context.cacheDir, "gallery_${System.currentTimeMillis()}.jpg")
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(photoFile).use { output -> input.copyTo(output) }
                    } ?: throw Exception("無法讀取照片")
                    val requestBody = photoFile.asRequestBody("image/jpeg".toMediaType())
                    val part = MultipartBody.Part.createFormData("photo", photoFile.name, requestBody)
                    val resp = NavigationApi.service.uploadPhoto(sid, part)
                    photoFile.delete()
                    resp
                }
                handleTurnResponse(response)
                photoCount++
            } catch (e: Exception) {
                Log.e("NavGallery", "Gallery upload failed", e)
                errorMessage = "上傳失敗: ${e.message}"
            } finally {
                isUploading = false
            }
        }
    }

    fun submitAnswer() {
        val sid = sessionId ?: return
        if (answerText.isBlank() || isAnswering) return
        isAnswering = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    NavigationApi.service.postAnswer(sid, AnswerRequest(answer = answerText))
                }
                handleTurnResponse(response)
                answerText = ""
            } catch (e: Exception) {
                Log.e("NavAnswer", "Answer failed", e)
                errorMessage = "回答提交失敗: ${e.message}"
            } finally {
                isAnswering = false
            }
        }
    }

    fun captureAndUpload(context: Context) {
        val sid = sessionId ?: return
        if (isUploading || hasArrived || pendingArrival || pendingQuestion != null) return
        isUploading = true
        errorMessage = null
        localizationPhase = LocalizationPhase.PROCESSING

        val photoFile = File(context.cacheDir, "nav_photo_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                viewModelScope.launch {
                    try {
                        val response = withContext(Dispatchers.IO) {
                            val requestBody = photoFile.asRequestBody("image/jpeg".toMediaType())
                            val part = MultipartBody.Part.createFormData("photo", photoFile.name, requestBody)
                            NavigationApi.service.uploadPhoto(sid, part)
                        }
                        handleTurnResponse(response)
                        photoCount++
                    } catch (e: Exception) {
                        Log.e("NavUpload", "Upload failed", e)
                        errorMessage = "上傳失敗: ${e.message}"
                    } finally {
                        isUploading = false
                        photoFile.delete()
                    }
                }
            }

            override fun onError(e: ImageCaptureException) {
                Log.e("NavCapture", "Capture failed", e)
                isUploading = false
                errorMessage = "拍照失敗: ${e.message}"
            }
        })
    }

    fun confirmLocation(photoId: Int) {
        val sid = sessionId ?: return
        if (isConfirmingLocation) return
        isConfirmingLocation = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    NavigationApi.service.confirmLocation(sid, NavConfirmLocationRequest(photo_id = photoId))
                }
                localizedPhotoId = response.localized_photo_id
                goalPhotoIds = response.goal_photo_ids
                routeGuidance = response.route_guidance
                routeWaypoints = response.route_waypoints
                // 保留上傳定位那一輪的 guidance（第一次），避免自動 confirm_location
                // 再覆寫一次造成指引文字連跳兩次。路線指引仍存在 routeGuidance。
                if (response.sub_goals.isNotEmpty()) {
                    subGoals = response.sub_goals
                }
                currentGoalIdx = response.current_goal_idx
                currentGoalName = response.current_goal_name
                totalGoals = response.total_goals
                localizationPhase = LocalizationPhase.NAVIGATING
            } catch (e: Exception) {
                Log.e("NavConfirmLoc", "Confirm location failed", e)
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

    fun confirmArrival(kind: String) {
        val sid = sessionId ?: return
        if (isConfirming) return
        isConfirming = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    NavigationApi.service.confirmArrival(sid, ConfirmArrivalRequest(kind = kind))
                }
                pendingArrival = false
                handleTurnResponse(response)
                if (response.action == "ARRIVED" && response.sub_goals.all { it.arrived }) {
                    hasArrived = true
                }
            } catch (e: Exception) {
                Log.e("NavConfirm", "Confirm failed", e)
                errorMessage = "確認失敗: ${e.message}"
            } finally {
                isConfirming = false
            }
        }
    }
}
