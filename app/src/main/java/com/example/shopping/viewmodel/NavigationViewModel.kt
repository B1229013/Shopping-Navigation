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
    var annotatedPhotoUrl: String? by mutableStateOf(null)
    var errorMessage: String? by mutableStateOf(null)
    var photoCount: Int by mutableIntStateOf(0)

    val imageCapture = ImageCapture.Builder().build()
    val captureExecutor = Executors.newSingleThreadExecutor()!!

    fun init(sessionId: String?, goal: String, initialGuidance: String) {
        this.sessionId = sessionId
        this.goal = goal
        this.guidance = initialGuidance
    }

    private fun handleTurnResponse(response: TurnResponse) {
        guidance = response.guidance
        currentAction = response.action
        annotatedPhotoUrl = response.annotated_photo_url

        when (response.action) {
            "ARRIVED" -> {
                hasArrived = true
                pendingQuestion = null
            }
            "ASK" -> {
                pendingQuestion = response.question
            }
            "MOVE" -> {
                pendingQuestion = null
            }
        }
    }

    fun uploadFromGallery(context: Context, uri: Uri) {
        val sid = sessionId ?: return
        if (isUploading || hasArrived || pendingQuestion != null) return
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
        if (isUploading || hasArrived || pendingQuestion != null) return
        isUploading = true
        errorMessage = null

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
}
