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
import com.example.shopping.network.NavigationApi
import com.example.shopping.network.PathResponse
import com.example.shopping.network.TurnResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.HttpException
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import kotlin.math.atan2

/** One shopping-list item and whether the user has already picked it up. */
data class GoalItem(val name: String, val visited: Boolean)

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

    // ── Localization (shared backend: Neo4j reference map, node ids = Neo4j nid) ──
    var correctedNodeId: Int? by mutableStateOf(null)
    var correctedConfidence: Double? by mutableStateOf(null)
    var currentLocation: String? by mutableStateOf(null)
    /** User heading in the map frame (0° = +y), from the matched directional photo. */
    var headingDeg: Double? by mutableStateOf(null)
    var headingConfidence: Double? by mutableStateOf(null)
    /** Photo unusable (blurred / dark / blocked) — ask for a retake. */
    var retakeHint: String? by mutableStateOf(null)
    /** Several places look alike — turn ~90° on the spot and take one more photo. */
    var secondPhotoHint: String? by mutableStateOf(null)
    /** The map's own verdict for the last photo (distance / "at the target" / "not localized"). */
    var mapInstruction: String? by mutableStateOf(null)

    // ── Route on the editor map (GET /session/{id}/path) ──
    var route: PathResponse? by mutableStateOf(null)
    /** First step of the route, e.g. "直走 12 公尺後右轉". */
    var routeGuidance: String? by mutableStateOf(null)
    /** Why there is no route (no editor map / goal not on it / request failed). */
    var routeStatusMessage: String? by mutableStateOf(null)

    /** Server heading when it is trustworthy enough to draw (mini-map arrow). */
    val confidentHeadingDeg: Double?
        get() = headingDeg?.takeIf { (headingConfidence ?: 0.0) >= MIN_HEADING_CONFIDENCE }

    // AR guidance arrow — egocentric direction to the first route step.
    // directionDeg: 0 = straight ahead, clockwise positive. null → hide arrow.
    var directionLabel: String? by mutableStateOf(null)
    var directionDeg: Float? by mutableStateOf(null)

    // ── Shopping progress (shopping → checkout → exit → done) ──
    var phase: String by mutableStateOf("shopping")
    var goalItems: List<GoalItem> by mutableStateOf(emptyList())
    var currentTarget: String? by mutableStateOf(null)

    val imageCapture = ImageCapture.Builder().build()
    val captureExecutor = Executors.newSingleThreadExecutor()!!

    fun init(sessionId: String?, goal: String, initialGuidance: String) {
        val isNewSession = sessionId != null && sessionId != this.sessionId
        this.sessionId = sessionId
        this.goal = goal
        this.guidance = initialGuidance
        // As on iOS: show the route from the entrance before the first photo.
        if (isNewSession) loadRoute()
    }

    private fun handleTurnResponse(response: TurnResponse) {
        guidance = response.guidance
        currentAction = response.action
        annotatedPhotoUrl = response.annotated_photo_url
        retakeHint = response.retake_hint
        secondPhotoHint = response.second_photo_hint
        mapInstruction = response.next_instruction

        val phaseChanged = response.phase != null && response.phase != phase
        response.phase?.let { phase = it }

        if (response.corrected_node_id != null) {
            correctedNodeId = response.corrected_node_id
            correctedConfidence = response.corrected_confidence
            headingDeg = response.heading_deg
            headingConfidence = response.heading_confidence
            currentLocation = response.corrected_location?.let { loc ->
                response.corrected_confidence?.let { "$loc（比對信心 ${(it * 100).toInt()}%）" } ?: loc
            }
        }
        // A new fix re-anchors the route at the server's node; a phase change means the
        // path endpoint now plans a different leg.
        if (response.action != "ARRIVED" && (response.corrected_node_id != null || phaseChanged)) loadRoute()

        response.goal_item_names?.takeIf { it.isNotEmpty() }?.let { names ->
            val visited = response.visited_item_names.orEmpty().toSet()
            goalItems = names.map { GoalItem(it, it in visited) }
        }
        currentTarget = response.current_target_name

        when (response.action) {
            "ARRIVED" -> {
                if (phase == "done") {
                    hasArrived = true
                    pendingArrival = false
                } else {
                    pendingArrival = true
                }
                pendingQuestion = null
                clearArrow()
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

    /** Plan from the last photo-localized node (server side) toward the current goal. */
    private fun loadRoute() {
        val sid = sessionId ?: return
        viewModelScope.launch {
            try {
                val path = withContext(Dispatchers.IO) {
                    NavigationApi.service.getPath(sid, heading = headingDeg)
                }
                route = path
                routeGuidance = path.turns.firstOrNull()?.text_zh
                routeStatusMessage = null
                updateArrow(path)
            } catch (e: HttpException) {
                // 404: no editor map for this place, or the goal is not on it — keep
                // the VLM guidance only.
                val body = e.response()?.errorBody()?.string().orEmpty()
                Log.w("NavRoute", "No route: HTTP ${e.code()} $body")
                routeStatusMessage = when {
                    "no_editor_map" in body -> "這個賣場沒有航點地圖，無法顯示路徑"
                    "goal_not_on_map" in body -> "地圖上找不到「${currentTarget ?: goal}」的位置"
                    else -> "路徑載入失敗（HTTP ${e.code()}）"
                }
                route = null
                routeGuidance = null
                clearArrow()
            } catch (e: Exception) {
                Log.e("NavRoute", "Route request failed", e)
                routeStatusMessage = "路徑載入失敗：${e.message}"
            }
        }
    }

    /**
     * Arrow toward the first route segment, relative to where the user faces. Only
     * shown when the heading is trustworthy: guessing it from look-alike reference
     * photos must not produce a confident "turn left".
     */
    private fun updateArrow(path: PathResponse) {
        val h = headingDeg
        val pts = path.polyline
        if (h == null || (headingConfidence ?: 0.0) < MIN_HEADING_CONFIDENCE || pts.size < 2) {
            clearArrow()
            return
        }
        val (ax, ay) = pts[0][0] to pts[0][1]
        val (bx, by) = pts[1][0] to pts[1][1]
        val bearing = Math.toDegrees(atan2(bx - ax, by - ay))   // map frame: 0° = +y
        val rel = normalizeSigned(bearing - h)
        directionDeg = rel.toFloat()
        directionLabel = directionLabelFor(rel)
    }

    private fun clearArrow() {
        directionDeg = null
        directionLabel = null
    }

    private fun photoPart(file: File): MultipartBody.Part =
        MultipartBody.Part.createFormData("photo", file.name, file.asRequestBody("image/jpeg".toMediaType()))

    private fun text(v: String) = v.toRequestBody("text/plain".toMediaType())

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
                    // Not taken here and now: the server ignores sensor data for it
                    val resp = NavigationApi.service.uploadPhoto(sid, photoPart(photoFile), source = text("gallery"))
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

        val photoFile = File(context.cacheDir, "nav_photo_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                viewModelScope.launch {
                    try {
                        val response = withContext(Dispatchers.IO) {
                            NavigationApi.service.uploadPhoto(sid, photoPart(photoFile))
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
                secondPhotoHint = null
                // POST /confirm is the only endpoint that advances the phase; reaching
                // "done" also ends the session (a later /photo would be refused).
                val phaseChanged = response.phase != null && response.phase != phase
                response.phase?.let { phase = it }
                guidance = response.guidance
                response.goal_item_names?.takeIf { it.isNotEmpty() }?.let { names ->
                    val visited = response.visited_item_names.orEmpty().toSet()
                    goalItems = names.map { GoalItem(it, it in visited) }
                }
                currentTarget = response.current_target_name
                when (response.action) {
                    "ARRIVED" -> hasArrived = true
                    "MOVE" -> pendingQuestion = null
                }
                if (!hasArrived && (phaseChanged || kind == "confirmed")) loadRoute()
            } catch (e: Exception) {
                Log.e("NavConfirm", "Confirm failed", e)
                errorMessage = "確認失敗: ${e.message}"
            } finally {
                isConfirming = false
            }
        }
    }

    companion object {
        /** Below this the server's heading is a guess from look-alike photos. */
        const val MIN_HEADING_CONFIDENCE = 0.5

        fun normalizeSigned(deg: Double): Double {
            var x = ((deg % 360.0) + 360.0) % 360.0
            if (x > 180.0) x -= 360.0
            return x
        }

        /** 8-way label for an egocentric bearing (0 = ahead, clockwise positive). */
        fun directionLabelFor(rel: Double): String = when {
            rel > -22.5 && rel <= 22.5 -> "正前方"
            rel > 22.5 && rel <= 67.5 -> "右前方"
            rel > 67.5 && rel <= 112.5 -> "右方"
            rel > 112.5 && rel <= 157.5 -> "右後方"
            rel < -22.5 && rel >= -67.5 -> "左前方"
            rel < -67.5 && rel >= -112.5 -> "左方"
            rel < -112.5 && rel >= -157.5 -> "左後方"
            else -> "後方"
        }
    }
}
