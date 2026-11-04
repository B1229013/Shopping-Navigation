package com.example.shopping.viewmodel

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.hardware.SensorManager
import android.media.ExifInterface
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.shopping.navigation.FollowState
import com.example.shopping.navigation.MapPositionEstimator
import com.example.shopping.navigation.MapPositionEstimator.FixResult
import com.example.shopping.navigation.PathFollower
import com.example.shopping.navigation.Vec2
import com.example.shopping.network.AnswerRequest
import com.example.shopping.network.ConfirmArrivalRequest
import com.example.shopping.network.NavigationApi
import com.example.shopping.network.PathResponse
import com.example.shopping.network.StartPoint
import com.example.shopping.network.StartPositionRequest
import com.example.shopping.network.TurnResponse
import com.example.shopping.sensor.PdrTracker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import retrofit2.HttpException
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
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
    /** [pendingQuestion] asks whether the user has arrived: answered with 有 / 沒有 buttons. */
    var pendingIsConfirm: Boolean by mutableStateOf(false)
    var answerText: String by mutableStateOf("")
    var isUploading: Boolean by mutableStateOf(false)
    var isAnswering: Boolean by mutableStateOf(false)
    var hasArrived: Boolean by mutableStateOf(false)
    var pendingArrival: Boolean by mutableStateOf(false)
    /** True when [pendingArrival] was triggered by PDR proximity, not a server photo response. */
    var arrivalFromPdr: Boolean by mutableStateOf(false)
    var isConfirming: Boolean by mutableStateOf(false)
    var annotatedPhotoUrl: String? by mutableStateOf(null)
    var errorMessage: String? by mutableStateOf(null)
    var photoCount: Int by mutableIntStateOf(0)
    /** Last photo taken or picked, shown beside the status card (iOS lastCapturedImage). */
    var lastPhoto: ImageBitmap? by mutableStateOf(null); private set

    // ── Localization (shared backend: Neo4j reference map, node ids = Neo4j nid) ──
    var correctedNodeId: Int? by mutableStateOf(null)
    var correctedConfidence: Double? by mutableStateOf(null)
    var currentLocation: String? by mutableStateOf(null)
    /** User heading in the map frame (0° = +y), from the matched directional photo. */
    var headingDeg: Double? by mutableStateOf(null)
    var headingConfidence: Double? by mutableStateOf(null)
    /** Photo unusable (blurred / dark / blocked) or not placed on the map — ask for a retake. */
    var retakeHint: String? by mutableStateOf(null)
    /** Several places look alike — turn ~90° on the spot and take one more photo. */
    var secondPhotoHint: String? by mutableStateOf(null)

    // ── Magnetometer calibration ──
    /** True when sensors are running but north calibration has not completed yet. */
    var needsMagCalibration: Boolean by mutableStateOf(false); private set

    // ── PDR position on the map (MapPositionEstimator) ──
    /** True while the step / heading sensors are running for this screen. */
    var sensorsRunning: Boolean by mutableStateOf(false); private set
    /** Map position: null until the first confident photo fix or "I'm at the entrance". */
    var mapPosition: Vec2? by mutableStateOf(null); private set
    /** Map metres walked since the last accepted fix (drives the photo reminder and the server's σ). */
    var distanceSinceFix: Double by mutableDoubleStateOf(0.0); private set
    /** Reference node of the last accepted fix (the server only considers nodes walkable from it). */
    var lastFixNodeId: Int? by mutableStateOf(null); private set
    /** Heading in the map frame: raw compass + offset learned from photo fixes. */
    var mapHeadingDeg: Double by mutableDoubleStateOf(0.0); private set
    /** Live route following (prompt, remaining metres, position on the route). */
    var followState: FollowState? by mutableStateOf(null); private set

    val isOnMap: Boolean get() = mapPosition != null

    /** The store's start points (map builder): 「我在起點」 offers them; empty → wait for a photo fix. */
    var startPoints: List<StartPoint> by mutableStateOf(emptyList()); private set

    // ── Route (GET /session/{id}/path, planned on the Neo4j map) ──
    var route: PathResponse? by mutableStateOf(null)
    /** First step of the route, e.g. "直走 12 公尺後右轉". */
    var routeGuidance: String? by mutableStateOf(null)
    /** Why there is no route (no route map / goal not on it / request failed). */
    var routeStatusMessage: String? by mutableStateOf(null)

    /** Server heading when it is trustworthy enough to draw (mini-map arrow before PDR has a fix). */
    val confidentHeadingDeg: Double?
        get() = headingDeg?.takeIf { (headingConfidence ?: 0.0) >= MIN_HEADING_CONFIDENCE }

    /** Heading to draw on the mini-map: live map heading once on the map, else the server's. */
    val displayHeadingDeg: Double?
        get() = if (isOnMap) mapHeadingDeg else confidentHeadingDeg

    // AR guidance arrow — egocentric direction along the route.
    // directionDeg: 0 = straight ahead, clockwise positive. null → hide arrow.
    var directionLabel: String? by mutableStateOf(null)
    var directionDeg: Float? by mutableStateOf(null)

    // ── Shopping progress (shopping → checkout → exit → done) ──
    var phase: String by mutableStateOf("shopping")
    var goalItems: List<GoalItem> by mutableStateOf(emptyList())
    var currentTarget: String? by mutableStateOf(null)

    // The backend shrinks every photo to 1600 px on the long side (JPEG 85) before using it,
    // so capture about that size: a full-resolution JPEG is several MB and timed out on store Wi-Fi.
    val imageCapture = ImageCapture.Builder()
        .setResolutionSelector(
            ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1600, 1200), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
                .build()
        )
        .setJpegQuality(85)
        .build()
    val captureExecutor = Executors.newSingleThreadExecutor()!!

    private val estimator = MapPositionEstimator()
    private val follower = PathFollower()
    private var tracker: PdrTracker? = null
    private var headingJob: Job? = null
    private var rawPdr = Vec2(0.0, 0.0)
    private var rawHeadingDeg = 0.0
    /** The follower is fed map-frame positions (identity anchor) once on the map. */
    private var followInMapFrame = false
    private var lastReplanAtMs = 0L
    private var isLoadingRoute = false
    /** Raw PDR where the server asked for a second photo; the request lapses after walking on. */
    private var secondPhotoAtRaw: Vec2? = null
    /** PDR already triggered pendingArrival for the current route; reset on new route. */
    private var pdrArrivalTriggered = false
    /** Node coordinates (map frame) from the last route payload. */
    private var nodeCoords: Map<Int, Vec2> = emptyMap()

    fun init(sessionId: String?, goal: String, initialGuidance: String, place: String? = null,
             mapNorthOffsetDeg: Double? = null) {
        val isNewSession = sessionId != null && sessionId != this.sessionId
        this.sessionId = sessionId
        this.goal = goal
        this.guidance = initialGuidance
        // As on iOS: show the route from the entrance before the first photo.
        if (isNewSession) {
            mapNorthOffsetDeg?.let { estimator.setMapNorth(it, tracker?.isNorthCalibrated == true) }
            loadRoute()
            if (place != null) loadStarts(place)
        }
    }

    // ── Sensors ──

    /** Start step / heading tracking for this navigation screen (idempotent). */
    fun startSensors(context: Context) {
        if (tracker != null) return
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager ?: return
        tracker = PdrTracker(sm).also { t ->
            t.onStep = { x, y, _ -> onStep(Vec2(x.toDouble(), y.toDouble())) }
            // Finding north rotates the raw heading (and later PDR steps) at once: keep the
            // map heading where it was (e.g. the "facing into the store" alignment), or, with
            // no alignment yet, take the venue's map north now that raw points to north.
            t.onNorthCalibrated = {
                t.northCorrectionDeg?.let { jump ->
                    estimator.onNorthCalibrated(jump)
                    rawHeadingDeg = normalize360(t.getCalibratedYawDeg().toDouble())
                    needsMagCalibration = false
                    publishEstimate()
                }
            }
            t.start()
        }
        sensorsRunning = true
        headingJob = viewModelScope.launch {
            while (isActive) {
                tracker?.let {
                    rawHeadingDeg = normalize360(it.getCalibratedYawDeg().toDouble())
                    needsMagCalibration = !it.isNorthCalibrated
                }
                mapHeadingDeg = estimator.mapHeading(rawHeadingDeg)
                if (isOnMap) updateLiveArrow()
                delay(HEADING_POLL_MS)
            }
        }
    }

    fun stopSensors() {
        headingJob?.cancel()
        headingJob = null
        tracker?.onStep = null
        tracker?.onNorthCalibrated = null
        tracker?.stop()
        tracker = null
        sensorsRunning = false
    }

    override fun onCleared() {
        stopSensors()
        super.onCleared()
    }

    private fun onStep(raw: Vec2) {
        rawPdr = raw
        estimator.ingest(raw)
        publishEstimate()
        secondPhotoAtRaw?.let { at ->
            if ((raw - at).length > SECOND_PHOTO_MAX_MOVE_M) {
                secondPhotoHint = null
                secondPhotoAtRaw = null
            }
        }
        val m = mapPosition
        if (m != null) followMapPosition(m) else followRaw(raw)
        followState?.let { st ->
            if (st.arrived && !pdrArrivalTriggered && !pendingArrival && !hasArrived && phase != "done") {
                pdrArrivalTriggered = true
                pendingArrival = true
                arrivalFromPdr = true
                guidance = st.prompt
                clearArrow()
            }
            val now = SystemClock.elapsedRealtime()
            if (st.needsReplan && now - lastReplanAtMs > REPLAN_MIN_INTERVAL_MS) {
                lastReplanAtMs = now
                loadRoute(fromPhone = true)
            }
        }
    }

    private fun publishEstimate() {
        mapPosition = estimator.mapPosition
        distanceSinceFix = estimator.distanceSinceFixM
        mapHeadingDeg = estimator.mapHeading(rawHeadingDeg)
    }

    /**
     * "I'm at this start, facing the view in its photo": put the user on the map there and
     * take the way they face as the start's map heading. The compass's own north was off
     * by 90° for a whole walk in the 2026-10-17 store test.
     */
    fun startAt(start: StartPoint) {
        val t = tracker ?: return
        val at = Vec2(start.x, start.y)
        val r = estimator.applyFix(at, 1.0, rawPdr)
        rawHeadingDeg = normalize360(t.getCalibratedYawDeg().toDouble())
        estimator.alignHeading(rawHeadingDeg, start.heading_deg)
        Log.i("NavStart", "${start.name}: fix=$r raw heading=${"%.1f".format(rawHeadingDeg)}° → map ${start.heading_deg}°")
        // the reference node nearest the start (the server's reachability starts there)
        lastFixNodeId = nodeCoords.filterKeys { it >= 0 }.minByOrNull { (it.value - at).length }?.key
        publishEstimate()
        mapPosition?.let { followMapPosition(it) }
        currentLocation = start.name
        // The server plans the visit order from here (not from its default start), then
        // the route is fetched again from this position.
        val sid = sessionId ?: return
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { NavigationApi.service.setStart(sid, StartPositionRequest(start.x, start.y)) }
            } catch (e: Exception) {
                Log.w("NavStart", "Could not tell the server the start point", e)
            }
            loadRoute(fromPhone = true)
        }
    }

    private fun loadStarts(place: String) {
        viewModelScope.launch {
            startPoints = try {
                withContext(Dispatchers.IO) { NavigationApi.service.getStarts(place).starts }
            } catch (e: Exception) {
                Log.w("NavStart", "No start points for $place", e)
                emptyList()
            }
        }
    }

    // ── Route following (port of the iOS RouteFollowerModel) ──

    private fun followRaw(raw: Vec2) {
        if (!follower.hasRoute || followInMapFrame) return
        followState = follower.update(raw, rawHeadingDeg)
    }

    private fun followMapPosition(m: Vec2) {
        val wasInMapFrame = followInMapFrame
        followInMapFrame = true          // a route loaded later anchors in the map frame too
        if (!follower.hasRoute) return
        if (!wasInMapFrame) follower.anchor(m, m)
        followState = follower.update(m, mapHeadingDeg)
        updateLiveArrow()
    }

    private fun installRoute(path: PathResponse, keepPhoneAnchor: Boolean) {
        pdrArrivalTriggered = false
        val polyline = path.polyline.map { Vec2(it[0], it[1]) }
        val turns = path.turns.map { Vec2(it.at[0], it.at[1]) to it.direction }
        val m = mapPosition
        if (followInMapFrame && m != null) {
            follower.setRoute(polyline, turns, m, keepAnchor = true)
            follower.anchor(m, m)
            followState = follower.update(m, mapHeadingDeg)
        } else if (sensorsRunning) {
            follower.setRoute(polyline, turns, rawPdr, keepAnchor = keepPhoneAnchor)
            followState = follower.update(rawPdr, rawHeadingDeg)
        } else {
            follower.setRoute(polyline, turns, Vec2(0.0, 0.0))
            followState = null
        }
    }

    private fun handleTurnResponse(response: TurnResponse, rawAtCapture: Vec2? = null) {
        guidance = response.guidance
        currentAction = response.action
        annotatedPhotoUrl = response.annotated_photo_url
        // next_instruction usually repeats the guidance; when the photo was not placed on
        // the map it says so instead ("這張照片對不到地圖上的位置…") — show that as a retake hint.
        retakeHint = response.retake_hint
            ?: response.next_instruction?.takeIf { response.corrected_node_id == null && it != response.guidance }
        secondPhotoHint = response.second_photo_hint
        secondPhotoAtRaw = if (response.second_photo_hint != null) rawAtCapture else null

        val phaseChanged = response.phase != null && response.phase != phase
        response.phase?.let { phase = it }

        // A localized photo corrects the map position (and, between two good fixes, the
        // heading offset and step length). When the server asks for a second photo, wait.
        var fixNote = if (response.second_photo_hint != null) "，等第二張照片再校正" else ""
        var fixAccepted = false
        val nid = response.corrected_node_id
        if (response.retake_hint == null && response.second_photo_hint == null &&
            nid != null && rawAtCapture != null && tracker != null) {
            val ex = response.estimated_x
            val ey = response.estimated_y
            val node = if (ex != null && ey != null) Vec2(ex, ey) else nodeCoords[nid]
            if (node != null) {
                when (val r = estimator.applyFix(node, response.corrected_confidence ?: 0.0, rawAtCapture)) {
                    FixResult.Accepted -> {
                        fixAccepted = true
                        lastFixNodeId = nid
                        // First fix with no heading alignment yet: use the DINO-matched
                        // directional photo's heading as the initial offset so the first
                        // PDR steps after this fix go in roughly the right direction.
                        if (!estimator.headingAligned) {
                            val dh = response.heading_deg
                            val dc = response.heading_confidence ?: 0.0
                            if (dh != null && dc >= MIN_HEADING_CONFIDENCE) {
                                val raw = tracker?.let { normalize360(it.getCalibratedYawDeg().toDouble()) } ?: rawHeadingDeg
                                estimator.alignHeading(raw, dh)
                                Log.i("NavHeading", "DINO heading init: raw=%.1f° dino=%.1f° conf=%.2f".format(raw, dh, dc))
                            }
                        }
                        estimator.ingest(rawPdr)          // replay the steps walked since the shutter
                        publishEstimate()
                        mapPosition?.let { followMapPosition(it) }
                    }
                    FixResult.RejectedLowConfidence -> fixNote = "，信心不足未校正位置"
                    is FixResult.RejectedJump -> fixNote = "，跳太遠（${r.distanceM.toInt()} m）未校正位置"
                }
            }
        }

        if (nid != null) {
            correctedNodeId = nid
            correctedConfidence = response.corrected_confidence
            headingDeg = response.heading_deg
            headingConfidence = response.heading_confidence
            currentLocation = response.corrected_location?.let { loc ->
                response.corrected_confidence?.let { "$loc（比對信心 ${(it * 100).toInt()}%$fixNote）" } ?: loc
            }
        }
        if (response.action != "ARRIVED") {
            // A fix the phone applied re-anchors the route at the server's node. A photo the
            // phone did not apply re-plans from the phone's own position once it is on the map;
            // before that, the phone's position is only "route start + steps", so plan from the
            // photo's node like the server's guidance does.
            if (nid != null) loadRoute(fromPhone = !fixAccepted && isOnMap)
            else if (phaseChanged) loadRoute(fromPhone = true)
        }

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
                    arrivalFromPdr = false
                }
                pendingQuestion = null
                clearArrow()
            }
            "ASK" -> {
                pendingQuestion = response.question
                pendingIsConfirm = response.confirm_question
            }
            "MOVE" -> {
                pendingQuestion = null
                pendingIsConfirm = false
                pendingArrival = false
            }
        }
        if (response.action != "ASK") pendingIsConfirm = false
    }

    /**
     * Fetch the route. [fromPhone] plans from the phone's own position (keeping the current
     * anchor); otherwise the server starts from the last photo fix, or the entrance.
     */
    private fun loadRoute(fromPhone: Boolean = false) {
        val sid = sessionId ?: return
        if (isLoadingRoute) return
        isLoadingRoute = true
        val st = followState
        val usePhone = fromPhone && st != null
        viewModelScope.launch {
            try {
                val path = withContext(Dispatchers.IO) {
                    if (usePhone && st != null) {
                        NavigationApi.service.getPath(sid, x = st.position.x, y = st.position.y, heading = st.headingDegrees)
                    } else {
                        NavigationApi.service.getPath(sid, heading = if (isOnMap) mapHeadingDeg else headingDeg)
                    }
                }
                route = path
                nodeCoords = path.nodes.associate { it.id to Vec2(it.x, it.y) }
                routeGuidance = path.turns.firstOrNull()?.text_zh
                routeStatusMessage = null
                installRoute(path, keepPhoneAnchor = usePhone)
                if (isOnMap) updateLiveArrow() else updateArrow(path)
            } catch (e: HttpException) {
                // 404: no route map for this place, or the goal is not on it — keep
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
                followState = null
                clearArrow()
            } catch (e: Exception) {
                Log.e("NavRoute", "Route request failed", e)
                routeStatusMessage = "路徑載入失敗：${e.message}"
            } finally {
                isLoadingRoute = false
            }
        }
    }

    /**
     * Before PDR is on the map: arrow toward the first route segment, relative to the
     * server's heading — only when that heading is trustworthy (a guess from look-alike
     * reference photos must not produce a confident "turn left").
     */
    private fun updateArrow(path: PathResponse) {
        val h = headingDeg
        val pts = path.polyline
        if (h == null || (headingConfidence ?: 0.0) < MIN_HEADING_CONFIDENCE || pts.size < 2) {
            clearArrow()
            return
        }
        val bearing = Math.toDegrees(atan2(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]))   // 0° = +y
        setArrow(normalizeSigned(bearing - h))
    }

    /** On the map: arrow toward a point a few metres further along the route, from the live heading. */
    private fun updateLiveArrow() {
        if (hasArrived || pendingArrival) return
        val st = followState ?: return
        val target = st.lookAhead ?: return
        val v = target - st.position
        if (v.length < 0.5) return
        val bearing = Math.toDegrees(atan2(v.x, v.y))
        setArrow(normalizeSigned(bearing - mapHeadingDeg))
    }

    private fun setArrow(rel: Double) {
        directionDeg = rel.toFloat()
        directionLabel = directionLabelFor(rel)
    }

    private fun clearArrow() {
        directionDeg = null
        directionLabel = null
    }

    private fun photoPart(file: File): MultipartBody.Part =
        MultipartBody.Part.createFormData("photo", file.name, file.asRequestBody("image/jpeg".toMediaType()))

    /**
     * Small upright copy of a photo for the thumbnail: the file itself is deleted after upload,
     * and BitmapFactory ignores the EXIF orientation CameraX / gallery JPEGs carry.
     */
    private fun decodeThumbnail(file: File, maxPx: Int = THUMBNAIL_MAX_PX): ImageBitmap? = try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxPx) sample *= 2
        BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })?.let { bmp ->
            val degrees = when (ExifInterface(file.path).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            val upright = if (degrees == 0f) bmp else
                Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(degrees) }, true)
            upright.asImageBitmap()
        }
    } catch (e: Exception) {
        Log.w("NavThumb", "Thumbnail decode failed", e)
        null
    }

    private fun text(v: String) = v.toRequestBody("text/plain".toMediaType())

    private fun num(v: Double): RequestBody = text(String.format(Locale.US, "%.3f", v))

    fun uploadFromGallery(context: Context, uri: Uri) {
        val sid = sessionId ?: return
        if (isUploading || hasArrived || pendingArrival || pendingQuestion != null) return
        isUploading = true
        errorMessage = null

        viewModelScope.launch {
            try {
                val photoFile = File(context.cacheDir, "gallery_${System.currentTimeMillis()}.jpg")
                withContext(Dispatchers.IO) {
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(photoFile).use { output -> input.copyTo(output) }
                    } ?: throw Exception("無法讀取照片")
                    decodeThumbnail(photoFile)
                }?.let { lastPhoto = it }
                val response = withContext(Dispatchers.IO) {
                    // Not taken here and now: the server ignores sensor data for it
                    val resp = NavigationApi.service.uploadPhoto(sid, photoPart(photoFile), source = text("gallery"))
                    photoFile.delete()
                    resp
                }
                handleTurnResponse(response)    // no rawAtCapture → never moves the map position
                photoCount++
            } catch (e: Exception) {
                Log.e("NavGallery", "Gallery upload failed", e)
                errorMessage = "上傳失敗: ${e.message}"
            } finally {
                isUploading = false
            }
        }
    }

    /** PDR arrival dismissed ("還沒到"): resume navigation without calling the server. */
    fun dismissPdrArrival() {
        pendingArrival = false
        arrivalFromPdr = false
    }

    /** Answer the "眼前有沒有看到要找的東西？" question with a button (有 / 沒有). */
    fun answerConfirm(yes: Boolean) {
        answerText = if (yes) "有" else "沒有"
        submitAnswer()
    }

    fun submitAnswer() {
        val sid = sessionId ?: return
        if (answerText.isBlank() || isAnswering) return
        isAnswering = true
        errorMessage = null
        val answer = answerText
        // 「有」 to the arrival question is the arrival confirmation itself (the server
        // finishes it like POST /confirm), so plan to the next item as confirmArrival does.
        val confirmsArrival = pendingIsConfirm && answer == "有"

        viewModelScope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    NavigationApi.service.postAnswer(sid, AnswerRequest(answer = answer))
                }
                handleTurnResponse(response)
                if (confirmsArrival && response.action == "MOVE") loadRoute(fromPhone = true)
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

        // Sensor state at the shutter, not when the upload finishes
        val running = tracker != null
        val raw = rawPdr
        val mapPos = mapPosition
        val dist = distanceSinceFix
        val fixNode = lastFixNodeId
        // Same raw heading as the live arrow and the PDR steps (calibrated GameRotVec yaw): the
        // offset is learned against it, and its yaw follows the phone's x axis, which stays
        // level when the phone is held upright for a photo. The accel+mag azimuth (logged for
        // comparison only) follows the phone's top, which points up then and can flip 180°.
        val rawNow = tracker?.let { normalize360(it.getCalibratedYawDeg().toDouble()) } ?: rawHeadingDeg
        val heading = estimator.mapHeading(rawNow)
        Log.i("HeadingDebug", "calYaw=%.1f° azimuth=%.1f° camera=%.1f° offset=%.1f° → heading=%.1f°".format(
            rawNow, tracker?.getAzimuthDeg() ?: -1f, tracker?.getCameraHeadingDeg() ?: -1f,
            estimator.headingOffsetDeg, heading))

        val photoFile = File(context.cacheDir, "nav_photo_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                viewModelScope.launch {
                    try {
                        withContext(Dispatchers.IO) { decodeThumbnail(photoFile) }?.let { lastPhoto = it }
                        val response = withContext(Dispatchers.IO) {
                            NavigationApi.service.uploadPhoto(
                                sid, photoPart(photoFile),
                                heading = if (running) num(heading) else null,
                                pdrX = if (running) num(raw.x) else null,
                                pdrY = if (running) num(raw.y) else null,
                                // map-frame position only once a fix has put the user on the map
                                mapX = mapPos?.let { num(it.x) },
                                mapY = mapPos?.let { num(it.y) },
                                distSinceFix = mapPos?.let { num(dist) },
                                fixNode = if (mapPos != null) fixNode?.let { text(it.toString()) } else null,
                            )
                        }
                        handleTurnResponse(response, rawAtCapture = if (running) raw else null)
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
                secondPhotoAtRaw = null
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
                // next item / checkout / exit: plan from where the phone is
                if (!hasArrived && (phaseChanged || kind == "confirmed")) loadRoute(fromPhone = true)
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
        /** Ask for a photo to re-fix the position after walking this far (iOS photoReminderM). */
        const val PHOTO_REMINDER_M = 12.0
        /** Long side of the last-photo thumbnail (shown at 56 dp, ~170 px on a 3x screen). */
        private const val THUMBNAIL_MAX_PX = 256
        /** Backend SECOND_PHOTO_MAX_MOVE_M: a second photo only counts within this distance. */
        const val SECOND_PHOTO_MAX_MOVE_M = 3.0
        private const val REPLAN_MIN_INTERVAL_MS = 8_000L
        private const val HEADING_POLL_MS = 200L

        fun normalizeSigned(deg: Double): Double {
            var x = ((deg % 360.0) + 360.0) % 360.0
            if (x > 180.0) x -= 360.0
            return x
        }

        private fun normalize360(deg: Double): Double = ((deg % 360.0) + 360.0) % 360.0

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
