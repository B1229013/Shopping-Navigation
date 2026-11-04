package com.example.shopping.network

import com.example.shopping.BuildConfig
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

// ── Request / Response models ──
// Mirrors the shared backend (APPNAV_ios backend/server/models.py) and the iOS client
// (APPNAV_ios/Services/NavigationAPI.swift), so both apps talk to the same server.

data class StartSessionRequest(
    val goal: String,
    val place: String? = null,
)

data class StartSessionResponse(
    val session_id: String,
    val guidance: String,
    val action: String,
    val goal_objects: List<String>,
    val place: String? = null,
    val goal_node_id: Int? = null,
    val map_north_offset_deg: Double? = null,
)

data class PlaceInfo(
    val name: String,
    val photo_count: Int = 0,
)

data class PlacesResponse(val places: List<PlaceInfo> = emptyList())

/**
 * Where a walk may begin on the store's map (set in the map builder): stand at (x, y),
 * face the view in [photo_url], and that facing is map heading [heading_deg] (0° = +y, clockwise).
 */
data class StartPoint(
    val id: String,
    val name: String,
    val x: Double,
    val y: Double,
    val heading_deg: Double,
    val photo_url: String? = null,
)

data class StartsResponse(val place: String? = null, val starts: List<StartPoint> = emptyList())

data class StartPositionRequest(val x: Double, val y: Double)

data class SetStartResponse(val start_node: Int? = null, val replanned: Boolean = false)

data class TurnResponse(
    val action: String,   // "ARRIVED", "MOVE", "ASK"
    val guidance: String,
    val question: String?,
    // The question asks "have you arrived?": answer it with 是 / 不是 buttons.
    val confirm_question: Boolean = false,
    val node_id: Int,
    val annotated_photo_url: String?,
    // Photo unusable (blurred / dark / blocked by people or carts): ask for a retake.
    // When set, the photo did not move the user's position.
    val retake_hint: String? = null,
    // Several places look alike: turn ~90° on the spot and take one more photo.
    val second_photo_hint: String? = null,
    // Visual localization against the Neo4j reference map
    val corrected_node_id: Int? = null,
    val corrected_confidence: Double? = null,
    val corrected_location: String? = null,
    // Map position between nodes (metres, origin = entrance, heading 0° = +y)
    val estimated_x: Double? = null,
    val estimated_y: Double? = null,
    // User heading in the map frame, from the matched directional photo
    val heading_deg: Double? = null,
    val heading_slot: String? = null,
    val heading_confidence: Double? = null,
    // Map-derived turn-by-turn line, shown verbatim
    val next_instruction: String? = null,
    val remaining_targets: Int? = null,
    val phase: String? = null,             // "shopping" | "checkout" | "exit" | "done"
    val goal_item_names: List<String>? = null,
    val visited_item_names: List<String>? = null,
    val current_target_name: String? = null,
    val goal_node_id: Int? = null,
)

/** Route on the hand-corrected editor map (GET /session/{id}/path). */
data class PathResponse(
    val place: String,
    val goal_item: String,
    val start: PathStart,
    val target: PathTarget,
    val path: List<Int>,
    val polyline: List<List<Double>>,      // [[x, y], ...] metres
    val turns: List<PathTurn>,
    val distance_m: Double,
    val nodes: List<PathNode> = emptyList(),
    val edges: List<PathEdge> = emptyList(),
)

data class PathStart(val wp: Int, val x: Double, val y: Double, val source: String)  // "phone" | "photo" | "entrance"

data class PathTarget(
    val wp: Int,
    val x: Double,
    val y: Double,
    val products: List<String> = emptyList(),
    val neo_nid: Int? = null,
)

data class PathTurn(
    val direction: String,                 // straight / right / left / slight_* / sharp_* / behind
    val text_zh: String,
    val at: List<Double>,
    val to: List<Double>,
    val distance_m: Double,
    val passed_nodes: Int,
)

data class PathNode(val id: Int, val x: Double, val y: Double)

data class PathEdge(val from: Int, val to: Int, val length: Double)

data class AnswerRequest(val answer: String)

data class ConfirmArrivalRequest(val kind: String)  // "confirmed", "confirmed_pdr", "false_positive", "wrong_instance"

data class MapNode(
    val id: Int,
    val photo: String,
    val detected: List<String>,
    val summary: String,
    val timestamp: String
)

data class MapEdge(
    val from: Int,
    val to: Int,
    val action: String
)

data class MapResponse(
    val nodes: List<MapNode>,
    val edges: List<MapEdge>,
    val current_node: Int?,
    val goal_node: Int?
)

data class HealthResponse(val status: String)

// ── Retrofit interface ──

interface NavigationApiService {

    @GET("places")
    suspend fun getPlaces(): PlacesResponse

    @POST("session")
    suspend fun startSession(@Body req: StartSessionRequest): StartSessionResponse

    @GET("starts")
    suspend fun getStarts(@Query("place") place: String): StartsResponse

    /** The user stands at a start point: the server re-plans the visit order from there. */
    @POST("session/{sessionId}/start")
    suspend fun setStart(@Path("sessionId") sessionId: String, @Body body: StartPositionRequest): SetStartResponse

    /**
     * Optional fields are omitted when null. `heading` / `map_*` / `dist_since_fix` /
     * `fix_node` are the phone's PDR signals (server/loc_fusion.py); `source = "gallery"`
     * tells the server the photo was not taken here and now.
     */
    @Multipart
    @POST("session/{sessionId}/photo")
    suspend fun uploadPhoto(
        @Path("sessionId") sessionId: String,
        @Part photo: MultipartBody.Part,
        @Part("heading") heading: RequestBody? = null,
        @Part("pdr_x") pdrX: RequestBody? = null,
        @Part("pdr_y") pdrY: RequestBody? = null,
        @Part("map_x") mapX: RequestBody? = null,
        @Part("map_y") mapY: RequestBody? = null,
        @Part("dist_since_fix") distSinceFix: RequestBody? = null,
        @Part("fix_node") fixNode: RequestBody? = null,
        @Part("source") source: RequestBody? = null,
    ): TurnResponse

    @POST("session/{sessionId}/answer")
    suspend fun postAnswer(
        @Path("sessionId") sessionId: String,
        @Body req: AnswerRequest
    ): TurnResponse

    @POST("session/{sessionId}/confirm")
    suspend fun confirmArrival(
        @Path("sessionId") sessionId: String,
        @Body req: ConfirmArrivalRequest
    ): TurnResponse

    /** Start = phone x/y when given, else the last photo-localized node, else the entrance. */
    @GET("session/{sessionId}/path")
    suspend fun getPath(
        @Path("sessionId") sessionId: String,
        @Query("x") x: Double? = null,
        @Query("y") y: Double? = null,
        @Query("heading") heading: Double? = null,
    ): PathResponse

    @GET("health")
    suspend fun health(): HealthResponse
}

// ── Singleton client ──

object NavigationApi {
    private val okClient: OkHttpClient by lazy {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
        OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(300, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()
    }

    private var _service: NavigationApiService? = null
    private var _lastUrl: String? = null

    val service: NavigationApiService
        get() {
            val url = BackendConfig.currentUrl
            if (_service == null || _lastUrl != url) {
                _service = Retrofit.Builder()
                    .baseUrl(url)
                    .addConverterFactory(GsonConverterFactory.create())
                    .client(okClient)
                    .build()
                    .create(NavigationApiService::class.java)
                _lastUrl = url
            }
            return _service!!
        }
}
