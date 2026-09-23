package com.example.shopping.network

import com.example.shopping.BuildConfig
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.*
import java.util.concurrent.TimeUnit

// ── Request / Response models ──

data class StartSessionRequest(
    val goal: String,
    val place_name: String? = null,
)

data class NavSubGoalInfo(
    val name: String,
    val arrived: Boolean = false,
)

data class StartSessionResponse(
    val session_id: String,
    val guidance: String,
    val action: String,
    val goal_objects: List<String>,
    val nav_mode: String = "explore",
    val goal_photo_ids: List<Int> = emptyList(),
    val place_name: String = "",
    val sub_goals: List<NavSubGoalInfo> = emptyList(),
    val tsp_order: List<Int> = emptyList(),
    val current_goal_idx: Int = 0,
    val current_goal_name: String = "",
    val total_goals: Int = 1,
)

data class TurnResponse(
    val action: String,   // "ARRIVED", "MOVE", "ASK"
    val guidance: String,
    val question: String?,
    val node_id: Int,
    val annotated_photo_url: String?,
    val localized_photo_id: Int? = null,
    val localization_score: Double? = null,
    val localization_candidates: List<Map<String, Any>> = emptyList(),
    val goal_photo_ids: List<Int> = emptyList(),
    val planned_route: List<Int> = emptyList(),
    val route_target_photo_id: Int? = null,
    val route_distance: Double? = null,
    val route_hops: Int? = null,
    val route_guidance: String? = null,
    val route_waypoints: List<Map<String, Any>> = emptyList(),
    val sub_goals: List<NavSubGoalInfo> = emptyList(),
    val current_goal_idx: Int = 0,
    val current_goal_name: String = "",
    val total_goals: Int = 1,
)

data class AnswerRequest(val answer: String)

data class ConfirmArrivalRequest(val kind: String)  // "confirmed", "false_positive", "wrong_instance"

data class NavConfirmLocationRequest(val photo_id: Int)

data class NavConfirmLocationResponse(
    val guidance: String,
    val localized_photo_id: Int,
    val goal_photo_ids: List<Int> = emptyList(),
    val route_guidance: String? = null,
    val route_waypoints: List<Map<String, Any>> = emptyList(),
    val route_distance: Double? = null,
    val route_hops: Int? = null,
    val sub_goals: List<NavSubGoalInfo> = emptyList(),
    val current_goal_idx: Int = 0,
    val current_goal_name: String = "",
    val total_goals: Int = 1,
)

data class SessionState(
    val id: String,
    val goal: String,
    val goal_objects: List<String>,
    val history: List<Map<String, Any>>,
    val pending_question: String?,
    val arrived: Boolean,
    val last_node_id: Int?,
    val goal_node: Int?,
    val created_at: String
)

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

    @POST("session")
    suspend fun startSession(@Body req: StartSessionRequest): StartSessionResponse

    @Multipart
    @POST("session/{sessionId}/photo")
    suspend fun uploadPhoto(
        @Path("sessionId") sessionId: String,
        @Part photo: MultipartBody.Part
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

    @POST("session/{sessionId}/confirm_location")
    suspend fun confirmLocation(
        @Path("sessionId") sessionId: String,
        @Body req: NavConfirmLocationRequest
    ): NavConfirmLocationResponse

    @GET("session/{sessionId}")
    suspend fun getSession(@Path("sessionId") sessionId: String): SessionState

    @GET("session/{sessionId}/map")
    suspend fun getMap(
        @Path("sessionId") sessionId: String,
        @Query("format") format: String = "json"
    ): MapResponse

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
