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

data class SNavStartRequest(
    val goal: String,
    val map_id: String? = null,
    val place_name: String? = null,
)

data class SNavSubGoalInfo(
    val name: String,
    val arrived: Boolean = false,
)

data class SNavStartResponse(
    val session_id: String,
    val guidance: String,
    val action: String,
    val goal_objects: List<String>,
    val nav_mode: String = "explore",
    val goal_photo_ids: List<Int> = emptyList(),
    val place_name: String = "",
    val sub_goals: List<SNavSubGoalInfo> = emptyList(),
    val tsp_order: List<Int> = emptyList(),
    val current_goal_idx: Int = 0,
    val current_goal_name: String = "",
    val total_goals: Int = 1,
)

data class Neo4jPlaceInfo(
    val place_name: String,
    val node_count: Int = 0,
)

data class SNavTurnResponse(
    val action: String,
    val guidance: String,
    val question: String? = null,
    val node_id: Int,
    val annotated_photo_url: String? = null,
    val step_guidance: String? = null,
    val estimated_remaining_steps: Int? = null,
    val estimated_remaining_distance: Double? = null,
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
    val sub_goals: List<SNavSubGoalInfo> = emptyList(),
    val current_goal_idx: Int = 0,
    val current_goal_name: String = "",
    val total_goals: Int = 1,
)

data class SNavMapInfo(
    val id: String,
    val name: String,
    val node_count: Int,
    val total_steps: Int,
    val total_distance_m: Double,
    val created_at: String,
)

data class ConfirmLocationRequest(
    val photo_id: Int,
)

data class ConfirmLocationResponse(
    val guidance: String,
    val localized_photo_id: Int,
    val goal_photo_ids: List<Int> = emptyList(),
    val route_guidance: String? = null,
    val route_waypoints: List<Map<String, Any>> = emptyList(),
    val route_distance: Double? = null,
    val route_hops: Int? = null,
    val sub_goals: List<SNavSubGoalInfo> = emptyList(),
    val current_goal_idx: Int = 0,
    val current_goal_name: String = "",
    val total_goals: Int = 1,
)

data class SaveMapRequest(
    val session_id: String,
    val name: String,
)

data class SaveMapResponse(
    val map_id: String,
    val name: String,
)

// ── Retrofit interface ──

interface SensorNavApiService {

    @POST("snav/session")
    suspend fun startSession(@Body req: SNavStartRequest): SNavStartResponse

    @Multipart
    @POST("snav/{sessionId}/photo")
    suspend fun uploadPhoto(
        @Path("sessionId") sessionId: String,
        @Part photo: MultipartBody.Part,
        @Part("pdr_data") pdrData: RequestBody? = null,
    ): SNavTurnResponse

    @POST("snav/{sessionId}/answer")
    suspend fun postAnswer(
        @Path("sessionId") sessionId: String,
        @Body req: AnswerRequest,
    ): SNavTurnResponse

    @POST("snav/{sessionId}/confirm")
    suspend fun confirmArrival(
        @Path("sessionId") sessionId: String,
        @Body req: ConfirmArrivalRequest,
    ): SNavTurnResponse

    @POST("snav/{sessionId}/confirm_location")
    suspend fun confirmLocation(
        @Path("sessionId") sessionId: String,
        @Body req: ConfirmLocationRequest,
    ): ConfirmLocationResponse

    @POST("snav/maps/save")
    suspend fun saveMap(@Body req: SaveMapRequest): SaveMapResponse

    @GET("snav/maps")
    suspend fun listMaps(): List<SNavMapInfo>

    @GET("snav/maps/{mapId}")
    suspend fun getMap(@Path("mapId") mapId: String): MapResponse

    @GET("snav/neo4j/places")
    suspend fun listNeo4jPlaces(): List<Neo4jPlaceInfo>
}

// ── Singleton client ──

object SensorNavApi {
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

    private var _service: SensorNavApiService? = null
    private var _lastUrl: String? = null

    val service: SensorNavApiService
        get() {
            val url = BackendConfig.currentUrl
            if (_service == null || _lastUrl != url) {
                _service = Retrofit.Builder()
                    .baseUrl(url)
                    .addConverterFactory(GsonConverterFactory.create())
                    .client(okClient)
                    .build()
                    .create(SensorNavApiService::class.java)
                _lastUrl = url
            }
            return _service!!
        }
}
