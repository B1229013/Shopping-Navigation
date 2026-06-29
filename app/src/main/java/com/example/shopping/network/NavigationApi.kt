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

data class StartSessionRequest(val goal: String)

data class StartSessionResponse(
    val session_id: String,
    val guidance: String,
    val action: String,
    val goal_objects: List<String>
)

data class TurnResponse(
    val action: String,   // "ARRIVED", "MOVE", "ASK"
    val guidance: String,
    val question: String?,
    val node_id: Int,
    val annotated_photo_url: String?
)

data class AnswerRequest(val answer: String)

data class ConfirmArrivalRequest(val kind: String)  // "confirmed", "false_positive", "wrong_instance"

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
    private val baseUrl: String
        get() {
            val configured = BuildConfig.BACKEND_URL
            return if (configured.isNotBlank()) configured else "http://10.0.2.2:8000/"
        }

    val service: NavigationApiService by lazy {
        val logging = HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        }
        val client = OkHttpClient.Builder()
            .addInterceptor(logging)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(300, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()

        Retrofit.Builder()
            .baseUrl(baseUrl)
            .addConverterFactory(GsonConverterFactory.create())
            .client(client)
            .build()
            .create(NavigationApiService::class.java)
    }
}
