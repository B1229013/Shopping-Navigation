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
)

data class SNavStartResponse(
    val session_id: String,
    val guidance: String,
    val action: String,
    val goal_objects: List<String>,
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
)

data class SNavMapInfo(
    val id: String,
    val name: String,
    val node_count: Int,
    val total_steps: Int,
    val total_distance_m: Double,
    val created_at: String,
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

    @POST("snav/maps/save")
    suspend fun saveMap(@Body req: SaveMapRequest): SaveMapResponse

    @GET("snav/maps")
    suspend fun listMaps(): List<SNavMapInfo>

    @GET("snav/maps/{mapId}")
    suspend fun getMap(@Path("mapId") mapId: String): MapResponse
}

// ── Singleton client ──

object SensorNavApi {
    private val baseUrl: String
        get() {
            val configured = BuildConfig.BACKEND_URL
            return if (configured.isNotBlank()) configured else "http://10.0.2.2:8000/"
        }

    val service: SensorNavApiService by lazy {
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
            .create(SensorNavApiService::class.java)
    }
}
