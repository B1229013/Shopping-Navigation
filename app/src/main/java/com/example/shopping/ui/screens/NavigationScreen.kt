package com.example.shopping.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.location.Location
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.NavController
import coil.compose.AsyncImage
import com.example.shopping.model.ShoppingItem
import com.example.shopping.network.AnswerRequest
import com.example.shopping.network.NavigationApi
import com.example.shopping.network.StartSessionRequest
import com.example.shopping.network.TurnResponse
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TeammateHomeScreen(navController: NavController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val shoppingListJson = navController.previousBackStackEntry?.savedStateHandle?.get<String>("shopping_list_json")
    val initialItems = remember(shoppingListJson) {
        try {
            if (shoppingListJson != null) {
                Json.decodeFromString<List<ShoppingItem>>(shoppingListJson)
            } else emptyList()
        } catch (e: Exception) { emptyList() }
    }

    val fusedLocationClient = remember { LocationServices.getFusedLocationProviderClient(context) }
    var nearbyLocation by remember { mutableStateOf<Location?>(null) }
    var showNearbySheet by remember { mutableStateOf(false) }
    var fetchingLocation by remember { mutableStateOf(false) }
    var isCreatingSession by remember { mutableStateOf(false) }
    var backendError by remember { mutableStateOf<String?>(null) }

    fun fetchLocationThenShow() {
        fetchingLocation = true
        try {
            fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                .addOnSuccessListener { loc ->
                    fetchingLocation = false
                    if (loc != null) {
                        nearbyLocation = loc
                        showNearbySheet = true
                    } else {
                        Toast.makeText(context, "無法取得位置，請確認 GPS 已開啟", Toast.LENGTH_SHORT).show()
                    }
                }
                .addOnFailureListener {
                    fetchingLocation = false
                    Toast.makeText(context, "位置取得失敗", Toast.LENGTH_SHORT).show()
                }
        } catch (e: SecurityException) {
            fetchingLocation = false
        }
    }

    val locationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            fetchLocationThenShow()
        } else {
            Toast.makeText(context, "需要定位權限才能搜尋附近商店", Toast.LENGTH_SHORT).show()
        }
    }

    fun openNearbyStores() {
        val hasPerm = ContextCompat.checkSelfPermission(
            context, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        if (hasPerm) fetchLocationThenShow()
        else locationPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
    }

    fun startNavigation() {
        if (initialItems.isEmpty()) return
        isCreatingSession = true
        backendError = null

        val goalText = initialItems.joinToString(", ") { "${it.name} x${it.qty}" }

        scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    NavigationApi.service.startSession(StartSessionRequest(goal = goalText))
                }
                navController.currentBackStackEntry?.savedStateHandle?.set("nav_session_id", response.session_id)
                navController.currentBackStackEntry?.savedStateHandle?.set("nav_goal", goalText)
                navController.currentBackStackEntry?.savedStateHandle?.set("nav_guidance", response.guidance)
                navController.navigate("ar_navigation")
            } catch (e: Exception) {
                Log.e("NavSession", "Failed to create session", e)
                backendError = "無法連線後端伺服器: ${e.message}"
            } finally {
                isCreatingSession = false
            }
        }
    }

    Scaffold(
        containerColor = Noir,
        topBar = {
            TopAppBar(
                title = { Text("確認購買目標", style = MaterialTheme.typography.titleLarge, color = TextPrimary) },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回", tint = TextSecondary)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = Noir)
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(horizontal = 20.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            StaggeredItem(index = 0) {
                Text(
                    "準備導航",
                    style = MaterialTheme.typography.displaySmall,
                    color = TextPrimary,
                    modifier = Modifier.padding(vertical = 16.dp)
                )
            }

            if (initialItems.isEmpty()) {
                Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.Center) {
                    Text(
                        "購物清單是空的，請先回主頁新增商品",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextTertiary
                    )
                }
            } else {
                StaggeredItem(index = 1) {
                    Text(
                        "以下是您清單中的待購物品：",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextSecondary,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 8.dp)
                    )
                }
                LazyColumn(
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(initialItems) { item ->
                        PressableSurface(onClick = {}, backgroundColor = SurfaceBase) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(14.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        item.name,
                                        style = MaterialTheme.typography.titleMedium,
                                        color = TextPrimary
                                    )
                                    if (!item.storeName.isNullOrBlank()) {
                                        Text(
                                            "預定地點: ${item.storeName}",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = Gold
                                        )
                                    }
                                }
                                Text(
                                    "x${item.qty}",
                                    style = MaterialTheme.typography.titleSmall,
                                    color = TextSecondary
                                )
                            }
                        }
                    }
                }
            }

            // Error message
            backendError?.let { err ->
                Spacer(Modifier.height(8.dp))
                Surface(
                    color = Color(0xFF3D1111),
                    shape = RoundedCornerShape(10.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        err,
                        color = Color(0xFFFF6B6B),
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(12.dp)
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            StaggeredItem(index = 2) {
                OutlinedButton(
                    onClick = { openNearbyStores() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Gold),
                    border = BorderStroke(1.dp, Gold),
                    enabled = !fetchingLocation
                ) {
                    if (fetchingLocation) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            color = Gold,
                            strokeWidth = 2.dp
                        )
                        Spacer(Modifier.width(10.dp))
                        Text("定位中...", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    } else {
                        Icon(Icons.Default.LocationOn, null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("附近商店 (1 公里內)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    }
                }
            }

            Spacer(Modifier.height(10.dp))

            StaggeredItem(index = 3) {
                Button(
                    onClick = { startNavigation() },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(54.dp),
                    enabled = initialItems.isNotEmpty() && !isCreatingSession,
                    colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    if (isCreatingSession) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            color = Noir,
                            strokeWidth = 2.dp
                        )
                        Spacer(Modifier.width(10.dp))
                        Text("建立導航中...", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    } else {
                        Text("開始導航", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                    }
                }
            }

            Spacer(Modifier.height(20.dp))
        }

        if (showNearbySheet) {
            nearbyLocation?.let { loc ->
                NearbyStoresSheet(location = loc, onDismiss = { showNearbySheet = false })
            }
        }
    }
}

// ── Navigation Screen with Backend Integration ──

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NavigationScreen(navController: NavController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current
    val fusedLocationClient = remember { LocationServices.getFusedLocationProviderClient(context) }

    val sessionId = navController.previousBackStackEntry?.savedStateHandle?.get<String>("nav_session_id")
    val goal = navController.previousBackStackEntry?.savedStateHandle?.get<String>("nav_goal") ?: ""
    val initialGuidance = navController.previousBackStackEntry?.savedStateHandle?.get<String>("nav_guidance") ?: ""

    var currentLocation by remember { mutableStateOf<Location?>(null) }
    var hasCameraPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    var hasLocationPermission by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED)
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        hasCameraPermission = permissions[Manifest.permission.CAMERA] ?: hasCameraPermission
        hasLocationPermission = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: hasLocationPermission
    }

    // Navigation state
    var guidance by remember { mutableStateOf(initialGuidance) }
    var currentAction by remember { mutableStateOf("TAKE_PHOTO") }
    var pendingQuestion by remember { mutableStateOf<String?>(null) }
    var answerText by remember { mutableStateOf("") }
    var isUploading by remember { mutableStateOf(false) }
    var isAnswering by remember { mutableStateOf(false) }
    var hasArrived by remember { mutableStateOf(false) }
    var annotatedPhotoUrl by remember { mutableStateOf<String?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var photoCount by remember { mutableIntStateOf(0) }

    // Camera capture
    val imageCapture = remember { ImageCapture.Builder().build() }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission || !hasLocationPermission) {
            permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION))
        }
    }

    LaunchedEffect(hasLocationPermission) {
        if (hasLocationPermission) {
            try {
                fusedLocationClient.getCurrentLocation(Priority.PRIORITY_HIGH_ACCURACY, null)
                    .addOnSuccessListener { location ->
                        currentLocation = location
                    }
            } catch (_: SecurityException) {}
        }
    }

    fun handleTurnResponse(response: TurnResponse) {
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

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri == null || sessionId == null || isUploading || hasArrived || pendingQuestion != null) return@rememberLauncherForActivityResult
        isUploading = true
        errorMessage = null
        scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    val photoFile = File(context.cacheDir, "gallery_${System.currentTimeMillis()}.jpg")
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        FileOutputStream(photoFile).use { output -> input.copyTo(output) }
                    } ?: throw Exception("無法讀取照片")
                    val requestBody = photoFile.asRequestBody("image/jpeg".toMediaType())
                    val part = MultipartBody.Part.createFormData("photo", photoFile.name, requestBody)
                    val resp = NavigationApi.service.uploadPhoto(sessionId, part)
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
        if (sessionId == null || answerText.isBlank() || isAnswering) return
        isAnswering = true
        errorMessage = null

        scope.launch {
            try {
                val response = withContext(Dispatchers.IO) {
                    NavigationApi.service.postAnswer(sessionId, AnswerRequest(answer = answerText))
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

    fun captureAndUpload() {
        if (sessionId == null || isUploading || hasArrived || pendingQuestion != null) return
        isUploading = true
        errorMessage = null

        val photoFile = File(context.cacheDir, "nav_photo_${System.currentTimeMillis()}.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, captureExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                scope.launch {
                    try {
                        val response = withContext(Dispatchers.IO) {
                            val requestBody = photoFile.asRequestBody("image/jpeg".toMediaType())
                            val part = MultipartBody.Part.createFormData("photo", photoFile.name, requestBody)
                            NavigationApi.service.uploadPhoto(sessionId, part)
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

    // No session — show error and go back
    if (sessionId == null) {
        Box(
            modifier = Modifier.fillMaxSize().background(Noir),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("未建立導航 Session", color = TextPrimary, style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(16.dp))
                Button(onClick = { navController.popBackStack() }) {
                    Text("返回")
                }
            }
        }
        return
    }

    Box(modifier = Modifier.fillMaxSize().background(Noir)) {
        // Camera preview
        if (hasCameraPermission) {
            CameraPreviewWithCapture(lifecycleOwner, imageCapture)
        }

        // AR store markers
        if (hasCameraPermission) {
            currentLocation?.let { loc ->
                NearbyStoresAr(location = loc)
            }
        }

        // ── Top: Goal & Status bar ──
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(top = 48.dp, start = 16.dp, end = 16.dp)
        ) {
            Surface(
                color = Noir.copy(alpha = 0.8f),
                shape = RoundedCornerShape(14.dp),
                border = BorderStroke(1.dp, Gold.copy(alpha = 0.3f))
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            if (hasArrived) Icons.Default.CheckCircle else Icons.Default.Navigation,
                            null,
                            tint = if (hasArrived) Color(0xFF4CAF50) else Gold,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            if (hasArrived) "已到達目標！" else "導航中",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = if (hasArrived) Color(0xFF4CAF50) else Gold
                        )
                        Spacer(Modifier.weight(1f))
                        Text(
                            "📷 $photoCount",
                            style = MaterialTheme.typography.labelMedium,
                            color = TextSecondary
                        )
                    }

                    Spacer(Modifier.height(6.dp))

                    Text(
                        goal,
                        style = MaterialTheme.typography.bodySmall,
                        color = TextSecondary,
                        maxLines = 1
                    )
                }
            }
        }

        // ── Center: Guidance panel ──
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(bottom = 140.dp, start = 16.dp, end = 16.dp)
        ) {
            // Guidance message
            Surface(
                color = Noir.copy(alpha = 0.85f),
                shape = RoundedCornerShape(16.dp),
                border = BorderStroke(
                    1.dp,
                    when (currentAction) {
                        "ARRIVED" -> Color(0xFF4CAF50).copy(alpha = 0.5f)
                        "ASK" -> Color(0xFFFF9800).copy(alpha = 0.5f)
                        else -> Gold.copy(alpha = 0.3f)
                    }
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Surface(
                            color = when (currentAction) {
                                "ARRIVED" -> Color(0xFF4CAF50)
                                "ASK" -> Color(0xFFFF9800)
                                "MOVE" -> Gold
                                else -> TextSecondary
                            },
                            shape = RoundedCornerShape(6.dp)
                        ) {
                            Text(
                                when (currentAction) {
                                    "ARRIVED" -> "到達"
                                    "ASK" -> "詢問"
                                    "MOVE" -> "移動"
                                    else -> "等待"
                                },
                                color = Noir,
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                            )
                        }
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "AI 導航指引",
                            style = MaterialTheme.typography.labelMedium,
                            color = TextSecondary
                        )
                    }

                    Spacer(Modifier.height(10.dp))

                    Text(
                        guidance,
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextPrimary,
                        lineHeight = 22.sp
                    )

                    // ASK flow — question + answer input
                    if (pendingQuestion != null && !hasArrived) {
                        Spacer(Modifier.height(12.dp))
                        Surface(
                            color = Color(0xFFFF9800).copy(alpha = 0.1f),
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Column(modifier = Modifier.padding(12.dp)) {
                                Text(
                                    pendingQuestion!!,
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = Color(0xFFFFCC80),
                                    fontWeight = FontWeight.Medium
                                )
                                Spacer(Modifier.height(8.dp))
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    OutlinedTextField(
                                        value = answerText,
                                        onValueChange = { answerText = it },
                                        placeholder = { Text("輸入回答...", color = TextTertiary) },
                                        modifier = Modifier.weight(1f),
                                        colors = OutlinedTextFieldDefaults.colors(
                                            focusedTextColor = TextPrimary,
                                            unfocusedTextColor = TextPrimary,
                                            focusedBorderColor = Gold,
                                            unfocusedBorderColor = TextSecondary.copy(alpha = 0.3f),
                                            cursorColor = Gold
                                        ),
                                        singleLine = true,
                                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                                        keyboardActions = KeyboardActions(onSend = { submitAnswer() })
                                    )
                                    Spacer(Modifier.width(8.dp))
                                    Button(
                                        onClick = { submitAnswer() },
                                        enabled = answerText.isNotBlank() && !isAnswering,
                                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF9800), contentColor = Noir),
                                        shape = CircleShape,
                                        contentPadding = PaddingValues(12.dp),
                                        modifier = Modifier.size(48.dp)
                                    ) {
                                        if (isAnswering) {
                                            CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Noir, strokeWidth = 2.dp)
                                        } else {
                                            Icon(Icons.Default.Send, null, modifier = Modifier.size(18.dp))
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Error
                    errorMessage?.let { err ->
                        Spacer(Modifier.height(8.dp))
                        Text(err, color = Color(0xFFFF6B6B), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }

        // ── Bottom: Action buttons ──
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(horizontal = 24.dp, vertical = 24.dp)
                .navigationBarsPadding(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Back button
            Button(
                onClick = { navController.popBackStack() },
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (hasArrived) Color(0xFF4CAF50).copy(alpha = 0.85f) else Color(0xFFCF6679).copy(alpha = 0.85f),
                    contentColor = TextPrimary
                ),
                shape = RoundedCornerShape(14.dp)
            ) {
                Text(
                    if (hasArrived) "完成" else "結束導航",
                    style = MaterialTheme.typography.labelLarge
                )
            }

            // Capture & Gallery buttons — only if not arrived and no pending question
            if (!hasArrived && pendingQuestion == null) {
                // Gallery pick
                Button(
                    onClick = { galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    enabled = !isUploading,
                    colors = ButtonDefaults.buttonColors(containerColor = SurfaceBase, contentColor = Gold),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Icon(Icons.Default.PhotoLibrary, null, modifier = Modifier.size(18.dp))
                    Spacer(Modifier.width(4.dp))
                    Text("相簿", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                }

                // Camera capture
                Button(
                    onClick = { captureAndUpload() },
                    enabled = !isUploading,
                    colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                    shape = RoundedCornerShape(14.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    if (isUploading) {
                        CircularProgressIndicator(modifier = Modifier.size(18.dp), color = Noir, strokeWidth = 2.dp)
                        Spacer(Modifier.width(8.dp))
                        Text("分析中...", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                    } else {
                        Icon(Icons.Default.CameraAlt, null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("拍照分析", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

// ── Camera preview with ImageCapture support ──

@Composable
fun CameraPreviewWithCapture(lifecycleOwner: LifecycleOwner, imageCapture: ImageCapture) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    AndroidView(factory = { ctx ->
        val previewView = PreviewView(ctx)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
            } catch (e: Exception) {
                Log.e("CameraX", "綁定失敗", e)
            }
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    }, modifier = Modifier.fillMaxSize())
}
