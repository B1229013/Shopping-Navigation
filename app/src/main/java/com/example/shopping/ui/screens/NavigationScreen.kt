package com.example.shopping.ui.screens

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.NavController
import coil.compose.AsyncImage
import com.example.shopping.model.ShoppingItem
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.serialization.json.Json
import java.util.concurrent.Executors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TeammateHomeScreen(navController: NavController) {
    val context = LocalContext.current
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
                    onClick = { navController.navigate("ar_navigation") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(54.dp),
                    enabled = initialItems.isNotEmpty(),
                    colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                    shape = RoundedCornerShape(14.dp)
                ) {
                    Text(
                        "開始導航",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
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

@Composable
fun NavigationScreen(navController: NavController) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val fusedLocationClient = remember { LocationServices.getFusedLocationProviderClient(context) }
    
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

    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    val photoPickerLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { selectedImageUri = it }

    var lastCaptureTime by remember { mutableLongStateOf(0L) }
    val captureInterval = 3000L

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
                        if (location != null) {
                            Log.d("GPS", "Current location: ${location.latitude}, ${location.longitude}")
                        }
                    }
            } catch (e: SecurityException) {
                Log.e("GPS", "Location permission error", e)
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize().background(Noir)) {
        if (selectedImageUri != null) {
            AsyncImage(
                model = selectedImageUri,
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
        } else if (hasCameraPermission) {
            CameraWithAnalysis(lifecycleOwner, { image ->
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastCaptureTime >= captureInterval) {
                    lastCaptureTime = currentTime
                    processImageForModel(image)
                } else {
                    image.close()
                }
            })
        }

        // AR store markers — only while live camera is active
        if (hasCameraPermission && selectedImageUri == null) {
            currentLocation?.let { loc ->
                NearbyStoresAr(location = loc)
            }
        }

        AROverlayUI(navController, selectedImageUri, { selectedImageUri = null }, photoPickerLauncher, currentLocation)

        if (hasCameraPermission && selectedImageUri == null) {
            Text(
                "自動分析中 (每3秒)...",
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 80.dp)
                    .clip(RoundedCornerShape(20.dp))
                    .background(Noir.copy(alpha = 0.6f))
                    .padding(horizontal = 16.dp, vertical = 8.dp),
                style = MaterialTheme.typography.labelMedium,
                color = TextPrimary
            )
        }
    }
}

private fun processImageForModel(image: ImageProxy) {
    Log.d("NavigationAI", "正在分析影像: ${image.width}x${image.height}")
    image.close()
}

@Composable
fun CameraWithAnalysis(lifecycleOwner: LifecycleOwner, onImageAnalyzed: (ImageProxy) -> Unit) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val analysisExecutor = remember { Executors.newSingleThreadExecutor() }

    AndroidView(factory = { ctx ->
        val previewView = PreviewView(ctx)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { it.setAnalyzer(analysisExecutor) { image -> onImageAnalyzed(image) } }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
            } catch (e: Exception) { Log.e("CameraX", "綁定失敗", e) }
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    }, modifier = Modifier.fillMaxSize())
}

@Composable
fun AROverlayUI(
    navController: NavController,
    uri: Uri?,
    onClearUri: () -> Unit,
    launcher: androidx.activity.result.ActivityResultLauncher<PickVisualMediaRequest>,
    location: Location?
) {
    Box(modifier = Modifier.fillMaxSize()) {
        if (location != null) {
            Surface(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(16.dp)
                    .padding(top = 40.dp),
                color = Noir.copy(alpha = 0.7f),
                shape = RoundedCornerShape(12.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, Gold.copy(alpha = 0.5f))
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(Icons.Default.LocationOn, null, tint = Gold, modifier = Modifier.size(16.dp))
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "GPS 已就緒",
                        color = TextPrimary,
                        style = MaterialTheme.typography.labelSmall
                    )
                }
            }
        }

        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(32.dp)
                .navigationBarsPadding(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = { if (uri != null) onClearUri() else navController.popBackStack() },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Danger.copy(alpha = 0.85f),
                    contentColor = TextPrimary
                ),
                shape = RoundedCornerShape(14.dp)
            ) {
                Text(
                    if (uri != null) "恢復相機" else "結束導航",
                    style = MaterialTheme.typography.labelLarge
                )
            }

            if (uri == null) {
                OutlinedButton(
                    onClick = { launcher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary),
                    border = androidx.compose.foundation.BorderStroke(1.dp, TextSecondary.copy(alpha = 0.5f))
                ) {
                    Text("選取模擬圖片", style = MaterialTheme.typography.labelLarge)
                }
            }
        }
    }
}
