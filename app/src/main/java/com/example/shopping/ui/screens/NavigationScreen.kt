package com.example.shopping.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
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
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import coil.compose.AsyncImage
import com.example.shopping.model.ShoppingItem
import com.example.shopping.network.NavigationApi
import com.example.shopping.network.PlaceInfo
import com.example.shopping.network.StartSessionRequest
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import com.example.shopping.viewmodel.NavigationViewModel
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File

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

    // Neo4j map selection
    var neo4jPlaces by remember { mutableStateOf<List<PlaceInfo>>(emptyList()) }
    var isLoadingPlaces by remember { mutableStateOf(false) }
    var selectedPlace by remember { mutableStateOf<String?>(null) }
    var placeDropdownExpanded by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        isLoadingPlaces = true
        try {
            val places = withContext(Dispatchers.IO) {
                NavigationApi.service.getPlaces().places
            }
            neo4jPlaces = places
            if (places.size == 1) selectedPlace = places[0].name
        } catch (_: Exception) { }
        isLoadingPlaces = false
    }

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
                    NavigationApi.service.startSession(StartSessionRequest(goal = goalText, place = selectedPlace))
                }
                navController.currentBackStackEntry?.savedStateHandle?.set("nav_session_id", response.session_id)
                navController.currentBackStackEntry?.savedStateHandle?.set("nav_goal", goalText)
                navController.currentBackStackEntry?.savedStateHandle?.set("nav_guidance", response.guidance)
                navController.navigate("ar_navigation")
            } catch (e: Exception) {
                Log.e("NavSession", "Failed to create session", e)
                backendError = "無法連線導航伺服器，請確認伺服器已啟動後再試。"
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

            // Map selection dropdown
            if (neo4jPlaces.isNotEmpty()) {
                Spacer(Modifier.height(12.dp))
                StaggeredItem(index = 2) {
                    ExposedDropdownMenuBox(
                        expanded = placeDropdownExpanded,
                        onExpandedChange = { placeDropdownExpanded = it },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        OutlinedTextField(
                            value = selectedPlace ?: "選擇地圖（可選）",
                            onValueChange = {},
                            readOnly = true,
                            label = { Text("導航地圖") },
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = placeDropdownExpanded) },
                            modifier = Modifier.fillMaxWidth().menuAnchor(),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedTextColor = TextPrimary,
                                unfocusedTextColor = TextSecondary,
                                focusedBorderColor = Gold,
                                unfocusedBorderColor = TextTertiary,
                                focusedLabelColor = Gold,
                                unfocusedLabelColor = TextTertiary,
                            ),
                            shape = RoundedCornerShape(14.dp),
                        )
                        ExposedDropdownMenu(
                            expanded = placeDropdownExpanded,
                            onDismissRequest = { placeDropdownExpanded = false }
                        ) {
                            DropdownMenuItem(
                                text = { Text("不使用地圖") },
                                onClick = {
                                    selectedPlace = null
                                    placeDropdownExpanded = false
                                }
                            )
                            neo4jPlaces.forEach { place ->
                                DropdownMenuItem(
                                    text = { Text("${place.name}（${place.photo_count} 張參考照片）") },
                                    onClick = {
                                        selectedPlace = place.name
                                        placeDropdownExpanded = false
                                    }
                                )
                            }
                        }
                    }
                }
            } else if (isLoadingPlaces) {
                Spacer(Modifier.height(12.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Gold, strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text("載入地圖列表...", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                }
            }

            Spacer(Modifier.height(16.dp))

            StaggeredItem(index = 3) {
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

            StaggeredItem(index = 4) {
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
fun NavigationScreen(navController: NavController, vm: NavigationViewModel = viewModel()) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val fusedLocationClient = remember { LocationServices.getFusedLocationProviderClient(context) }

    val navSessionId = navController.previousBackStackEntry?.savedStateHandle?.get<String>("nav_session_id")
    val navGoal = navController.previousBackStackEntry?.savedStateHandle?.get<String>("nav_goal") ?: ""
    val navGuidance = navController.previousBackStackEntry?.savedStateHandle?.get<String>("nav_guidance") ?: ""

    LaunchedEffect(navSessionId) {
        vm.init(navSessionId, navGoal, navGuidance)
    }

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

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) vm.uploadFromGallery(context, uri)
    }

    // No session — show error and go back
    if (vm.sessionId == null) {
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

    // Camera torch (CameraX Camera handle from the preview binding)
    var boundCamera by remember { mutableStateOf<Camera?>(null) }
    var torchOn by remember { mutableStateOf(false) }
    val torchAvailable = boundCamera?.cameraInfo?.hasFlashUnit() == true
    // First-use photo guide; reopen with the "?" button
    var showPhotoGuide by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { if (consumeFirstPhotoGuide(context)) showPhotoGuide = true }
    val isPhoneUpright by rememberPhoneUpright()

    val statusTitle = when {
        vm.hasArrived -> if (vm.phase == "done") "導航完成！" else "已到達！"
        vm.pendingArrival -> "請確認是否到達"
        else -> "導航中"
    }
    val idle = !vm.hasArrived && !vm.pendingArrival && vm.pendingQuestion == null

    Box(modifier = Modifier.fillMaxSize().background(Noir)) {
        // Camera preview (full screen background)
        if (hasCameraPermission) {
            CameraPreviewWithCapture(lifecycleOwner, vm.imageCapture, onCameraBound = { boundCamera = it })
        }

        // AR store markers
        if (hasCameraPermission) {
            currentLocation?.let { loc ->
                // compass chip off: it sat on top of the status card
                NearbyStoresAr(location = loc, showCompass = false)
            }
        }

        // ── Overlay (same order as iOS): status card, checklist, …, route, guidance, buttons ──
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            NavStatusCard(
                title = statusTitle,
                phase = vm.phase,
                goal = vm.goal,
                currentTarget = vm.currentTarget,
                currentLocation = vm.currentLocation,
                torchOn = torchOn,
                torchAvailable = torchAvailable,
                onToggleTorch = {
                    boundCamera?.let { cam ->
                        torchOn = !torchOn
                        cam.cameraControl.enableTorch(torchOn)
                    }
                },
                onShowPhotoGuide = { showPhotoGuide = true },
            )
            if (vm.goalItems.isNotEmpty()) {
                ShoppingChecklist(items = vm.goalItems, currentTarget = vm.currentTarget)
            }
            if (torchAvailable && !torchOn) {
                Text(
                    "光線不足嗎？點擊右上角閃電圖示開啟閃光燈輔助拍照",
                    color = Color.White.copy(alpha = 0.75f),
                    fontSize = 11.sp,
                )
            }

            // Middle: camera view area — AR guidance arrow overlay
            Box(
                modifier = Modifier.weight(1f).fillMaxWidth(),
                contentAlignment = Alignment.Center,
            ) {
                if (vm.directionDeg != null && idle) {
                    ArGuidanceArrow(
                        directionDeg = vm.directionDeg ?: 0f,
                        directionLabel = vm.directionLabel,
                    )
                }
            }

            // Hints, route and guidance scroll together and never push the action
            // buttons off screen (large system font sizes overflowed them).
            Column(
                modifier = Modifier
                    .weight(2f, fill = false)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
            // One hint at a time, most urgent first (as on iOS)
            when {
                vm.retakeHint != null -> NavHintBanner("📷 ${vm.retakeHint}", Color(0xFFFF9800).copy(alpha = 0.9f))
                vm.secondPhotoHint != null -> NavHintBanner("📷 ${vm.secondPhotoHint}", Color(0xFF2196F3).copy(alpha = 0.85f))
                idle && !isPhoneUpright -> NavHintBanner("📱 手機請拿直、對準走道前方", Color.Black.copy(alpha = 0.55f))
            }

            RoutePanel(
                route = vm.route,
                prompt = vm.routeGuidance,
                statusMessage = vm.routeStatusMessage,
                userHeadingDeg = vm.confidentHeadingDeg,
            )

            // ── Guidance panel ──
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(16.dp))
                    .background(Color(0xFF616161).copy(alpha = 0.45f))
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("導航指引", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = Color.White.copy(alpha = 0.7f))
                Text(vm.guidance, fontSize = 14.sp, color = Color.White, lineHeight = 20.sp)
                vm.mapInstruction?.let { m ->
                    Text("🗺 $m", fontSize = 14.sp, fontWeight = FontWeight.Medium, color = Color(0xFF00BCD4))
                }

                // ASK flow
                if (vm.pendingQuestion != null && !vm.hasArrived) {
                    Text(vm.pendingQuestion!!, fontSize = 14.sp, fontWeight = FontWeight.Medium, color = Color(0xFFFF9800))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        OutlinedTextField(
                            value = vm.answerText,
                            onValueChange = { vm.answerText = it },
                            placeholder = { Text("輸入回答...", color = Color.White.copy(alpha = 0.5f)) },
                            modifier = Modifier.weight(1f),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedTextColor = Color.White,
                                unfocusedTextColor = Color.White,
                                focusedBorderColor = Color(0xFFFF9800),
                                unfocusedBorderColor = Color.White.copy(alpha = 0.5f),
                                cursorColor = Color(0xFFFF9800)
                            ),
                            singleLine = true,
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                            keyboardActions = KeyboardActions(onSend = { vm.submitAnswer() })
                        )
                        Spacer(Modifier.width(8.dp))
                        Button(
                            onClick = { vm.submitAnswer() },
                            enabled = vm.answerText.isNotBlank() && !vm.isAnswering,
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFFFF9800),
                                contentColor = Color.White
                            ),
                            shape = CircleShape,
                            contentPadding = PaddingValues(12.dp),
                            modifier = Modifier.size(48.dp)
                        ) {
                            if (vm.isAnswering) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                            } else {
                                Icon(Icons.Default.Send, null, modifier = Modifier.size(18.dp))
                            }
                        }
                    }
                }

                vm.errorMessage?.let { err ->
                    Text(err, color = Color(0xFFFF6B6B), fontSize = 12.sp)
                }
            }
            } // hints + route + guidance

            // ── Bottom: Action buttons ──
            if (vm.pendingArrival) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Button(
                        onClick = { vm.confirmArrival("confirmed") },
                        enabled = !vm.isConfirming,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF4CAF50), contentColor = Color.White),
                        shape = RoundedCornerShape(24.dp),
                        modifier = Modifier.fillMaxWidth().height(46.dp)
                    ) {
                        if (vm.isConfirming) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                        } else {
                            Text(
                                when (vm.phase) {
                                    "checkout" -> "確認已結帳"
                                    "exit" -> "確認已到出口"
                                    else -> "確認到達"
                                },
                                fontSize = 14.sp, fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = { vm.confirmArrival("false_positive") },
                            enabled = !vm.isConfirming,
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF9E9E9E), contentColor = Color.White),
                            shape = RoundedCornerShape(24.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                            modifier = Modifier.weight(1f).height(42.dp)
                        ) {
                            Text("不是目標", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
                        }
                        Button(
                            onClick = { vm.confirmArrival("wrong_instance") },
                            enabled = !vm.isConfirming,
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF9C27B0), contentColor = Color.White),
                            shape = RoundedCornerShape(24.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                            modifier = Modifier.weight(1f).height(42.dp)
                        ) {
                            Text("同類非目標", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
                        }
                    }
                }
            } else {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(
                        onClick = { navController.popBackStack() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (vm.hasArrived) Color(0xFF4CAF50) else Color(0xFFEF5350),
                            contentColor = Color.White
                        ),
                        shape = RoundedCornerShape(24.dp),
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                        modifier = Modifier.weight(1f).height(46.dp)
                    ) {
                        Text(if (vm.hasArrived) "完成" else "結束導航", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
                    }

                    if (!vm.hasArrived && vm.pendingQuestion == null) {
                        Button(
                            onClick = { vm.captureAndUpload(context) },
                            enabled = !vm.isUploading,
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF66BB6A), contentColor = Color.White),
                            shape = RoundedCornerShape(24.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                            modifier = Modifier.weight(1f).height(46.dp)
                        ) {
                            if (vm.isUploading) {
                                CircularProgressIndicator(modifier = Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp)
                                Spacer(Modifier.width(4.dp))
                                Text("分析中...", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
                            } else {
                                Text("拍照分析", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
                            }
                        }

                        Button(
                            onClick = { galleryLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                            enabled = !vm.isUploading,
                            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF42A5F5), contentColor = Color.White),
                            shape = RoundedCornerShape(24.dp),
                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp),
                            modifier = Modifier.weight(1f).height(46.dp)
                        ) {
                            Text("選擇照片", fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
                        }
                    }
                }
            }
        }

        if (showPhotoGuide) {
            PhotoGuideDialog(onDismiss = { showPhotoGuide = false })
        }
    }
}

// ── Camera preview with ImageCapture support ──

@Composable
fun CameraPreviewWithCapture(
    lifecycleOwner: LifecycleOwner,
    imageCapture: ImageCapture,
    onCameraBound: (Camera) -> Unit = {},
) {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    AndroidView(factory = { ctx ->
        val previewView = PreviewView(ctx).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }

            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
                onCameraBound(camera)
            } catch (e: Exception) {
                Log.e("CameraX", "綁定失敗", e)
            }
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    }, modifier = Modifier.fillMaxSize())
}
