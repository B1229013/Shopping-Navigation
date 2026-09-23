package com.example.shopping.ui.screens

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.SensorManager
import android.os.Build
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavController
import coil.compose.AsyncImage
import com.example.shopping.network.SensorNavApi
import com.example.shopping.sensor.PdrTracker
import com.example.shopping.ui.theme.*
import com.example.shopping.viewmodel.LocalizationPhase
import com.example.shopping.viewmodel.SensorNavViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// ── Entry screen: set goal + optional map selection ──

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SensorNavHomeScreen(navController: NavController) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var goalText by remember { mutableStateOf("") }
    var isStarting by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var offlineMode by remember { mutableStateOf(false) }
    var pdrEnabled by remember { mutableStateOf(true) }
    var selectedPlace by remember { mutableStateOf<String?>(null) }
    var placesExpanded by remember { mutableStateOf(false) }
    var currentBackendUrl by remember { mutableStateOf(com.example.shopping.network.BackendConfig.currentUrl) }

    // Local sessions
    val vm: SensorNavViewModel = viewModel()
    LaunchedEffect(Unit) {
        vm.loadLocalSessions(context)
        vm.loadNeo4jPlaces()
    }

    Scaffold(
        containerColor = Noir,
        topBar = {
            TopAppBar(
                title = { Text("AI 視覺導航", style = MaterialTheme.typography.titleLarge, color = TextPrimary) },
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
                .padding(horizontal = 20.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Spacer(Modifier.height(20.dp))

            Text(
                "AI 視覺導航",
                style = MaterialTheme.typography.headlineSmall,
                color = TextPrimary,
                fontWeight = FontWeight.Bold,
            )
            Text(
                "拍照時 AI 分析場景給出方向指示。可選擇預建地圖進行路線規劃，或即時探索。",
                style = MaterialTheme.typography.bodyMedium,
                color = TextSecondary,
            )

            // Offline mode toggle
            Surface(
                color = if (offlineMode) ChartBlue.copy(alpha = 0.12f) else SurfaceBase,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.clickable { offlineMode = !offlineMode },
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        if (offlineMode) Icons.Default.PhoneAndroid else Icons.Default.Cloud,
                        contentDescription = null,
                        tint = if (offlineMode) ChartBlue else TextTertiary,
                        modifier = Modifier.size(20.dp),
                    )
                    Spacer(Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            if (offlineMode) "離線採集模式" else "連線導航模式",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary,
                        )
                        Text(
                            if (offlineMode) "不需後端，照片+感測器資料存手機" else "即時上傳後端，AI 分析導航",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                        )
                    }
                    Switch(
                        checked = offlineMode,
                        onCheckedChange = null,
                        colors = SwitchDefaults.colors(checkedTrackColor = ChartBlue),
                    )
                }
            }

            // PDR sensor toggle
            Surface(
                color = if (pdrEnabled) Gold.copy(alpha = 0.12f) else SurfaceBase,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.clickable { pdrEnabled = !pdrEnabled },
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Default.DirectionsWalk,
                        contentDescription = null,
                        tint = if (pdrEnabled) Gold else TextTertiary,
                        modifier = Modifier.size(20.dp),
                    )
                    Spacer(Modifier.width(12.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "PDR 感測器",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary,
                        )
                        Text(
                            if (pdrEnabled) "計步、距離、方向追蹤" else "不使用感測器",
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                        )
                    }
                    Switch(
                        checked = pdrEnabled,
                        onCheckedChange = null,
                        colors = SwitchDefaults.colors(checkedTrackColor = Gold),
                    )
                }
            }

            // Backend URL info (only in online mode)
            if (!offlineMode) {
                Surface(
                    color = SurfaceBase,
                    shape = RoundedCornerShape(12.dp),
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 10.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Icon(
                            Icons.Default.Dns,
                            contentDescription = null,
                            tint = TextTertiary,
                            modifier = Modifier.size(18.dp),
                        )
                        Spacer(Modifier.width(10.dp))
                        Text(
                            currentBackendUrl,
                            style = MaterialTheme.typography.bodySmall,
                            color = TextSecondary,
                            modifier = Modifier.weight(1f),
                        )
                        Text(
                            "設定頁可變更",
                            style = MaterialTheme.typography.labelSmall,
                            color = TextTertiary,
                        )
                    }
                }
            }

            Spacer(Modifier.height(8.dp))

            OutlinedTextField(
                value = goalText,
                onValueChange = { goalText = it },
                label = { Text(if (offlineMode) "路線標記" else "目的地") },
                placeholder = { Text(if (offlineMode) "例如：3F走廊到飲水機" else "例如：找飲水機、去 305 教室") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Go),
                keyboardActions = KeyboardActions(
                    onGo = {
                        if (goalText.isNotBlank() && !isStarting) {
                            if (offlineMode) {
                                navController.currentBackStackEntry?.savedStateHandle?.apply {
                                    set("snav_goal", goalText)
                                    set("snav_offline", true)
                                    set("snav_pdr_enabled", pdrEnabled)
                                }
                                navController.navigate("sensor_nav_active")
                            } else {
                                isStarting = true
                                scope.launch {
                                    try {
                                        val resp = withContext(Dispatchers.IO) {
                                            SensorNavApi.service.startSession(
                                                com.example.shopping.network.SNavStartRequest(
                                                    goal = goalText,
                                                    place_name = selectedPlace,
                                                )
                                            )
                                        }
                                        navController.currentBackStackEntry?.savedStateHandle?.apply {
                                            set("snav_session_id", resp.session_id)
                                            set("snav_goal", goalText)
                                            set("snav_guidance", resp.guidance)
                                            set("snav_nav_mode", resp.nav_mode)
                                            set("snav_place_name", resp.place_name)
                                            set("snav_pdr_enabled", pdrEnabled)
                                        }
                                        navController.navigate("sensor_nav_active")
                                    } catch (e: Exception) {
                                        errorMessage = "建立導航失敗: ${e.message}"
                                        isStarting = false
                                    }
                                }
                            }
                        }
                    }
                ),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = Gold,
                    cursorColor = Gold,
                )
            )

            // Neo4j map selection (only in online mode)
            if (!offlineMode) {
                ExposedDropdownMenuBox(
                    expanded = placesExpanded,
                    onExpandedChange = { placesExpanded = it },
                ) {
                    OutlinedTextField(
                        value = selectedPlace ?: "不使用地圖（即時探索）",
                        onValueChange = {},
                        readOnly = true,
                        label = { Text("選擇地圖") },
                        trailingIcon = {
                            if (vm.isLoadingPlaces) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = Gold,
                                    strokeWidth = 2.dp,
                                )
                            } else {
                                ExposedDropdownMenuDefaults.TrailingIcon(expanded = placesExpanded)
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor(),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Gold,
                        ),
                    )
                    ExposedDropdownMenu(
                        expanded = placesExpanded,
                        onDismissRequest = { placesExpanded = false },
                    ) {
                        DropdownMenuItem(
                            text = { Text("不使用地圖（即時探索）") },
                            onClick = {
                                selectedPlace = null
                                placesExpanded = false
                            },
                        )
                        vm.neo4jPlaces.forEach { place ->
                            DropdownMenuItem(
                                text = { Text("${place.place_name}（${place.node_count} 節點）") },
                                onClick = {
                                    selectedPlace = place.place_name
                                    placesExpanded = false
                                },
                            )
                        }
                    }
                }
            }

            Button(
                onClick = {
                    if (goalText.isNotBlank() && !isStarting) {
                        if (offlineMode) {
                            navController.currentBackStackEntry?.savedStateHandle?.apply {
                                set("snav_goal", goalText)
                                set("snav_offline", true)
                                set("snav_pdr_enabled", pdrEnabled)
                            }
                            navController.navigate("sensor_nav_active")
                        } else {
                            isStarting = true
                            scope.launch {
                                try {
                                    val resp = withContext(Dispatchers.IO) {
                                        SensorNavApi.service.startSession(
                                            com.example.shopping.network.SNavStartRequest(
                                                goal = goalText,
                                                place_name = selectedPlace,
                                            )
                                        )
                                    }
                                    navController.currentBackStackEntry?.savedStateHandle?.apply {
                                        set("snav_session_id", resp.session_id)
                                        set("snav_goal", goalText)
                                        set("snav_guidance", resp.guidance)
                                        set("snav_nav_mode", resp.nav_mode)
                                        set("snav_place_name", resp.place_name)
                                        set("snav_pdr_enabled", pdrEnabled)
                                    }
                                    navController.navigate("sensor_nav_active")
                                } catch (e: Exception) {
                                    errorMessage = "建立導航失敗: ${e.message}"
                                    isStarting = false
                                }
                            }
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth().height(52.dp),
                enabled = goalText.isNotBlank() && !isStarting,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (offlineMode) ChartBlue else Gold
                ),
                shape = RoundedCornerShape(12.dp),
            ) {
                if (isStarting) {
                    CircularProgressIndicator(modifier = Modifier.size(20.dp), color = Color.White, strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                }
                Icon(
                    if (offlineMode) Icons.Default.PhoneAndroid else Icons.Default.Navigation,
                    contentDescription = null,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (offlineMode) "開始採集" else "開始導航",
                    fontWeight = FontWeight.Bold,
                )
            }

            errorMessage?.let {
                Text(it, color = Danger, style = MaterialTheme.typography.bodySmall)
            }

            // Local sessions list
            if (vm.localSessions.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "本地採集紀錄",
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary,
                    fontWeight = FontWeight.Bold,
                )
                vm.localSessions.forEach { session ->
                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = SurfaceBase,
                        shape = RoundedCornerShape(12.dp),
                    ) {
                        Row(
                            modifier = Modifier.padding(14.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    session.goal,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.SemiBold,
                                    color = TextPrimary,
                                )
                                Text(
                                    "${session.photoCount} 張照片 · ${session.totalSteps} 步 · ${"%.1f".format(session.totalDistanceM)}m",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextSecondary,
                                )
                                Text(
                                    session.createdAt,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextTertiary,
                                    fontSize = 11.sp,
                                )
                            }
                            IconButton(onClick = {
                                vm.shareLocalSession(context, session.id)
                            }) {
                                Icon(Icons.Default.Share, contentDescription = "匯出", tint = ChartBlue, modifier = Modifier.size(18.dp))
                            }
                            IconButton(onClick = {
                                vm.deleteLocalSession(context, session.id)
                            }) {
                                Icon(Icons.Default.Delete, contentDescription = "刪除", tint = Danger, modifier = Modifier.size(18.dp))
                            }
                        }
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
        }
    }
}

// ── Active navigation screen ──

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SensorNavActiveScreen(navController: NavController) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val vm: SensorNavViewModel = viewModel()

    // Get session data from previous screen
    val sessionId = navController.previousBackStackEntry?.savedStateHandle?.get<String>("snav_session_id")
    val goal = navController.previousBackStackEntry?.savedStateHandle?.get<String>("snav_goal") ?: ""
    val initialGuidance = navController.previousBackStackEntry?.savedStateHandle?.get<String>("snav_guidance") ?: ""
    val isOffline = navController.previousBackStackEntry?.savedStateHandle?.get<Boolean>("snav_offline") == true
    val pdrEnabled = navController.previousBackStackEntry?.savedStateHandle?.get<Boolean>("snav_pdr_enabled") != false

    // Camera permission
    var cameraPermissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        cameraPermissionGranted = it
    }
    LaunchedEffect(Unit) {
        if (!cameraPermissionGranted) permLauncher.launch(Manifest.permission.CAMERA)
    }

    // Activity recognition permission — only needed when PDR is enabled
    var activityPermGranted by remember {
        mutableStateOf(
            !pdrEnabled ||
            Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
        )
    }
    val activityPermLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        activityPermGranted = it
    }
    LaunchedEffect(pdrEnabled) {
        if (pdrEnabled && !activityPermGranted) activityPermLauncher.launch(Manifest.permission.ACTIVITY_RECOGNITION)
    }

    // Initialize session + PDR
    LaunchedEffect(sessionId, isOffline, activityPermGranted) {
        if (pdrEnabled && !activityPermGranted) return@LaunchedEffect

        if (vm.sessionId == null) {
            if (isOffline) {
                vm.isOfflineMode = true
                vm.startSession(context, goal)
            } else if (sessionId != null) {
                vm.sessionId = sessionId
                vm.goal = goal
                vm.guidance = initialGuidance
                if (pdrEnabled) vm.startPdr(context)
            }
        } else if (pdrEnabled && vm.pdrTracker == null) {
            vm.startPdr(context)
        }
    }

    // Gallery picker
    val galleryLauncher = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { vm.uploadFromGallery(context, it) }
    }

    // Save map dialog
    var showSaveDialog by remember { mutableStateOf(false) }
    var mapName by remember { mutableStateOf("") }

    // Update PDR display values periodically
    LaunchedEffect(vm.pdrTracker) {
        while (vm.pdrTracker != null) {
            vm.pdrTracker?.let {
                vm.pdrX = it.x
                vm.pdrY = it.y
                vm.pdrTotalSteps = it.totalSteps
                vm.pdrTotalDistance = it.totalDistance
            }
            kotlinx.coroutines.delay(500)
        }
    }

    DisposableEffect(Unit) {
        onDispose { vm.stopPdr() }
    }

    Scaffold(
        containerColor = Noir,
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            if (vm.isOfflineMode) "離線採集" else "感測器導航",
                            style = MaterialTheme.typography.titleMedium, color = TextPrimary,
                        )
                        Text(vm.goal, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                    }
                },
                navigationIcon = {
                    IconButton(onClick = {
                        vm.stopPdr()
                        navController.popBackStack()
                    }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "返回", tint = TextSecondary)
                    }
                },
                actions = {
                    if (vm.hasArrived && !vm.isOfflineMode) {
                        IconButton(onClick = { showSaveDialog = true }) {
                            Icon(Icons.Default.Save, contentDescription = "儲存地圖", tint = Gold)
                        }
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
        ) {
            // Camera preview (top half)
            if (cameraPermissionGranted && !vm.hasArrived) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                ) {
                    SNavCameraPreview(lifecycleOwner, vm.imageCapture)

                    // PDR stats overlay (top-left)
                    Surface(
                        modifier = Modifier
                            .align(Alignment.TopStart)
                            .padding(8.dp),
                        color = Color.Black.copy(alpha = 0.6f),
                        shape = RoundedCornerShape(8.dp),
                    ) {
                        Column(modifier = Modifier.padding(8.dp)) {
                            Text(
                                "${vm.pdrTotalSteps} 步 · ${"%.1f".format(vm.pdrTotalDistance)}m",
                                color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                            )
                            vm.stepGuidance?.let {
                                Text(it, color = Color(0xFF86EFAC), fontSize = 11.sp)
                            }
                        }
                    }

                    // Photo count overlay (top-right)
                    if (vm.photoCount > 0) {
                        Surface(
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .padding(8.dp),
                            color = Color.Black.copy(alpha = 0.6f),
                            shape = RoundedCornerShape(8.dp),
                        ) {
                            Text(
                                "  #${vm.photoCount}  ",
                                color = Color.White, fontSize = 12.sp,
                                modifier = Modifier.padding(vertical = 4.dp),
                            )
                        }
                    }
                }
            }

            // Annotated photo (show when arrived)
            if (vm.hasArrived && vm.annotatedPhotoUrl != null) {
                val baseUrl = com.example.shopping.network.BackendConfig.currentUrl.trimEnd('/')
                AsyncImage(
                    model = "$baseUrl${vm.annotatedPhotoUrl}",
                    contentDescription = "標註照片",
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentScale = ContentScale.Fit,
                )
            }

            // Mini PDR trail map / topomap graph
            if (vm.hasArrived && vm.photoNodes.isNotEmpty()) {
                PdrTopoGraph(
                    photoNodes = vm.photoNodes,
                    pathPoints = vm.pdrPathPoints,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(220.dp)
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                )
            } else if (vm.pdrPathPoints.isNotEmpty()) {
                PdrMiniTrail(
                    pathPoints = vm.pdrPathPoints,
                    currentX = vm.pdrX,
                    currentY = vm.pdrY,
                    photoNodes = vm.photoNodes,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(100.dp)
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                )
            }

            // Guidance panel
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = SurfaceBase,
                shape = RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp),
                shadowElevation = 4.dp,
            ) {
                Column(
                    modifier = Modifier
                        .padding(16.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    // Multi-goal progress chips
                    if (vm.totalGoals > 1 && vm.subGoals.isNotEmpty()) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .horizontalScroll(rememberScrollState()),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            vm.subGoals.forEachIndexed { idx, sg ->
                                val isCurrent = idx == vm.currentGoalIdx
                                Surface(
                                    color = when {
                                        sg.arrived -> Color(0xFF10B981)
                                        isCurrent -> Color(0xFF3B82F6)
                                        else -> Color(0xFF374151)
                                    },
                                    shape = RoundedCornerShape(16.dp),
                                ) {
                                    Row(
                                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
                                        verticalAlignment = Alignment.CenterVertically,
                                    ) {
                                        if (sg.arrived) {
                                            Icon(Icons.Default.Check, contentDescription = null, tint = Color.White, modifier = Modifier.size(14.dp))
                                            Spacer(Modifier.width(4.dp))
                                        }
                                        Text(
                                            sg.name,
                                            color = Color.White,
                                            fontSize = 12.sp,
                                            fontWeight = if (isCurrent) FontWeight.Bold else FontWeight.Normal,
                                        )
                                    }
                                }
                            }
                        }
                    }

                    // Guidance text
                    Text(
                        vm.guidance,
                        style = MaterialTheme.typography.bodyLarge,
                        color = TextPrimary,
                        fontWeight = FontWeight.Medium,
                    )

                    // Step guidance
                    vm.stepGuidance?.let {
                        Surface(
                            color = GoldSurface,
                            shape = RoundedCornerShape(8.dp),
                        ) {
                            Row(
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                                verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Icon(Icons.Default.NearMe, contentDescription = null, tint = Gold, modifier = Modifier.size(18.dp))
                                Spacer(Modifier.width(8.dp))
                                Text(it, color = GoldDim, fontSize = 13.sp, fontWeight = FontWeight.Medium)
                            }
                        }
                    }

                    // Error message
                    vm.errorMessage?.let {
                        Text(it, color = Danger, style = MaterialTheme.typography.bodySmall)
                    }

                    // --- Sensor calibration section ---
                    // Only show calibration cards while sensors are NOT all ready.
                    // "All ready" = mag >= MEDIUM, gyro == CALIBRATED, north calibrated, GPS ready.
                    val allSensorsReady = pdrEnabled &&
                        vm.magCalibrationStatus >= SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM &&
                        vm.gyroCalState == PdrTracker.GyroCalState.CALIBRATED &&
                        vm.northCalibrated &&
                        vm.gpsStatus == SensorNavViewModel.GpsStatus.READY

                    if (pdrEnabled && !allSensorsReady && !vm.hasArrived) {
                        // Magnetometer calibration hint with live accuracy feedback
                        if (vm.magCalibrationStatus <= SensorManager.SENSOR_STATUS_ACCURACY_LOW) {
                            val (statusText, statusColor, barWidth) = when (vm.magCalibrationStatus) {
                                SensorManager.SENSOR_STATUS_UNRELIABLE -> Triple("未校正", Color(0xFFDC2626), 0.1f)
                                else -> Triple("精度低", Color(0xFFD97706), 0.35f)
                            }
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF7ED)),
                            ) {
                                Column(modifier = Modifier.padding(12.dp)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(Icons.Default.Warning, contentDescription = null, tint = Color(0xFFD97706), modifier = Modifier.size(20.dp))
                                        Spacer(Modifier.width(8.dp))
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text("磁力計校正：$statusText", fontWeight = FontWeight.Medium, fontSize = 13.sp, color = Color(0xFF92400E))
                                            Text("請將手機在空中畫 8 字形", fontSize = 12.sp, color = Color(0xFFB45309))
                                        }
                                    }
                                    Spacer(Modifier.height(8.dp))
                                    Box(
                                        modifier = Modifier.fillMaxWidth().height(4.dp)
                                            .clip(RoundedCornerShape(2.dp))
                                            .background(Color(0xFFE5E7EB))
                                    ) {
                                        Box(
                                            modifier = Modifier.fillMaxHeight()
                                                .fillMaxWidth(barWidth)
                                                .clip(RoundedCornerShape(2.dp))
                                                .background(statusColor)
                                        )
                                    }
                                }
                            }
                        }

                        // Gyro calibration — only show when not yet calibrated
                        if (vm.gyroCalState != PdrTracker.GyroCalState.CALIBRATED) {
                            val (gyroText, gyroColor, gyroBg) = when (vm.gyroCalState) {
                                PdrTracker.GyroCalState.UNCALIBRATED -> Triple("未校正", Color(0xFFD97706), Color(0xFFFFF7ED))
                                PdrTracker.GyroCalState.CALIBRATING -> Triple("校正中…", Color(0xFF2563EB), Color(0xFFEFF6FF))
                                PdrTracker.GyroCalState.FAILED -> Triple("校正失敗", Color(0xFFDC2626), Color(0xFFFEF2F2))
                                else -> Triple("", Color.Transparent, Color.Transparent)
                            }
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = gyroBg),
                            ) {
                                Column(modifier = Modifier.padding(12.dp)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            when (vm.gyroCalState) {
                                                PdrTracker.GyroCalState.CALIBRATING -> Icons.Default.Sync
                                                PdrTracker.GyroCalState.FAILED -> Icons.Default.ErrorOutline
                                                else -> Icons.Default.Warning
                                            },
                                            null, tint = gyroColor, modifier = Modifier.size(20.dp),
                                        )
                                        Spacer(Modifier.width(8.dp))
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text("陀螺儀校正：$gyroText", fontWeight = FontWeight.Medium, fontSize = 13.sp, color = TextPrimary)
                                            when (vm.gyroCalState) {
                                                PdrTracker.GyroCalState.UNCALIBRATED -> Text("保持手機靜止 3 秒", fontSize = 12.sp, color = TextTertiary)
                                                PdrTracker.GyroCalState.CALIBRATING -> Text("請保持靜止不動…", fontSize = 12.sp, color = Color(0xFF2563EB))
                                                PdrTracker.GyroCalState.FAILED -> Text("手機晃動過大，請握穩後重試", fontSize = 12.sp, color = Color(0xFFDC2626))
                                                else -> {}
                                            }
                                        }
                                    }
                                    if (vm.gyroCalState == PdrTracker.GyroCalState.CALIBRATING) {
                                        Spacer(Modifier.height(8.dp))
                                        LinearProgressIndicator(
                                            progress = { vm.gyroCalProgress },
                                            modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                                            color = gyroColor, trackColor = Color(0xFFE5E7EB),
                                        )
                                    }
                                    if (vm.gyroCalState != PdrTracker.GyroCalState.CALIBRATING) {
                                        Spacer(Modifier.height(8.dp))
                                        Button(
                                            onClick = { vm.startGyroCalibration() },
                                            modifier = Modifier.fillMaxWidth().height(36.dp),
                                            colors = ButtonDefaults.buttonColors(containerColor = gyroColor),
                                            shape = RoundedCornerShape(8.dp),
                                        ) {
                                            Text(
                                                when (vm.gyroCalState) {
                                                    PdrTracker.GyroCalState.FAILED -> "重試校正"
                                                    else -> "開始校正"
                                                },
                                                fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        // North calibration warning
                        if (!vm.northCalibrated) {
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFFFF7ED)),
                            ) {
                                Row(
                                    modifier = Modifier.padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                ) {
                                    Icon(Icons.Default.Warning, null, tint = Color(0xFFD97706), modifier = Modifier.size(20.dp))
                                    Text(
                                        if (vm.magCalibrationStatus < SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM)
                                            "磁力計精度不足，北方校正未完成。請畫 8 字形校準。"
                                        else
                                            "等待北方校正中…",
                                        fontSize = 12.sp, color = Color(0xFFB45309),
                                    )
                                }
                            }
                        }

                        // GPS convergence status
                        if (vm.gpsStatus != SensorNavViewModel.GpsStatus.READY) {
                            val (gpsText, gpsColor, gpsBar) = when (vm.gpsStatus) {
                                SensorNavViewModel.GpsStatus.WAITING -> Triple("等待衛星訊號…", Color(0xFFDC2626), 0.1f)
                                SensorNavViewModel.GpsStatus.CONVERGING -> Triple("GPS 收斂中…", Color(0xFFD97706), 0.5f)
                                else -> Triple("", Color.Transparent, 0f)
                            }
                            Card(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFFEF3C7)),
                            ) {
                                Column(modifier = Modifier.padding(12.dp)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(Icons.Default.LocationOn, contentDescription = null, tint = gpsColor, modifier = Modifier.size(20.dp))
                                        Spacer(Modifier.width(8.dp))
                                        Column {
                                            Text(gpsText, fontWeight = FontWeight.Medium, fontSize = 13.sp, color = Color(0xFF92400E))
                                            Text("請在空曠處等待 GPS 定位穩定", fontSize = 12.sp, color = Color(0xFFB45309))
                                        }
                                    }
                                    Spacer(Modifier.height(8.dp))
                                    Box(
                                        modifier = Modifier.fillMaxWidth().height(4.dp)
                                            .clip(RoundedCornerShape(2.dp))
                                            .background(Color(0xFFE5E7EB))
                                    ) {
                                        Box(
                                            modifier = Modifier.fillMaxHeight()
                                                .fillMaxWidth(gpsBar)
                                                .clip(RoundedCornerShape(2.dp))
                                                .background(gpsColor)
                                        )
                                    }
                                }
                            }
                        }
                    }

                    // Action buttons
                    when {
                        vm.pendingArrival -> {
                            Text("系統認為已到達目的地，請確認：", color = TextSecondary, fontSize = 13.sp)
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                Button(
                                    onClick = { vm.confirmArrival("confirmed") },
                                    colors = ButtonDefaults.buttonColors(containerColor = Success),
                                    modifier = Modifier.weight(1f),
                                    enabled = !vm.isConfirming,
                                ) { Text("確認到達") }

                                OutlinedButton(
                                    onClick = { vm.confirmArrival("false_positive") },
                                    modifier = Modifier.weight(1f),
                                    enabled = !vm.isConfirming,
                                ) { Text("看錯了") }

                                OutlinedButton(
                                    onClick = { vm.confirmArrival("wrong_instance") },
                                    modifier = Modifier.weight(1f),
                                    enabled = !vm.isConfirming,
                                ) { Text("不是這個") }
                            }
                        }

                        vm.pendingQuestion != null -> {
                            Text(vm.pendingQuestion!!, color = Info, fontWeight = FontWeight.Medium)
                            OutlinedTextField(
                                value = vm.answerText,
                                onValueChange = { vm.answerText = it },
                                label = { Text("回答") },
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true,
                                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                                keyboardActions = KeyboardActions(onSend = { vm.submitAnswer() }),
                            )
                            Button(
                                onClick = { vm.submitAnswer() },
                                modifier = Modifier.fillMaxWidth(),
                                enabled = vm.answerText.isNotBlank() && !vm.isAnswering,
                                colors = ButtonDefaults.buttonColors(containerColor = Gold),
                            ) { Text("送出回答") }
                        }

                        vm.hasArrived -> {
                            Surface(color = SuccessDim, shape = RoundedCornerShape(8.dp)) {
                                Row(
                                    modifier = Modifier.padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                ) {
                                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Success)
                                    Spacer(Modifier.width(8.dp))
                                    Column {
                                        Text(
                                            if (vm.isOfflineMode) "採集完成！" else "已到達！",
                                            fontWeight = FontWeight.Bold, color = GoldDim,
                                        )
                                        Text(
                                            "共 ${vm.pdrTotalSteps} 步 · ${"%.1f".format(vm.pdrTotalDistance)}m · ${vm.photoCount} 張照片",
                                            fontSize = 12.sp, color = TextSecondary,
                                        )
                                    }
                                }
                            }
                            if (!vm.isOfflineMode) {
                                Button(
                                    onClick = { showSaveDialog = true },
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = ButtonDefaults.buttonColors(containerColor = Gold),
                                ) {
                                    Icon(Icons.Default.Save, contentDescription = null)
                                    Spacer(Modifier.width(8.dp))
                                    Text("儲存地圖")
                                }
                            }
                        }

                        else -> {
                            if (vm.localizationPhase == LocalizationPhase.CONFIRMING && vm.localizationCandidates.isNotEmpty()) {
                                // Candidate confirmation UI
                                Text(
                                    "請確認您的位置（選擇最接近的參考照片）：",
                                    color = Color(0xFF93C5FD),
                                    fontSize = 14.sp,
                                    fontWeight = FontWeight.Medium,
                                )

                                val baseUrl = com.example.shopping.network.BackendConfig.currentUrl.trimEnd('/')

                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .horizontalScroll(rememberScrollState()),
                                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                                ) {
                                    vm.localizationCandidates.forEach { candidate ->
                                        val nid = (candidate["nid"] as? Number)?.toInt() ?: return@forEach
                                        val score = (candidate["score"] as? Number)?.toDouble() ?: 0.0
                                        val region = candidate["region"] as? String ?: ""
                                        val photoUrl = candidate["photo_url"] as? String

                                        Card(
                                            modifier = Modifier
                                                .width(150.dp)
                                                .clickable(enabled = !vm.isConfirmingLocation) {
                                                    vm.confirmLocation(nid)
                                                },
                                            colors = CardDefaults.cardColors(containerColor = Color(0xFF1E293B)),
                                            shape = RoundedCornerShape(12.dp),
                                            border = BorderStroke(1.dp, Color(0xFF3B82F6).copy(alpha = 0.5f)),
                                        ) {
                                            Column {
                                                if (photoUrl != null) {
                                                    AsyncImage(
                                                        model = "$baseUrl$photoUrl",
                                                        contentDescription = "P$nid",
                                                        modifier = Modifier
                                                            .fillMaxWidth()
                                                            .height(100.dp),
                                                        contentScale = ContentScale.Crop,
                                                    )
                                                } else {
                                                    Box(
                                                        modifier = Modifier
                                                            .fillMaxWidth()
                                                            .height(100.dp)
                                                            .background(Color(0xFF374151)),
                                                        contentAlignment = Alignment.Center,
                                                    ) {
                                                        Text("P$nid", color = Color(0xFF9CA3AF), fontSize = 18.sp, fontWeight = FontWeight.Bold)
                                                    }
                                                }
                                                Column(modifier = Modifier.padding(8.dp)) {
                                                    Text(
                                                        "P$nid",
                                                        color = Color.White,
                                                        fontSize = 14.sp,
                                                        fontWeight = FontWeight.Bold,
                                                    )
                                                    Text(
                                                        "${"%.0f".format(score * 100)}%",
                                                        color = Color(0xFF60A5FA),
                                                        fontSize = 13.sp,
                                                    )
                                                    if (region.isNotEmpty()) {
                                                        Text(
                                                            region,
                                                            color = Color(0xFF9CA3AF),
                                                            fontSize = 11.sp,
                                                            maxLines = 1,
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if (vm.isConfirmingLocation) {
                                    LinearProgressIndicator(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(4.dp)
                                            .clip(RoundedCornerShape(2.dp)),
                                        color = Color(0xFF3B82F6),
                                    )
                                }

                                OutlinedButton(
                                    onClick = { vm.retakePhoto() },
                                    modifier = Modifier.fillMaxWidth().height(44.dp),
                                    enabled = !vm.isConfirmingLocation,
                                    shape = RoundedCornerShape(12.dp),
                                    border = BorderStroke(1.dp, Color(0xFF6B7280)),
                                ) {
                                    Icon(Icons.Default.Refresh, contentDescription = null, tint = Color(0xFF9CA3AF))
                                    Spacer(Modifier.width(6.dp))
                                    Text("都不是，重新拍照", color = Color(0xFF9CA3AF))
                                }
                            } else {
                                // Normal capture buttons
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                                ) {
                                    Button(
                                        onClick = { vm.captureAndUpload(context) },
                                        modifier = Modifier.weight(1f).height(48.dp),
                                        enabled = !vm.isUploading,
                                        colors = ButtonDefaults.buttonColors(containerColor = Gold),
                                        shape = RoundedCornerShape(12.dp),
                                    ) {
                                        if (vm.isUploading) {
                                            CircularProgressIndicator(
                                                modifier = Modifier.size(18.dp),
                                                color = Color.White,
                                                strokeWidth = 2.dp,
                                            )
                                        } else {
                                            Icon(Icons.Default.CameraAlt, contentDescription = null)
                                        }
                                        Spacer(Modifier.width(6.dp))
                                        Text(if (vm.isOfflineMode) "拍照存檔" else "拍照上傳")
                                    }

                                    OutlinedButton(
                                        onClick = {
                                            galleryLauncher.launch(
                                                PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                                            )
                                        },
                                        modifier = Modifier.height(48.dp),
                                        enabled = !vm.isUploading,
                                        shape = RoundedCornerShape(12.dp),
                                    ) {
                                        Icon(Icons.Default.PhotoLibrary, contentDescription = "相簿")
                                    }
                                }

                                if (vm.isOfflineMode && vm.photoCount > 0) {
                                    Button(
                                        onClick = { vm.finishOfflineSession() },
                                        modifier = Modifier.fillMaxWidth().height(44.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = ChartBlue),
                                        shape = RoundedCornerShape(12.dp),
                                    ) {
                                        Icon(Icons.Default.CheckCircle, contentDescription = null)
                                        Spacer(Modifier.width(8.dp))
                                        Text("結束採集", fontWeight = FontWeight.Bold)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Save map dialog
    if (showSaveDialog) {
        AlertDialog(
            onDismissRequest = { showSaveDialog = false },
            title = { Text("儲存地圖") },
            text = {
                Column {
                    Text("為這張地圖命名，下次可以直接載入使用", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                    Spacer(Modifier.height(8.dp))
                    OutlinedTextField(
                        value = mapName,
                        onValueChange = { mapName = it },
                        label = { Text("地圖名稱") },
                        placeholder = { Text("例如：3樓飲水機路線") },
                        singleLine = true,
                    )
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        if (mapName.isNotBlank()) {
                            vm.saveMap(mapName)
                            showSaveDialog = false
                            Toast.makeText(context, "地圖已儲存", Toast.LENGTH_SHORT).show()
                        }
                    },
                    enabled = mapName.isNotBlank() && !vm.isSavingMap,
                ) { Text("儲存") }
            },
            dismissButton = {
                TextButton(onClick = { showSaveDialog = false }) { Text("取消") }
            },
        )
    }
}

// ── Camera preview ──

@Composable
private fun SNavCameraPreview(lifecycleOwner: androidx.lifecycle.LifecycleOwner, imageCapture: androidx.camera.core.ImageCapture) {
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
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture,
                )
            } catch (e: Exception) {
                Log.e("SNavCamera", "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(ctx))
        previewView
    }, modifier = Modifier.fillMaxSize())
}

// ── Mini PDR trail map (with photo node markers) ──

@Composable
private fun PdrMiniTrail(
    pathPoints: List<com.example.shopping.sensor.PdrPoint>,
    currentX: Float,
    currentY: Float,
    photoNodes: List<SensorNavViewModel.PhotoNode> = emptyList(),
    modifier: Modifier = Modifier,
) {
    val pathColor = Gold
    val dotColor = GoldBright
    val bgColor = SurfaceBase
    val gridColor = Border
    val nodeColor = ChartBlue
    val labelColor = Color.White

    Surface(
        modifier = modifier,
        color = bgColor,
        shape = RoundedCornerShape(8.dp),
        shadowElevation = 1.dp,
    ) {
        Canvas(modifier = Modifier.fillMaxSize().padding(4.dp)) {
            if (pathPoints.isEmpty()) return@Canvas

            val allX = pathPoints.map { it.x } + currentX + photoNodes.map { it.x }
            val allY = pathPoints.map { it.y } + currentY + photoNodes.map { it.y }
            val minX = allX.min() - 1f
            val maxX = allX.max() + 1f
            val minY = allY.min() - 1f
            val maxY = allY.max() + 1f
            val rangeX = (maxX - minX).coerceAtLeast(2f)
            val rangeY = (maxY - minY).coerceAtLeast(2f)
            val scale = minOf(size.width / rangeX, size.height / rangeY) * 0.85f
            val offX = (size.width - rangeX * scale) / 2f
            val offY = (size.height - rangeY * scale) / 2f

            fun toScreen(px: Float, py: Float): Offset {
                return Offset(
                    offX + (px - minX) * scale,
                    offY + (rangeY - (py - minY)) * scale,
                )
            }

            // Grid
            val gridSpacing = 2f
            var gx = (minX / gridSpacing).toInt() * gridSpacing
            while (gx <= maxX) {
                val sx = offX + (gx - minX) * scale
                drawLine(gridColor, Offset(sx, 0f), Offset(sx, size.height), strokeWidth = 0.5f)
                gx += gridSpacing
            }
            var gy = (minY / gridSpacing).toInt() * gridSpacing
            while (gy <= maxY) {
                val sy = offY + (rangeY - (gy - minY)) * scale
                drawLine(gridColor, Offset(0f, sy), Offset(size.width, sy), strokeWidth = 0.5f)
                gy += gridSpacing
            }

            // Path
            for (i in 1 until pathPoints.size) {
                val p0 = toScreen(pathPoints[i - 1].x, pathPoints[i - 1].y)
                val p1 = toScreen(pathPoints[i].x, pathPoints[i].y)
                drawLine(pathColor, p0, p1, strokeWidth = 2.5f, cap = StrokeCap.Round)
            }

            // Start point
            if (pathPoints.isNotEmpty()) {
                val start = toScreen(pathPoints.first().x, pathPoints.first().y)
                drawCircle(Color(0xFF3B82F6), radius = 5f, center = start)
            }

            // Photo nodes
            val paint = android.graphics.Paint().apply {
                color = labelColor.toArgb()
                textSize = 22f
                textAlign = android.graphics.Paint.Align.CENTER
                isAntiAlias = true
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
            for (node in photoNodes) {
                val pos = toScreen(node.x, node.y)
                drawCircle(nodeColor, radius = 9f, center = pos)
                drawCircle(Color.White, radius = 6.5f, center = pos)
                drawCircle(nodeColor, radius = 5f, center = pos)
                drawContext.canvas.nativeCanvas.drawText(
                    "${node.index}", pos.x, pos.y - 14f, paint,
                )
            }

            // Current position
            val cur = toScreen(currentX, currentY)
            drawCircle(dotColor, radius = 6f, center = cur)
            drawCircle(Color.White, radius = 3.5f, center = cur)
        }
    }
}

// ── Topomap graph: nodes (photo points) connected by edges with step/distance labels ──

@Composable
private fun PdrTopoGraph(
    photoNodes: List<SensorNavViewModel.PhotoNode>,
    pathPoints: List<com.example.shopping.sensor.PdrPoint>,
    modifier: Modifier = Modifier,
) {
    if (photoNodes.isEmpty()) return

    val bgColor = SurfaceBase
    val gridColor = Border
    val edgeColor = Gold
    val nodeColor = ChartBlue
    val startColor = Color(0xFF3B82F6)

    Surface(
        modifier = modifier,
        color = bgColor,
        shape = RoundedCornerShape(12.dp),
        shadowElevation = 2.dp,
    ) {
        Canvas(modifier = Modifier.fillMaxSize().padding(12.dp)) {
            val allX = photoNodes.map { it.x } + pathPoints.map { it.x }
            val allY = photoNodes.map { it.y } + pathPoints.map { it.y }
            if (allX.isEmpty()) return@Canvas

            val minX = allX.min() - 1f
            val maxX = allX.max() + 1f
            val minY = allY.min() - 1f
            val maxY = allY.max() + 1f
            val rangeX = (maxX - minX).coerceAtLeast(2f)
            val rangeY = (maxY - minY).coerceAtLeast(2f)
            val scale = minOf(size.width / rangeX, size.height / rangeY) * 0.8f
            val offX = (size.width - rangeX * scale) / 2f
            val offY = (size.height - rangeY * scale) / 2f

            fun toScreen(px: Float, py: Float): Offset {
                return Offset(
                    offX + (px - minX) * scale,
                    offY + (rangeY - (py - minY)) * scale,
                )
            }

            // Grid
            val gridSpacing = 2f
            var gx = (minX / gridSpacing).toInt() * gridSpacing
            while (gx <= maxX) {
                val sx = offX + (gx - minX) * scale
                drawLine(gridColor, Offset(sx, 0f), Offset(sx, size.height), strokeWidth = 0.5f)
                gx += gridSpacing
            }
            var gy = (minY / gridSpacing).toInt() * gridSpacing
            while (gy <= maxY) {
                val sy = offY + (rangeY - (gy - minY)) * scale
                drawLine(gridColor, Offset(0f, sy), Offset(size.width, sy), strokeWidth = 0.5f)
                gy += gridSpacing
            }

            // Background path (thin dotted line)
            if (pathPoints.size > 1) {
                for (i in 1 until pathPoints.size) {
                    val p0 = toScreen(pathPoints[i - 1].x, pathPoints[i - 1].y)
                    val p1 = toScreen(pathPoints[i].x, pathPoints[i].y)
                    drawLine(
                        edgeColor.copy(alpha = 0.25f), p0, p1,
                        strokeWidth = 1.5f, cap = StrokeCap.Round,
                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 4f)),
                    )
                }
            }

            // Edges between consecutive photo nodes
            val edgePaint = android.graphics.Paint().apply {
                color = edgeColor.toArgb()
                textSize = 24f
                textAlign = android.graphics.Paint.Align.CENTER
                isAntiAlias = true
            }
            for (i in 1 until photoNodes.size) {
                val prev = photoNodes[i - 1]
                val curr = photoNodes[i]
                val p0 = toScreen(prev.x, prev.y)
                val p1 = toScreen(curr.x, curr.y)
                drawLine(edgeColor, p0, p1, strokeWidth = 3f, cap = StrokeCap.Round)

                val mid = Offset((p0.x + p1.x) / 2, (p0.y + p1.y) / 2)
                val label = "${curr.segmentSteps}步 ${"%.1f".format(curr.segmentDistanceM)}m"
                drawContext.canvas.nativeCanvas.drawText(label, mid.x, mid.y - 8f, edgePaint)
            }

            // Nodes
            val nodePaint = android.graphics.Paint().apply {
                color = Color.White.toArgb()
                textSize = 28f
                textAlign = android.graphics.Paint.Align.CENTER
                isAntiAlias = true
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
            val subPaint = android.graphics.Paint().apply {
                color = nodeColor.copy(alpha = 0.8f).toArgb()
                textSize = 20f
                textAlign = android.graphics.Paint.Align.CENTER
                isAntiAlias = true
            }

            // Start node (origin)
            val origin = toScreen(0f, 0f)
            drawCircle(startColor, radius = 14f, center = origin)
            drawContext.canvas.nativeCanvas.drawText("S", origin.x, origin.y + 10f, nodePaint)

            for (node in photoNodes) {
                val pos = toScreen(node.x, node.y)
                drawCircle(nodeColor, radius = 16f, center = pos)
                drawCircle(Color.White, radius = 13f, center = pos)
                drawCircle(nodeColor, radius = 11f, center = pos)
                drawContext.canvas.nativeCanvas.drawText(
                    "${node.index}", pos.x, pos.y + 10f, nodePaint,
                )
                drawContext.canvas.nativeCanvas.drawText(
                    "${node.totalSteps}步", pos.x, pos.y + 32f, subPaint,
                )
            }
        }
    }
}

// ── Backend URL dialog ──

@Composable
fun BackendUrlDialog(
    currentUrl: String,
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit,
) {
    var customIp by remember { mutableStateOf("") }
    val port = "8000"

    val presetOptions = listOf(
        "172.20.10.10" to "iPhone 熱點",
        "192.168.68.107" to "家用 Wi-Fi",
        "10.0.2.2" to "Android 模擬器",
    )

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("後端伺服器設定", fontWeight = FontWeight.Bold) },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("選擇或輸入後端 IP：", style = MaterialTheme.typography.bodySmall, color = TextSecondary)

                presetOptions.forEach { (ip, label) ->
                    val url = "http://$ip:$port/"
                    val isSelected = currentUrl == url
                    Surface(
                        color = if (isSelected) Gold.copy(alpha = 0.15f) else SurfaceBase,
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .clickable { onConfirm(url) },
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(label, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                                Text(ip, style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                            }
                            if (isSelected) {
                                Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Gold, modifier = Modifier.size(20.dp))
                            }
                        }
                    }
                }

                Spacer(Modifier.height(4.dp))
                Text("自訂 IP：", style = MaterialTheme.typography.bodySmall, color = TextSecondary)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(
                        value = customIp,
                        onValueChange = { customIp = it },
                        placeholder = { Text("192.168.x.x") },
                        modifier = Modifier.weight(1f),
                        singleLine = true,
                        colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = Gold, cursorColor = Gold),
                    )
                    Spacer(Modifier.width(8.dp))
                    Button(
                        onClick = {
                            if (customIp.isNotBlank()) {
                                onConfirm("http://$customIp:$port/")
                            }
                        },
                        enabled = customIp.isNotBlank(),
                        colors = ButtonDefaults.buttonColors(containerColor = Gold),
                        shape = RoundedCornerShape(8.dp),
                    ) {
                        Text("連線")
                    }
                }
            }
        },
        confirmButton = {},
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        },
    )
}


@Composable
fun RouteWaypointsRow(
    waypoints: List<Map<String, Any>>,
    sessionId: String,
    isSnav: Boolean = false,
) {
    val baseUrl = com.example.shopping.network.BackendConfig.currentUrl.trimEnd('/')
    val prefix = if (isSnav) "snav" else "session"
    var expandedPhotoUrl by remember { mutableStateOf<String?>(null) }
    var expandedPhotoLabel by remember { mutableStateOf("") }

    Column {
        Text(
            "路徑規劃（${waypoints.size} 站）",
            color = Color(0xFF6EE7B7),
            fontSize = 14.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 6.dp),
        )
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            waypoints.forEachIndexed { index, wp ->
                val photoId = (wp["photo_id"] as? Number)?.toInt() ?: return@forEachIndexed
                val rank = (wp["rank"] as? Number)?.toInt() ?: photoId
                val role = wp["role"] as? String ?: "waypoint"
                val region = wp["region"] as? String ?: ""
                @Suppress("UNCHECKED_CAST")
                val objects = (wp["objects"] as? List<*>)?.filterIsInstance<String>() ?: emptyList()
                val photoUrl = "$baseUrl/$prefix/$sessionId/topo_photo/$photoId"

                val borderColor = when (role) {
                    "start" -> Color(0xFF60A5FA)
                    "goal" -> Color(0xFFFBBF24)
                    else -> Color(0xFF475569)
                }
                val roleLabel = when (role) {
                    "start" -> "目前"
                    "goal" -> "目標"
                    else -> ""
                }

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.width(130.dp),
                ) {
                    Surface(
                        shape = RoundedCornerShape(8.dp),
                        color = Color(0xFF1E293B),
                        shadowElevation = 2.dp,
                        border = BorderStroke(2.dp, borderColor.copy(alpha = 0.6f)),
                        modifier = Modifier
                            .size(130.dp, 97.dp)
                            .clickable {
                                expandedPhotoUrl = photoUrl
                                expandedPhotoLabel = buildString {
                                    append(if (roleLabel.isNotEmpty()) "$roleLabel " else "")
                                    append("P$rank")
                                    if (region.isNotEmpty()) append(" - $region")
                                }
                            },
                    ) {
                        Box {
                            AsyncImage(
                                model = photoUrl,
                                contentDescription = "P$rank",
                                modifier = Modifier
                                    .fillMaxSize()
                                    .clip(RoundedCornerShape(8.dp)),
                                contentScale = ContentScale.Crop,
                            )
                            Surface(
                                color = borderColor.copy(alpha = 0.85f),
                                shape = RoundedCornerShape(bottomStart = 8.dp, topEnd = 8.dp),
                                modifier = Modifier.align(Alignment.TopEnd),
                            ) {
                                Text(
                                    if (roleLabel.isNotEmpty()) "$roleLabel P$rank" else "P$rank",
                                    color = Color.White,
                                    fontSize = 10.sp,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                                )
                            }
                            if (region.isNotEmpty()) {
                                Surface(
                                    color = Color.Black.copy(alpha = 0.6f),
                                    shape = RoundedCornerShape(topStart = 8.dp, bottomEnd = 8.dp),
                                    modifier = Modifier.align(Alignment.BottomStart),
                                ) {
                                    Text(
                                        region,
                                        color = Color.White,
                                        fontSize = 9.sp,
                                        modifier = Modifier.padding(horizontal = 5.dp, vertical = 2.dp),
                                        maxLines = 1,
                                    )
                                }
                            }
                        }
                    }
                    if (objects.isNotEmpty()) {
                        Text(
                            objects.take(3).joinToString(", "),
                            color = Color(0xFF94A3B8),
                            fontSize = 9.sp,
                            maxLines = 2,
                            modifier = Modifier.padding(top = 2.dp),
                        )
                    }
                }
                if (index < waypoints.size - 1) {
                    Text("→", color = Color(0xFF64748B), fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }

    // Full-screen photo dialog
    if (expandedPhotoUrl != null) {
        AlertDialog(
            onDismissRequest = { expandedPhotoUrl = null },
            confirmButton = {
                TextButton(onClick = { expandedPhotoUrl = null }) {
                    Text("關閉", color = Color(0xFF6EE7B7))
                }
            },
            title = { Text(expandedPhotoLabel, color = Color.White, fontSize = 16.sp, fontWeight = FontWeight.Bold) },
            text = {
                AsyncImage(
                    model = expandedPhotoUrl,
                    contentDescription = expandedPhotoLabel,
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(4f / 3f)
                        .clip(RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.Fit,
                )
            },
            containerColor = Color(0xFF1E293B),
            shape = RoundedCornerShape(16.dp),
        )
    }
}
