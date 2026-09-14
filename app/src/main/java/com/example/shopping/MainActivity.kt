package com.example.shopping

import android.os.Bundle
import android.view.KeyEvent
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.*
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.shopping.ui.screens.*
import com.example.shopping.ui.theme.*
import com.google.android.libraries.places.api.Places
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase

val LocalVolumeKeyHandler = staticCompositionLocalOf<VolumeKeyHandler> { VolumeKeyHandler() }

class VolumeKeyHandler {
    var onVolumeDown: (() -> Unit)? = null
    var onVolumeUp: (() -> Unit)? = null
}

private val CinemaEasing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)

class MainActivity : ComponentActivity() {
    private lateinit var auth: FirebaseAuth
    private val volumeKeyHandler = VolumeKeyHandler()

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_DOWN -> {
                volumeKeyHandler.onVolumeDown?.let { it(); return true }
            }
            KeyEvent.KEYCODE_VOLUME_UP -> {
                volumeKeyHandler.onVolumeUp?.let { it(); return true }
            }
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        auth = Firebase.auth
        com.example.shopping.network.BackendConfig.init(this)

        if (!Places.isInitialized() && BuildConfig.MAPS_API_KEY.isNotBlank()) {
            Places.initializeWithNewPlacesApiEnabled(applicationContext, BuildConfig.MAPS_API_KEY)
        }

        enableEdgeToEdge()
        setContent {
            CompositionLocalProvider(LocalVolumeKeyHandler provides volumeKeyHandler) {
            ShoppingTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = androidx.compose.ui.graphics.Color(0xFFECFDF5)
                ) {
                    val navController = rememberNavController()
                    val startDestination = "login"

                    NavHost(
                        navController = navController,
                        startDestination = startDestination,
                        enterTransition = {
                            fadeIn(animationSpec = tween(300, easing = CinemaEasing)) +
                            slideInVertically(
                                initialOffsetY = { it / 16 },
                                animationSpec = tween(300, easing = CinemaEasing)
                            )
                        },
                        exitTransition = {
                            fadeOut(animationSpec = tween(200, easing = CinemaEasing))
                        },
                        popEnterTransition = {
                            fadeIn(animationSpec = tween(300, easing = CinemaEasing))
                        },
                        popExitTransition = {
                            fadeOut(animationSpec = tween(200, easing = CinemaEasing)) +
                            slideOutVertically(
                                targetOffsetY = { it / 16 },
                                animationSpec = tween(200, easing = CinemaEasing)
                            )
                        }
                    ) {
                        composable("login") {
                            LoginScreen(onLoginSuccess = {
                                navController.navigate("main_list") {
                                    popUpTo("login") { inclusive = true }
                                }
                            })
                        }
                        composable("main_list") {
                            MainContainer(navController)
                        }
                        composable("teammate_home") {
                            TeammateHomeScreen(navController)
                        }
                        composable("ar_navigation") {
                            NavigationScreen(navController)
                        }
                        composable("settings") {
                            SettingsScreen(navController)
                        }
                        composable("sensor_lab") {
                            SensorLabScreen(navController)
                        }
                        composable("sensor_nav") {
                            SensorNavHomeScreen(navController)
                        }
                        composable("sensor_nav_active") {
                            SensorNavActiveScreen(navController)
                        }
                    }
                }
            }
            }
        }
    }
}
