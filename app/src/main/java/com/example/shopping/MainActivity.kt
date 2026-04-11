package com.example.shopping

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.*
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.shopping.ui.screens.*
import com.example.shopping.ui.theme.*
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase

private val CinemaEasing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)

class MainActivity : ComponentActivity() {
    private lateinit var auth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        auth = Firebase.auth

        enableEdgeToEdge()
        setContent {
            ShoppingTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Noir
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
                    }
                }
            }
        }
    }
}
