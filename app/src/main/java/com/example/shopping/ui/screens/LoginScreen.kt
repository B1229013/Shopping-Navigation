package com.example.shopping.ui.screens

import android.widget.Toast
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.shopping.R
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.ktx.Firebase
import com.google.firebase.auth.ktx.auth
import kotlinx.coroutines.delay

// ── Login light palette (matches HomeScreen) ───────────────
private val LoginBg       = Color(0xFFECFDF5)
private val LoginSurface  = Color(0xFFFFFFFF)
private val LoginAccent   = Color(0xFF059669)
private val LoginAccentBg = Color(0xFFD1FAE5)
private val LoginText     = Color(0xFF0F172A)
private val LoginTextMid  = Color(0xFF475569)
private val LoginTextLight= Color(0xFF94A3B8)
private val LoginBorder   = Color(0xFFE1F2ED)

@Composable
fun LoginScreen(onLoginSuccess: () -> Unit) {
    val context = LocalContext.current
    val auth: FirebaseAuth = Firebase.auth

    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isPasswordVisible by remember { mutableStateOf(false) }
    var isSignUpMode by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }

    // ── Animation choreography ──
    // Phase 0: blank
    // Phase 1: Hero illustration fades in large (fills top half)
    // Phase 2: Illustration zooms out + moves up, form area reveals below
    // Phase 3-8: Form elements stagger in
    var phase by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        delay(300);  phase = 1  // illustration appears big
        delay(1200); phase = 2  // illustration shrinks, form space opens
        delay(500);  phase = 3  // title
        delay(150);  phase = 4  // subtitle
        delay(120);  phase = 5  // email
        delay(120);  phase = 6  // password
        delay(100);  phase = 7  // button
        delay(80);   phase = 8  // toggle
    }

    val ease = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)
    val revealSpec: AnimationSpec<Float> = tween(400, easing = ease)

    // Hero illustration: starts large, zooms out
    val heroScale by animateFloatAsState(
        targetValue = when {
            phase >= 2 -> 0.55f   // shrunk, sitting at top
            phase >= 1 -> 1.1f    // big entrance
            else -> 1.4f          // offscreen-large
        },
        animationSpec = tween(
            durationMillis = if (phase >= 2) 700 else 600,
            easing = CubicBezierEasing(0.16f, 1f, 0.3f, 1f)
        ),
        label = "hero_scale"
    )
    val heroAlpha by animateFloatAsState(
        targetValue = if (phase >= 1) 1f else 0f,
        animationSpec = tween(500, easing = ease),
        label = "hero_alpha"
    )
    val heroOffsetY by animateFloatAsState(
        targetValue = when {
            phase >= 2 -> -40f    // moves up
            phase >= 1 -> 0f
            else -> 60f           // starts below
        },
        animationSpec = tween(700, easing = ease),
        label = "hero_y"
    )

    // Form reveal
    val formAlpha by animateFloatAsState(
        targetValue = if (phase >= 3) 1f else 0f,
        animationSpec = tween(400, easing = ease),
        label = "form_alpha"
    )

    // Individual field stagger
    val titleAlpha by animateFloatAsState(if (phase >= 3) 1f else 0f, revealSpec, label = "a3")
    val titleOffset by animateFloatAsState(if (phase >= 3) 0f else 24f, revealSpec, label = "o3")
    val subAlpha by animateFloatAsState(if (phase >= 4) 1f else 0f, revealSpec, label = "a4")
    val subOffset by animateFloatAsState(if (phase >= 4) 0f else 24f, revealSpec, label = "o4")
    val emailAlpha by animateFloatAsState(if (phase >= 5) 1f else 0f, revealSpec, label = "a5")
    val emailOffset by animateFloatAsState(if (phase >= 5) 0f else 24f, revealSpec, label = "o5")
    val passAlpha by animateFloatAsState(if (phase >= 6) 1f else 0f, revealSpec, label = "a6")
    val passOffset by animateFloatAsState(if (phase >= 6) 0f else 24f, revealSpec, label = "o6")
    val btnAlpha by animateFloatAsState(if (phase >= 7) 1f else 0f, revealSpec, label = "a7")
    val btnOffset by animateFloatAsState(if (phase >= 7) 0f else 24f, revealSpec, label = "o7")
    val toggleAlpha by animateFloatAsState(if (phase >= 8) 1f else 0f, revealSpec, label = "a8")
    val toggleOffset by animateFloatAsState(if (phase >= 8) 0f else 24f, revealSpec, label = "o8")

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(LoginBg)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .systemBarsPadding()
                .padding(horizontal = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.weight(0.05f))

            // ── Hero Illustration ──
            // Shows the grocery cart image, starts big then zooms out
            Image(
                painter = painterResource(id = R.drawable.grocery_hero),
                contentDescription = "智慧購物助理",
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(220.dp)
                    .graphicsLayer {
                        scaleX = heroScale
                        scaleY = heroScale
                        alpha = heroAlpha
                        translationY = heroOffsetY * density
                    }
            )

            Spacer(modifier = Modifier.height(8.dp))

            // ── Form Area (reveals after illustration zooms out) ──
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .graphicsLayer { alpha = formAlpha },
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Title
                Text(
                    text = if (isSignUpMode) "建立帳號" else "智慧購物助理",
                    style = MaterialTheme.typography.headlineMedium.copy(
                        fontWeight = FontWeight.Bold,
                        letterSpacing = (-0.3).sp
                    ),
                    color = LoginText,
                    modifier = Modifier.graphicsLayer {
                        alpha = titleAlpha
                        translationY = titleOffset * density
                    }
                )

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = if (isSignUpMode) "請輸入您的 Email 與密碼" else "AI 驅動的購物體驗",
                    style = MaterialTheme.typography.bodyMedium,
                    color = LoginTextMid,
                    modifier = Modifier.graphicsLayer {
                        alpha = subAlpha
                        translationY = subOffset * density
                    }
                )

                Spacer(modifier = Modifier.height(32.dp))

                // Email
                Box(modifier = Modifier.graphicsLayer {
                    alpha = emailAlpha
                    translationY = emailOffset * density
                }) {
                    OutlinedTextField(
                        value = email,
                        onValueChange = { email = it },
                        label = { Text("Email", color = LoginTextLight) },
                        leadingIcon = {
                            Icon(Icons.Default.Email, contentDescription = "Email", tint = LoginTextMid)
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = LoginAccent,
                            unfocusedBorderColor = Color.Transparent,
                            cursorColor = LoginAccent,
                            focusedTextColor = LoginText,
                            unfocusedTextColor = LoginText,
                            focusedContainerColor = LoginSurface,
                            unfocusedContainerColor = LoginSurface,
                            focusedLabelColor = LoginAccent,
                            unfocusedLabelColor = LoginTextLight
                        ),
                        singleLine = true
                    )
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Password
                Box(modifier = Modifier.graphicsLayer {
                    alpha = passAlpha
                    translationY = passOffset * density
                }) {
                    OutlinedTextField(
                        value = password,
                        onValueChange = { password = it },
                        label = { Text("密碼", color = LoginTextLight) },
                        leadingIcon = {
                            Icon(Icons.Default.Lock, contentDescription = "密碼", tint = LoginTextMid)
                        },
                        trailingIcon = {
                            IconButton(onClick = { isPasswordVisible = !isPasswordVisible }) {
                                Icon(
                                    imageVector = if (isPasswordVisible) Icons.Default.Visibility else Icons.Default.VisibilityOff,
                                    contentDescription = if (isPasswordVisible) "隱藏密碼" else "顯示密碼",
                                    tint = LoginTextMid
                                )
                            }
                        },
                        visualTransformation = if (isPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(56.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = LoginAccent,
                            unfocusedBorderColor = Color.Transparent,
                            cursorColor = LoginAccent,
                            focusedTextColor = LoginText,
                            unfocusedTextColor = LoginText,
                            focusedContainerColor = LoginSurface,
                            unfocusedContainerColor = LoginSurface,
                            focusedLabelColor = LoginAccent,
                            unfocusedLabelColor = LoginTextLight
                        ),
                        singleLine = true
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Button
                Box(modifier = Modifier.graphicsLayer {
                    alpha = btnAlpha
                    translationY = btnOffset * density
                }) {
                    Button(
                        onClick = {
                            if (email.isEmpty() || password.isEmpty()) {
                                Toast.makeText(context, "請填寫所有欄位", Toast.LENGTH_SHORT).show()
                                return@Button
                            }
                            isLoading = true
                            if (isSignUpMode) {
                                auth.createUserWithEmailAndPassword(email, password)
                                    .addOnCompleteListener { task ->
                                        if (task.isSuccessful) {
                                            auth.currentUser?.sendEmailVerification()
                                                ?.addOnCompleteListener { verifyTask ->
                                                    isLoading = false
                                                    if (verifyTask.isSuccessful) {
                                                        Toast.makeText(context, "驗證信已寄出，請至信箱查收", Toast.LENGTH_LONG).show()
                                                        isSignUpMode = false
                                                    } else {
                                                        Toast.makeText(context, "寄送驗證信失敗: ${verifyTask.exception?.message}", Toast.LENGTH_SHORT).show()
                                                    }
                                                }
                                        } else {
                                            isLoading = false
                                            Toast.makeText(context, "註冊失敗: ${task.exception?.message}", Toast.LENGTH_SHORT).show()
                                        }
                                    }
                            } else {
                                auth.signInWithEmailAndPassword(email, password)
                                    .addOnCompleteListener { task ->
                                        if (task.isSuccessful) {
                                            val user = auth.currentUser
                                            if (user != null && user.isEmailVerified) {
                                                isLoading = false
                                                onLoginSuccess()
                                            } else {
                                                isLoading = false
                                                Toast.makeText(context, "請先驗證您的 Email", Toast.LENGTH_LONG).show()
                                                auth.signOut()
                                            }
                                        } else {
                                            isLoading = false
                                            Toast.makeText(context, "登入失敗: ${task.exception?.message}", Toast.LENGTH_SHORT).show()
                                        }
                                    }
                            }
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = LoginAccent,
                            contentColor = Color.White
                        ),
                        enabled = !isLoading
                    ) {
                        if (isLoading) {
                            CircularProgressIndicator(
                                color = Color.White,
                                modifier = Modifier.size(22.dp),
                                strokeWidth = 2.dp
                            )
                        } else {
                            Text(
                                if (isSignUpMode) "建立帳號" else "登入",
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                // Toggle
                TextButton(
                    onClick = { isSignUpMode = !isSignUpMode },
                    modifier = Modifier
                        .padding(top = 12.dp)
                        .graphicsLayer {
                            alpha = toggleAlpha
                            translationY = toggleOffset * density
                        }
                ) {
                    Text(
                        text = if (isSignUpMode) "已有帳號？立即登入" else "沒有帳號？立即註冊",
                        color = LoginAccent,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }

            Spacer(modifier = Modifier.weight(0.1f))

            // Footer
            Text(
                text = "Powered by AI",
                style = MaterialTheme.typography.labelSmall.copy(letterSpacing = 1.5.sp),
                color = LoginTextLight,
                textAlign = TextAlign.Center,
                modifier = Modifier
                    .padding(bottom = 16.dp)
                    .graphicsLayer { alpha = formAlpha * 0.5f }
            )
        }
    }
}
