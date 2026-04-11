package com.example.shopping.ui.screens

import android.widget.Toast
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Email
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.shopping.ui.theme.*
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import kotlinx.coroutines.delay

@Composable
fun LoginScreen(onLoginSuccess: () -> Unit) {
    val context = LocalContext.current
    val auth: FirebaseAuth = Firebase.auth

    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isPasswordVisible by remember { mutableStateOf(false) }
    var isSignUpMode by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }

    // ── Staggered entrance animations ──
    val itemCount = 5
    val visibilities = remember { List(itemCount) { mutableStateOf(false) } }
    LaunchedEffect(Unit) {
        visibilities.forEachIndexed { i, state ->
            delay(100L + i * 80L)
            state.value = true
        }
    }

    // Animate each element
    val alphas = visibilities.map { vis ->
        animateFloatAsState(
            targetValue = if (vis.value) 1f else 0f,
            animationSpec = tween(500, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
            label = "login_alpha"
        )
    }
    val offsets = visibilities.map { vis ->
        animateFloatAsState(
            targetValue = if (vis.value) 0f else 24f,
            animationSpec = tween(500, easing = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)),
            label = "login_offset"
        )
    }

    // Button press feedback
    var isPressed by remember { mutableStateOf(false) }
    val buttonScale by animateFloatAsState(
        targetValue = if (isPressed) 0.97f else 1f,
        animationSpec = tween(if (isPressed) 100 else 200),
        label = "btn_scale"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Noir)
    ) {
        // Subtle ambient glow at top
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(320.dp)
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Gold.copy(alpha = 0.03f),
                            Color.Transparent
                        )
                    )
                )
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // ── Title ──
            Box(modifier = Modifier.graphicsLayer {
                alpha = alphas[0].value
                translationY = offsets[0].value * density
            }) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = if (isSignUpMode) "建立帳號" else "歡迎回來",
                        style = MaterialTheme.typography.displayMedium,
                        color = TextPrimary
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = if (isSignUpMode) "請輸入您的 Email 與密碼" else "請登入以繼續使用",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TextSecondary
                    )
                }
            }

            Spacer(modifier = Modifier.height(48.dp))

            // ── Email field ──
            Box(modifier = Modifier.graphicsLayer {
                alpha = alphas[1].value
                translationY = offsets[1].value * density
            }) {
                OutlinedTextField(
                    value = email,
                    onValueChange = { email = it },
                    label = { Text("Email", color = TextTertiary) },
                    leadingIcon = {
                        Icon(Icons.Default.Email, contentDescription = null, tint = TextTertiary)
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Gold,
                        unfocusedBorderColor = Border,
                        cursorColor = Gold,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedContainerColor = SurfaceBase,
                        unfocusedContainerColor = SurfaceBase,
                        focusedLabelColor = Gold,
                        unfocusedLabelColor = TextTertiary
                    ),
                    singleLine = true
                )
            }

            Spacer(modifier = Modifier.height(14.dp))

            // ── Password field ──
            Box(modifier = Modifier.graphicsLayer {
                alpha = alphas[2].value
                translationY = offsets[2].value * density
            }) {
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    label = { Text("密碼", color = TextTertiary) },
                    leadingIcon = {
                        Icon(Icons.Default.Lock, contentDescription = null, tint = TextTertiary)
                    },
                    trailingIcon = {
                        IconButton(onClick = { isPasswordVisible = !isPasswordVisible }) {
                            Icon(
                                imageVector = if (isPasswordVisible) Icons.Default.Visibility else Icons.Default.VisibilityOff,
                                contentDescription = if (isPasswordVisible) "隱藏密碼" else "顯示密碼",
                                tint = TextTertiary
                            )
                        }
                    },
                    visualTransformation = if (isPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Gold,
                        unfocusedBorderColor = Border,
                        cursorColor = Gold,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedContainerColor = SurfaceBase,
                        unfocusedContainerColor = SurfaceBase,
                        focusedLabelColor = Gold,
                        unfocusedLabelColor = TextTertiary
                    ),
                    singleLine = true
                )
            }

            Spacer(modifier = Modifier.height(32.dp))

            // ── Login / SignUp button ──
            Box(modifier = Modifier.graphicsLayer {
                alpha = alphas[3].value
                translationY = offsets[3].value * density
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
                        .height(54.dp)
                        .scale(buttonScale),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Gold,
                        contentColor = Noir
                    ),
                    enabled = !isLoading
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            color = Noir,
                            modifier = Modifier.size(22.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text(
                            if (isSignUpMode) "註冊" else "登入",
                            style = MaterialTheme.typography.labelLarge,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // ── Toggle mode ──
            Box(modifier = Modifier.graphicsLayer {
                alpha = alphas[4].value
                translationY = offsets[4].value * density
            }) {
                TextButton(
                    onClick = { isSignUpMode = !isSignUpMode },
                    modifier = Modifier.padding(top = 16.dp)
                ) {
                    Text(
                        text = if (isSignUpMode) "已有帳號？立即登入" else "沒有帳號？立即註冊",
                        color = Gold,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}
