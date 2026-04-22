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

private val LoginBg       = Color(0xFFECFDF5)
private val LoginSurface  = Color(0xFFFFFFFF)
private val LoginAccent   = Color(0xFF059669)
private val LoginText     = Color(0xFF0F172A)
private val LoginTextMid  = Color(0xFF475569)
private val LoginTextLight= Color(0xFF94A3B8)

@Composable
fun LoginScreen(onLoginSuccess: () -> Unit) {
    val context = LocalContext.current
    val auth: FirebaseAuth = Firebase.auth

    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isPasswordVisible by remember { mutableStateOf(false) }
    var isSignUpMode by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }

    var phase by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        delay(200); phase = 1
        delay(800); phase = 2
        delay(400); phase = 3
    }

    val ease = CubicBezierEasing(0.23f, 1f, 0.32f, 1f)
    val formAlpha by animateFloatAsState(targetValue = if (phase >= 2) 1f else 0f, animationSpec = tween(600, easing = ease))

    Box(modifier = Modifier.fillMaxSize().background(LoginBg)) {
        Column(modifier = Modifier.fillMaxSize().systemBarsPadding().padding(horizontal = 32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Spacer(modifier = Modifier.weight(0.1f))

            Image(
                painter = painterResource(id = R.drawable.grocery_hero),
                contentDescription = null,
                modifier = Modifier.fillMaxWidth().height(150.dp).graphicsLayer { alpha = if(phase>=1) 1f else 0f }
            )

            Spacer(modifier = Modifier.height(24.dp))

            Column(modifier = Modifier.fillMaxWidth().graphicsLayer { alpha = formAlpha }, horizontalAlignment = Alignment.CenterHorizontally) {
                Text(text = if (isSignUpMode) "建立帳號" else "智慧購物助理", style = MaterialTheme.typography.headlineMedium.copy(fontWeight = FontWeight.Bold), color = LoginText)
                Text(text = "AI 驅動的購物體驗", style = MaterialTheme.typography.bodyMedium, color = LoginTextMid)

                Spacer(modifier = Modifier.height(32.dp))

                OutlinedTextField(
                    value = email, onValueChange = { email = it }, label = { Text("Email") },
                    leadingIcon = { Icon(Icons.Default.Email, null, tint = LoginTextMid) },
                    modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = LoginAccent, unfocusedBorderColor = Color.Transparent, focusedContainerColor = LoginSurface, unfocusedContainerColor = LoginSurface),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(12.dp))

                OutlinedTextField(
                    value = password, onValueChange = { password = it }, label = { Text("密碼") },
                    leadingIcon = { Icon(Icons.Default.Lock, null, tint = LoginTextMid) },
                    visualTransformation = if (isPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(14.dp),
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = LoginAccent, unfocusedBorderColor = Color.Transparent, focusedContainerColor = LoginSurface, unfocusedContainerColor = LoginSurface),
                    singleLine = true
                )

                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = {
                        if (email.isEmpty() || password.isEmpty()) {
                            Toast.makeText(context, "請填寫欄位", Toast.LENGTH_SHORT).show()
                            return@Button
                        }
                        isLoading = true
                        if (isSignUpMode) {
                            auth.createUserWithEmailAndPassword(email, password).addOnCompleteListener { task ->
                                isLoading = false
                                if (task.isSuccessful) {
                                    Toast.makeText(context, "註冊成功，請登入", Toast.LENGTH_SHORT).show()
                                    isSignUpMode = false
                                } else Toast.makeText(context, "失敗: ${task.exception?.message}", Toast.LENGTH_LONG).show()
                            }
                        } else {
                            auth.signInWithEmailAndPassword(email, password).addOnCompleteListener { task ->
                                isLoading = false
                                if (task.isSuccessful) onLoginSuccess()
                                else Toast.makeText(context, "登入失敗 (請確認網路): ${task.exception?.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = LoginAccent),
                    enabled = !isLoading
                ) {
                    if (isLoading) CircularProgressIndicator(color = Color.White, modifier = Modifier.size(22.dp))
                    else Text(if (isSignUpMode) "建立帳號" else "登入", fontWeight = FontWeight.Bold)
                }

                // --- GUEST / SKIP BUTTON FOR DEBUGGING ---
                Spacer(modifier = Modifier.height(12.dp))
                OutlinedButton(
                    onClick = { onLoginSuccess() },
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                    border = androidx.compose.foundation.BorderStroke(1.dp, LoginAccent.copy(0.3f))
                ) {
                    Text("測試用：直接進入 (跳過登入)", color = LoginTextMid, fontSize = 14.sp)
                }

                TextButton(onClick = { isSignUpMode = !isSignUpMode }, modifier = Modifier.padding(top = 12.dp)) {
                    Text(text = if (isSignUpMode) "已有帳號？立即登入" else "沒有帳號？立即註冊", color = LoginAccent)
                }
            }
            Spacer(modifier = Modifier.weight(0.1f))
        }
    }
}
