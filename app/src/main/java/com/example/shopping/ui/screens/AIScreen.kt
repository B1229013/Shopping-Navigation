package com.example.shopping.ui.screens

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.shopping.R
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun AIScreen() {
    val context = LocalContext.current
    val groqApiKey = stringResource(id = R.string.groq_api_key)

    val scope = rememberCoroutineScope()
    var inputText by remember { mutableStateOf("") }
    var chatHistory by remember { mutableStateOf(listOf<Pair<String, String>>()) }
    var isLoading by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    LaunchedEffect(chatHistory.size) {
        scrollState.animateScrollTo(scrollState.maxValue)
    }

    var isSendPressed by remember { mutableStateOf(false) }
    val sendScale by animateFloatAsState(
        targetValue = if (isSendPressed) 0.88f else 1f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "send_scale"
    )

    FadeInScreen {
        Column(modifier = Modifier.fillMaxSize()) {
            // ── Header ──
            StaggeredItem(index = 0) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 20.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "購物助理",
                        style = MaterialTheme.typography.displayMedium,
                        color = TextPrimary
                    )
                    Text(
                        "Powered by Groq",
                        style = MaterialTheme.typography.labelSmall,
                        color = TextDisabled
                    )
                }
            }

            // ── Chat Area ──
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .padding(horizontal = 20.dp)
                    .verticalScroll(scrollState)
            ) {
                Spacer(Modifier.height(8.dp))

                if (chatHistory.isEmpty()) {
                    StaggeredItem(index = 1) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(18.dp))
                                .background(SurfaceBase)
                                .padding(18.dp)
                        ) {
                            Text(
                                "您好！我是您的 AI 助理，由 Groq 提供支援。\n我可以幫您分析購物清單或回答相關問題。",
                                style = MaterialTheme.typography.bodyMedium,
                                color = TextSecondary,
                                lineHeight = MaterialTheme.typography.bodyMedium.lineHeight
                            )
                        }
                    }
                }

                chatHistory.forEachIndexed { index, (user, ai) ->
                    Spacer(Modifier.height(12.dp))
                    ChatBubble(text = user, isUser = true)
                    Spacer(modifier = Modifier.height(8.dp))
                    ChatBubble(text = ai, isUser = false)
                }

                if (isLoading) {
                    Spacer(Modifier.height(12.dp))
                    TypingIndicator()
                }

                Spacer(Modifier.height(16.dp))
            }

            // ── Input Area ──
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(SurfaceDim)
                    .padding(horizontal = 16.dp, vertical = 12.dp)
                    .navigationBarsPadding(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    placeholder = { Text("輸入問題...", color = TextTertiary) },
                    modifier = Modifier.weight(1f),
                    shape = RoundedCornerShape(24.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedBorderColor = Border,
                        unfocusedBorderColor = Border,
                        cursorColor = Gold,
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedContainerColor = SurfaceBase,
                        unfocusedContainerColor = SurfaceBase
                    ),
                    enabled = !isLoading,
                    singleLine = false,
                    maxLines = 4
                )
                Spacer(modifier = Modifier.width(10.dp))
                FloatingActionButton(
                    onClick = {
                        if (inputText.isNotBlank() && !isLoading) {
                            val userMsg = inputText
                            inputText = ""
                            isLoading = true
                            scope.launch {
                                try {
                                    val response = groqApi.getCompletion(
                                        apiKey = "Bearer $groqApiKey",
                                        request = GroqRequest(
                                            messages = listOf(
                                                GroqMessage(role = "system", content = "You are a helpful shopping assistant. Respond in Traditional Chinese."),
                                                GroqMessage(role = "user", content = userMsg)
                                            )
                                        )
                                    )
                                    val aiMsg = response.choices[0].message.content
                                    chatHistory = chatHistory + (userMsg to aiMsg)
                                } catch (e: Exception) {
                                    val errorMsg = e.localizedMessage ?: "發生錯誤"
                                    chatHistory = chatHistory + (userMsg to "錯誤: $errorMsg")
                                } finally {
                                    isLoading = false
                                }
                            }
                        }
                    },
                    modifier = Modifier
                        .size(48.dp)
                        .scale(sendScale),
                    containerColor = Gold,
                    contentColor = Noir,
                    shape = CircleShape
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "發送", modifier = Modifier.size(20.dp))
                }
            }
        }
    }
}

// ── Chat Bubble ─────────────────────────────────────────────

@Composable
fun ChatBubble(text: String, isUser: Boolean) {
    val backgroundColor = if (isUser) Gold else SurfaceBase
    val textColor = if (isUser) Noir else TextPrimary
    val alignment = if (isUser) Alignment.End else Alignment.Start
    val shape = if (isUser) {
        RoundedCornerShape(18.dp, 18.dp, 4.dp, 18.dp)
    } else {
        RoundedCornerShape(18.dp, 18.dp, 18.dp, 4.dp)
    }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = alignment
    ) {
        Box(
            modifier = Modifier
                .widthIn(max = 300.dp)
                .clip(shape)
                .background(backgroundColor)
                .padding(14.dp)
        ) {
            Text(
                text = text,
                style = MaterialTheme.typography.bodyMedium,
                color = textColor
            )
        }
    }
}

// ── Typing Indicator ────────────────────────────────────────

@Composable
fun TypingIndicator() {
    val infiniteTransition = rememberInfiniteTransition(label = "typing")

    Row(
        modifier = Modifier
            .clip(RoundedCornerShape(18.dp))
            .background(SurfaceBase)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(3) { i ->
            val alpha by infiniteTransition.animateFloat(
                initialValue = 0.25f,
                targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(600),
                    repeatMode = RepeatMode.Reverse,
                    initialStartOffset = StartOffset(i * 150)
                ),
                label = "dot_$i"
            )
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .graphicsLayer { this.alpha = alpha }
                    .background(Gold.copy(alpha = 0.7f), CircleShape)
            )
        }
    }
}
