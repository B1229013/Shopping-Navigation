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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.shopping.BuildConfig
import com.example.shopping.model.DietRecord
import com.example.shopping.model.ShoppingContext
import com.example.shopping.model.ShoppingItem
import com.example.shopping.model.buildShoppingContext
import com.example.shopping.model.toJsonString
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import kotlinx.coroutines.launch

@Composable
fun AIScreen(
    shoppingItems: List<ShoppingItem> = emptyList(),
    dietRecords: List<DietRecord> = emptyList(),
    budgetTotal: Int = 0
) {
    val context = LocalContext.current
    val groqApiKey = BuildConfig.GROQ_API_KEY

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

    // ── Build context snapshot on each recomposition ──
    val shoppingContext = remember(shoppingItems, dietRecords, budgetTotal) {
        buildShoppingContext(shoppingItems, dietRecords, budgetTotal)
    }

    // ── Context-aware system prompt ──
    val systemPrompt = remember(shoppingContext) {
        buildSystemPrompt(shoppingContext)
    }

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
                                "您好！我是您的 AI 購物助理。\n\n" +
                                "我可以存取您的購物清單、預算和飲食紀錄，幫您：\n" +
                                "• 分析消費習慣（「這個月零食花了多少？」）\n" +
                                "• 規劃健康飲食（「幫我規劃不超標的菜單」）\n" +
                                "• 預算建議（「我還剩多少預算？」）\n" +
                                "• 交叉分析（「清單裡的東西會超過預算嗎？」）",
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
                                    // Build message list with conversation history for continuity
                                    val messages = mutableListOf(
                                        GroqMessage(role = "system", content = systemPrompt)
                                    )
                                    // Include recent history for multi-turn context
                                    chatHistory.takeLast(5).forEach { (prevUser, prevAi) ->
                                        messages.add(GroqMessage(role = "user", content = prevUser))
                                        messages.add(GroqMessage(role = "assistant", content = prevAi))
                                    }
                                    messages.add(GroqMessage(role = "user", content = userMsg))

                                    val response = groqApi.getCompletion(
                                        apiKey = "Bearer $groqApiKey",
                                        request = GroqRequest(messages = messages)
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
                    contentColor = SurfaceBase,
                    shape = CircleShape
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = "發送", modifier = Modifier.size(20.dp))
                }
            }
        }
    }
}

// ── Context-Aware System Prompt Builder ─────────────────────
// Injects the full ShoppingContext JSON so the AI can cross-reference
// inventory, budget, and health data in a single query.

private fun buildSystemPrompt(ctx: ShoppingContext): String {
    val contextJson = ctx.toJsonString()
    return """
You are an intelligent shopping assistant for a Traditional Chinese (繁體中文) grocery app.
Always respond in Traditional Chinese. Be concise and helpful.

You have FULL ACCESS to the user's real-time data across three modules:

## Your Data Access (JSON)
```json
$contextJson
```

## How to Use This Data

### Module 1: Inventory (inventory)
- `pending_items`: Items the user still needs to buy (name, qty, price, category)
- `purchased_items`: Items already bought
- Use this to answer: "What's on my list?", "What did I buy?", "Do I have eggs?"

### Module 2: Budget (budget)
- `monthly_budget`: User's spending limit
- `total_spent` / `remaining`: Current spending status
- `spending_by_category`: Breakdown by Food/Beverages/Groceries/Other
- Use this to answer: "How much did I spend on snacks?", "Am I over budget?", "What's my top spending category?"

### Module 3: Health (health)
- `diet_records_today`: What the user ate today (calories, macros)
- `total_calories_today`: Total calorie intake
- `recent_ingredients`: Recent ingredient lists for allergen awareness
- Use this to answer: "How many calories today?", "Does my list exceed sodium goals?"

### Cross-Module Queries
When the user asks complex questions, CROSS-REFERENCE modules:
- "Plan a meal within budget" → check inventory + budget.remaining + health targets
- "Is my list healthy?" → check pending_items against health.diet_records_today
- "What should I buy?" → combine budget.remaining + health gaps + inventory.pending_items

## Summary
- Date: ${ctx.summary.date}
- Budget used: ${ctx.summary.budget_utilization_percent}%
- Top spending: ${ctx.summary.top_spending_category ?: "N/A"}
- Pending items: ${ctx.summary.pending_item_names.joinToString(", ").ifEmpty { "無" }}

## Rules
1. Always base answers on ACTUAL data, never fabricate numbers.
2. When data is empty (e.g., no items), say so honestly.
3. For health advice, give practical suggestions, not medical diagnoses.
4. Reference specific item names and prices from the data.
5. Format currency as NT$ or $ consistently.
""".trimIndent()
}

// ── Chat Bubble ─────────────────────────────────────────────

@Composable
fun ChatBubble(text: String, isUser: Boolean) {
    val backgroundColor = if (isUser) Gold else SurfaceBase
    val textColor = if (isUser) SurfaceBase else TextPrimary
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
