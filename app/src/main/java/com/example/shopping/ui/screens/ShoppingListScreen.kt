package com.example.shopping.ui.screens

import android.app.DatePickerDialog
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import com.example.shopping.model.ShoppingItem
import com.example.shopping.ui.utils.guessCategory
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun ShoppingListScreen(
    items: List<ShoppingItem>,
    onItemsUpdate: (List<ShoppingItem>) -> Unit
) {
    val context = LocalContext.current
    val sdf = SimpleDateFormat("dd/MM/yyyy", Locale.getDefault())

    var selectedDateText by remember { mutableStateOf("") }
    var newItemName by remember { mutableStateOf("") }
    var newItemQty by remember { mutableStateOf("1") }
    var newItemPrice by remember { mutableStateOf("") }

    var editingItemId by remember { mutableStateOf<String?>(null) }
    var editName by remember { mutableStateOf("") }
    var editQty by remember { mutableStateOf("") }
    var editPrice by remember { mutableStateOf("") }

    val filteredItems = remember(items, selectedDateText) {
        if (selectedDateText.isEmpty()) items
        else items.filter { it.dueDate?.let { d -> sdf.format(Date(d)) } == selectedDateText }
    }

    val availableDates = remember(items) {
        items.mapNotNull { it.dueDate }
            .map { sdf.format(Date(it)) }
            .distinct()
            .sortedDescending()
    }

    FadeInScreen {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(bottom = 80.dp, top = 8.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // ── Add Item Section ──
            item {
                StaggeredItem(index = 0) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .clip(RoundedCornerShape(18.dp))
                            .background(SurfaceBase)
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            OutlinedTextField(
                                value = newItemName,
                                onValueChange = { newItemName = it },
                                placeholder = { Text("商品名稱", color = TextTertiary) },
                                modifier = Modifier.weight(1f),
                                shape = RoundedCornerShape(14.dp),
                                colors = cinemaTextFieldColors(),
                                singleLine = true
                            )
                            OutlinedTextField(
                                value = newItemPrice,
                                onValueChange = { newItemPrice = it.filter { c -> c.isDigit() } },
                                placeholder = { Text("$ 價格", color = TextTertiary) },
                                modifier = Modifier.width(90.dp),
                                shape = RoundedCornerShape(14.dp),
                                colors = cinemaTextFieldColors(),
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                singleLine = true
                            )
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Text("數量", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                                Spacer(Modifier.width(8.dp))
                                IconButton(
                                    onClick = {
                                        val q = newItemQty.toIntOrNull() ?: 1
                                        if (q > 1) newItemQty = (q - 1).toString()
                                    },
                                    modifier = Modifier.size(32.dp)
                                ) {
                                    Icon(Icons.Default.Remove, null, tint = TextSecondary, modifier = Modifier.size(16.dp))
                                }
                                Text(
                                    newItemQty,
                                    style = MaterialTheme.typography.titleMedium,
                                    color = TextPrimary
                                )
                                IconButton(
                                    onClick = {
                                        val q = newItemQty.toIntOrNull() ?: 1
                                        newItemQty = (q + 1).toString()
                                    },
                                    modifier = Modifier.size(32.dp)
                                ) {
                                    Icon(Icons.Default.Add, null, tint = TextSecondary, modifier = Modifier.size(16.dp))
                                }
                            }
                            Button(
                                onClick = {
                                    if (newItemName.isNotBlank()) {
                                        val dueDateLong = if (selectedDateText.isNotEmpty()) {
                                            try { sdf.parse(selectedDateText)?.time } catch(e: Exception) { null }
                                        } else null

                                        onItemsUpdate(items + ShoppingItem(
                                            name = newItemName,
                                            qty = newItemQty.toIntOrNull() ?: 1,
                                            price = newItemPrice.toIntOrNull() ?: 0,
                                            dueDate = dueDateLong,
                                            location = guessCategory(newItemName)
                                        ))
                                        newItemName = ""; newItemPrice = ""; newItemQty = "1"
                                    }
                                },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = Gold,
                                    contentColor = Noir
                                ),
                                shape = RoundedCornerShape(12.dp),
                                contentPadding = PaddingValues(horizontal = 22.dp, vertical = 10.dp)
                            ) {
                                Text("加入", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                }
            }

            // ── Date Filter Section ──
            item {
                StaggeredItem(index = 1) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .clip(RoundedCornerShape(18.dp))
                            .background(SurfaceBase)
                            .padding(14.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("截止日期", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                            if (selectedDateText.isNotEmpty()) {
                                Text(
                                    "顯示全部",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = Gold,
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(Gold.copy(alpha = 0.08f))
                                        .clickable { selectedDateText = "" }
                                        .padding(horizontal = 10.dp, vertical = 4.dp)
                                )
                            }
                        }
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = if (selectedDateText.isEmpty()) "設置清單期限..." else selectedDateText,
                            onValueChange = {},
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(48.dp)
                                .clickable {
                                    val cal = Calendar.getInstance()
                                    DatePickerDialog(context, { _, y, m, d ->
                                        val c = Calendar.getInstance().apply { set(y, m, d) }
                                        selectedDateText = sdf.format(c.time)
                                    }, cal.get(Calendar.YEAR), cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_MONTH)).show()
                                },
                            readOnly = true,
                            trailingIcon = {
                                Icon(Icons.Default.CalendarToday, null, modifier = Modifier.size(18.dp), tint = Gold)
                            },
                            shape = RoundedCornerShape(12.dp),
                            colors = cinemaTextFieldColors(),
                            textStyle = MaterialTheme.typography.bodyMedium
                        )

                        if (availableDates.isNotEmpty()) {
                            Spacer(Modifier.height(10.dp))
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                items(availableDates) { dateStr ->
                                    val itemCount = items.count { it.dueDate?.let { d -> sdf.format(Date(d)) } == dateStr }
                                    DateChip(
                                        dateStr.substring(0, 5),
                                        "$itemCount 項",
                                        selected = selectedDateText == dateStr
                                    ) { selectedDateText = dateStr }
                                }
                            }
                        }
                    }
                }
            }

            // ── Progress bar ──
            if (filteredItems.isNotEmpty()) {
                item {
                    StaggeredItem(index = 2) {
                        Column(modifier = Modifier.padding(horizontal = 20.dp)) {
                            val completed = filteredItems.count { it.isChecked }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    "購物項目",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = TextPrimary
                                )
                                Text(
                                    "$completed/${filteredItems.size} 已購買",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = TextTertiary
                                )
                            }
                            Spacer(Modifier.height(8.dp))
                            LinearProgressIndicator(
                                progress = { completed.toFloat() / filteredItems.size },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(3.dp)
                                    .clip(RoundedCornerShape(2.dp)),
                                color = Gold,
                                trackColor = SurfaceBright
                            )
                        }
                    }
                }
            }

            // ── Shopping Items ──
            itemsIndexed(filteredItems) { index, item ->
                StaggeredItem(index = index + 3, delayPerItem = 35L) {
                    Box(modifier = Modifier.padding(horizontal = 20.dp)) {
                        if (editingItemId == item.id) {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(18.dp))
                                    .background(SurfaceBright)
                                    .padding(14.dp),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                OutlinedTextField(
                                    value = editName,
                                    onValueChange = { editName = it },
                                    label = { Text("商品名稱", color = TextTertiary) },
                                    modifier = Modifier.fillMaxWidth(),
                                    colors = cinemaTextFieldColors(),
                                    shape = RoundedCornerShape(12.dp)
                                )
                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    OutlinedTextField(
                                        value = editQty,
                                        onValueChange = { editQty = it },
                                        label = { Text("數量", color = TextTertiary) },
                                        modifier = Modifier.weight(1f),
                                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                        colors = cinemaTextFieldColors(),
                                        shape = RoundedCornerShape(12.dp)
                                    )
                                    OutlinedTextField(
                                        value = editPrice,
                                        onValueChange = { editPrice = it.filter { c -> c.isDigit() } },
                                        label = { Text("價格", color = TextTertiary) },
                                        modifier = Modifier.weight(1f),
                                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                        colors = cinemaTextFieldColors(),
                                        shape = RoundedCornerShape(12.dp)
                                    )
                                }
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    TextButton(onClick = { editingItemId = null }) {
                                        Text("取消", color = TextSecondary)
                                    }
                                    Spacer(Modifier.width(8.dp))
                                    Button(
                                        onClick = {
                                            onItemsUpdate(items.map {
                                                if (it.id == item.id) it.copy(
                                                    name = editName,
                                                    qty = editQty.toIntOrNull() ?: 1,
                                                    price = editPrice.toIntOrNull() ?: 0
                                                ) else it
                                            })
                                            editingItemId = null
                                        },
                                        colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Noir),
                                        shape = RoundedCornerShape(12.dp)
                                    ) { Text("儲存", fontWeight = FontWeight.SemiBold) }
                                }
                            }
                        } else {
                            ShoppingItemCard(
                                item = item,
                                onToggle = {
                                    onItemsUpdate(items.map {
                                        if (it.id == item.id) {
                                            val checked = !it.isChecked
                                            it.copy(isChecked = checked, purchasedAt = if (checked) System.currentTimeMillis() else null)
                                        } else it
                                    })
                                },
                                onDelete = { onItemsUpdate(items.filter { it.id != item.id }) },
                                onEdit = {
                                    editingItemId = item.id
                                    editName = item.name
                                    editQty = item.qty.toString()
                                    editPrice = item.price.toString()
                                }
                            )
                        }
                    }
                }
            }
        }
    }
}

// ── Date Chip ───────────────────────────────────────────────

@Composable
fun DateChip(date: String, count: String, selected: Boolean, onClick: () -> Unit) {
    val bgColor by animateColorAsState(
        targetValue = if (selected) Gold.copy(alpha = 0.12f) else SurfaceBright,
        animationSpec = tween(250),
        label = "chip_bg"
    )
    val textColor by animateColorAsState(
        targetValue = if (selected) Gold else TextSecondary,
        animationSpec = tween(250),
        label = "chip_text"
    )

    Column(
        modifier = Modifier
            .width(64.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(bgColor)
            .clickable { onClick() }
            .padding(vertical = 8.dp, horizontal = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            date,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
            color = textColor
        )
        Text(
            count,
            style = MaterialTheme.typography.labelSmall,
            color = textColor.copy(alpha = 0.7f)
        )
    }
}

// ── Shopping Item Card ──────────────────────────────────────

@Composable
fun ShoppingItemCard(
    item: ShoppingItem,
    onToggle: () -> Unit,
    onDelete: () -> Unit,
    onEdit: () -> Unit
) {
    val checkScale by animateFloatAsState(
        targetValue = if (item.isChecked) 1f else 0f,
        animationSpec = spring(dampingRatio = Spring.DampingRatioMediumBouncy),
        label = "check_scale"
    )
    val itemAlpha by animateFloatAsState(
        targetValue = if (item.isChecked) 0.45f else 1f,
        animationSpec = tween(250),
        label = "item_alpha"
    )

    PressableSurface(
        onClick = onEdit,
        backgroundColor = if (item.isChecked) SurfaceDim else SurfaceBase,
        glowColor = if (item.isChecked) Color.Transparent else Gold.copy(alpha = 0.04f)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp)
                .graphicsLayer { alpha = itemAlpha },
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(24.dp)
                    .clip(RoundedCornerShape(7.dp))
                    .background(if (item.isChecked) Gold else Border)
                    .then(
                        if (!item.isChecked) Modifier.padding(1.5.dp)
                            .clip(RoundedCornerShape(6.dp))
                            .background(SurfaceBase)
                        else Modifier
                    )
                    .clickable { onToggle() },
                contentAlignment = Alignment.Center
            ) {
                if (item.isChecked) {
                    Icon(
                        Icons.Default.Check,
                        contentDescription = null,
                        tint = Noir,
                        modifier = Modifier
                            .size(14.dp)
                            .scale(checkScale)
                    )
                }
            }

            Spacer(modifier = Modifier.width(14.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    item.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = if (item.isChecked) TextTertiary else TextPrimary,
                    textDecoration = if (item.isChecked) TextDecoration.LineThrough else TextDecoration.None
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "$${item.price} x ${item.qty}",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }

            IconButton(
                onClick = onDelete,
                modifier = Modifier.size(36.dp)
            ) {
                Icon(
                    Icons.Default.Close,
                    null,
                    tint = TextDisabled,
                    modifier = Modifier.size(16.dp)
                )
            }
        }
    }
}

// ── Shared TextField Colors ─────────────────────────────────

@Composable
fun cinemaTextFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = Gold,
    unfocusedBorderColor = Border,
    cursorColor = Gold,
    focusedTextColor = TextPrimary,
    unfocusedTextColor = TextPrimary,
    focusedContainerColor = SurfaceBright,
    unfocusedContainerColor = SurfaceBright,
    focusedLabelColor = Gold,
    unfocusedLabelColor = TextTertiary,
    focusedPlaceholderColor = TextTertiary,
    unfocusedPlaceholderColor = TextTertiary
)
