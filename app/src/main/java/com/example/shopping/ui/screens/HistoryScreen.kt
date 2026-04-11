package com.example.shopping.ui.screens

import android.app.DatePickerDialog
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CalendarMonth
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.shopping.model.ShoppingItem
import com.example.shopping.ui.components.*
import com.example.shopping.ui.theme.*
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun HistoryScreen(shoppingItems: List<ShoppingItem>) {
    val context = LocalContext.current
    val sdfDate = SimpleDateFormat("dd/MM/yyyy", Locale.getDefault())

    var selectedFilterDate by remember { mutableStateOf("") }

    val completedItems = shoppingItems.filter { it.isChecked }

    val filteredHistory = remember(completedItems, selectedFilterDate) {
        if (selectedFilterDate.isEmpty()) completedItems
        else completedItems.filter {
            it.purchasedAt?.let { pAt -> sdfDate.format(Date(pAt)) } == selectedFilterDate
        }
    }.reversed()

    FadeInScreen {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(top = 8.dp, bottom = 80.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // ── Header ──
            item {
                StaggeredItem(index = 0) {
                    Text(
                        "歷史紀錄",
                        style = MaterialTheme.typography.displayMedium,
                        color = TextPrimary,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp)
                    )
                }
            }

            // ── Date Filter ──
            item {
                StaggeredItem(index = 1) {
                    OutlinedButton(
                        onClick = {
                            val cal = Calendar.getInstance()
                            DatePickerDialog(context, { _, y, m, d ->
                                val c = Calendar.getInstance().apply { set(y, m, d) }
                                selectedFilterDate = sdfDate.format(c.time)
                            }, cal.get(Calendar.YEAR), cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_MONTH)).show()
                        },
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp)
                            .height(48.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.outlinedButtonColors(contentColor = TextSecondary),
                        border = androidx.compose.foundation.BorderStroke(1.dp, Border)
                    ) {
                        Icon(Icons.Default.CalendarMonth, null, tint = Gold, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(10.dp))
                        Text(
                            if (selectedFilterDate.isEmpty()) "篩選特定日期" else "篩選: $selectedFilterDate",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }

            // ── Clear filter ──
            if (selectedFilterDate.isNotEmpty()) {
                item {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 20.dp),
                        horizontalArrangement = Arrangement.End
                    ) {
                        TextButton(onClick = { selectedFilterDate = "" }) {
                            Icon(Icons.Default.Close, null, tint = TextTertiary, modifier = Modifier.size(14.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("清除篩選", style = MaterialTheme.typography.bodySmall, color = TextTertiary)
                        }
                    }
                }
            }

            // ── Section Title ──
            item {
                StaggeredItem(index = 2) {
                    Text(
                        "已購買清單",
                        style = MaterialTheme.typography.titleMedium,
                        color = TextSecondary,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp)
                    )
                }
            }

            // ── Items or Empty State ──
            if (filteredHistory.isEmpty()) {
                item {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            "尚無符合條件的購買紀錄",
                            style = MaterialTheme.typography.bodyMedium,
                            color = TextTertiary
                        )
                    }
                }
            } else {
                itemsIndexed(filteredHistory) { index, item ->
                    StaggeredItem(index = index + 3, delayPerItem = 35L) {
                        HistoryItemCard(item)
                    }
                }
            }
        }
    }
}

@Composable
fun HistoryItemCard(item: ShoppingItem) {
    PressableSurface(
        onClick = {},
        modifier = Modifier.padding(horizontal = 20.dp),
        backgroundColor = SurfaceBase
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Time indicator
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .width(48.dp)
                    .clip(RoundedCornerShape(8.dp))
                    .background(SurfaceBright)
                    .padding(vertical = 8.dp, horizontal = 4.dp)
            ) {
                val dateStr = item.purchasedAt?.let {
                    SimpleDateFormat("MM/dd", Locale.getDefault()).format(Date(it))
                } ?: "--"
                val timeStr = item.purchasedAt?.let {
                    SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(it))
                } ?: "--"
                Text(dateStr, style = MaterialTheme.typography.labelSmall, color = TextSecondary)
                Text(timeStr, style = MaterialTheme.typography.labelSmall, color = TextTertiary)
            }

            Spacer(Modifier.width(14.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    item.name,
                    style = MaterialTheme.typography.titleMedium,
                    color = TextPrimary
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "x${item.qty}",
                    style = MaterialTheme.typography.bodySmall,
                    color = TextTertiary
                )
            }

            Text(
                "$${item.price * item.qty}",
                style = MaterialTheme.typography.titleMedium,
                color = Gold
            )
        }
    }
}
