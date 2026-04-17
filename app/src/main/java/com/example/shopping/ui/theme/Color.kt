package com.example.shopping.ui.theme

import androidx.compose.ui.graphics.Color

// ── Fresh Grocery Light Palette ───────────────────────────
// Warm mint background, emerald green accent, flat design.
// Matches the HomeScreen grocery aesthetic.

// Backgrounds — layered light surfaces
val Noir         = Color(0xFFECFDF5)   // fresh mint background
val SurfaceDim   = Color(0xFFF0F8F6)   // subtle mint surface
val SurfaceBase  = Color(0xFFFFFFFF)   // white cards/containers
val SurfaceBright= Color(0xFFF5FAF8)   // hover / active states
val SurfaceGlow  = Color(0xFFD1FAE5)   // elevated elements (green tint)

// Text — dark on light
val TextPrimary  = Color(0xFF0F172A)   // slate-900 primary
val TextSecondary= Color(0xFF475569)   // slate-600 secondary
val TextTertiary = Color(0xFF94A3B8)   // slate-400 subtle
val TextDisabled = Color(0xFFCBD5E1)   // slate-300 disabled

// Accent — emerald green (fresh grocery feel)
val Gold         = Color(0xFF059669)   // primary accent (emerald-600)
val GoldBright   = Color(0xFF10B981)   // active / hover (emerald-500)
val GoldDim      = Color(0xFF047857)   // deeper green
val GoldSurface  = Color(0xFFD1FAE5)   // green-tinted surface

// Semantic
val Danger       = Color(0xFFDC2626)   // red-600
val DangerDim    = Color(0xFFFEE2E2)   // red-100 background
val Success      = Color(0xFF059669)   // emerald-600
val SuccessDim   = Color(0xFFD1FAE5)   // emerald-100 background
val Info         = Color(0xFF2563EB)   // blue-600
val InfoDim      = Color(0xFFDBEAFE)   // blue-100 background

// Dividers & borders
val Border       = Color(0xFFE1F2ED)   // green-tinted border
val BorderBright = Color(0xFFD1FAE5)   // stronger green border

// Chart palette — four distinct hues for budget categories
val ChartBlue    = Color(0xFF2563EB)
val ChartGreen   = Color(0xFF059669)
val ChartAmber   = Color(0xFFD97706)
val ChartViolet  = Color(0xFF7C3AED)
