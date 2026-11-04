package com.example.shopping.navigation

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** A point in metres. */
data class Vec2(val x: Double, val y: Double) {
    operator fun minus(o: Vec2) = Vec2(x - o.x, y - o.y)
    operator fun plus(o: Vec2) = Vec2(x + o.x, y + o.y)
    operator fun times(s: Double) = Vec2(x * s, y * s)
    val length: Double get() = sqrt(x * x + y * y)
}

/**
 * The user's position on the store map, kept by dead reckoning and corrected by photo fixes.
 * Port of the iOS MapPositionEstimator (APPNAV_ios/Navigation/MapPositionEstimator.swift).
 *
 * Frames: raw PDR (PdrTracker) starts at (0,0) wherever tracking began; the map frame is
 * the Neo4j reference map's node coordinates (metres, +y = heading 0°, clockwise).
 * Until the first confident photo fix (or "I'm at the entrance") the map position is
 * unknown (`mapPosition == null`).
 *
 * A photo fix corrects:
 * - **position** — snapped to the localized node (cancels accumulated drift);
 * - **heading offset** — over a straight stretch ≥ [Config.minCalibrationDistanceM] between
 *   fixes A and B, the direction actually walked (node A → node B) vs the direction PDR
 *   thought it walked; fixes closer together extend the stretch instead. Within
 *   [Config.maxHeadingCorrectionDeg] it is smoothed in; beyond it, two stretches in a row
 *   must agree before the offset is replaced. [alignHeading] sets it from the user
 *   ("facing into the store"); before either, [setMapNorth] gives a rough first offset;
 * - **step scale** — map distance A→B vs PDR distance over the same stretch, clamped.
 * Both are smoothed with [Config.learningRate] and reset only with the session.
 *
 * Fixes below minConfidence (minFirstFixConfidence before any position), or implausible
 * jumps (> maxJumpM from the dead-reckoned position when the photo was taken, while
 * confidence < jumpOverrideConfidence) are rejected. Pure logic — unit tested.
 */
class MapPositionEstimator(val config: Config = Config()) {

    data class Config(
        val minConfidence: Double = 0.6,
        /** The first fix has no sensor position to agree with, so the server's confidence is image-only. */
        val minFirstFixConfidence: Double = 0.3,
        val maxJumpM: Double = 15.0,
        val jumpOverrideConfidence: Double = 0.85,
        val minCalibrationDistanceM: Double = 8.0,
        val learningRate: Double = 0.3,
        val maxHeadingCorrectionDeg: Double = 45.0,
        /** Two straight stretches in a row that both disagree by more than [maxHeadingCorrectionDeg]
         *  and agree with each other within this replace the offset outright. */
        val largeCorrectionAgreementDeg: Double = 30.0,
        val stepScaleMin: Double = 0.7,
        val stepScaleMax: Double = 1.3,
    )

    sealed class FixResult {
        object Accepted : FixResult()
        object RejectedLowConfidence : FixResult()
        data class RejectedJump(val distanceM: Double) : FixResult()
    }

    var mapPosition: Vec2? = null; private set
    /** Added to the raw PDR heading (degrees, clockwise). */
    var headingOffsetDeg = 0.0; private set
    /** Multiplies raw PDR distances. */
    var stepScale = 1.0; private set
    /** Map metres walked since the last accepted fix (backend: σ = σ0 + drift_rate · distance). */
    var distanceSinceFixM = 0.0; private set
    var acceptedFixes = 0; private set

    private var lastRaw: Vec2? = null
    private var baseMap: Vec2? = null
    private var baseRaw: Vec2? = null
    private var rawPathSinceBase = 0.0
    /** A heading offset observed far from the current one, waiting for a second stretch to confirm it. */
    private var pendingLargeOffsetDeg: Double? = null
    /** Compass bearing of the map's +Y axis from the venue config ([setMapNorth]), or null. */
    private var mapNorthBearingDeg: Double? = null
    /** The offset came from the user ([alignHeading]) or a walked stretch: map north no longer applies. */
    var headingAligned = false; private set

    val isCalibrated: Boolean get() = mapPosition != null

    fun reset() {
        mapPosition = null
        headingOffsetDeg = 0.0
        stepScale = 1.0
        distanceSinceFixM = 0.0
        acceptedFixes = 0
        lastRaw = null
        baseMap = null
        baseRaw = null
        rawPathSinceBase = 0.0
        pendingLargeOffsetDeg = null
        mapNorthBearingDeg = null
        headingAligned = false
    }

    /**
     * The user says which way they face on the map ("I'm at the entrance, facing into the
     * store" = 0°): set the offset so [rawHeadingDeg] maps to [mapHeadingDeg]. Replaces any
     * learned offset — the compass's own north can be off by any amount.
     */
    fun alignHeading(rawHeadingDeg: Double, mapHeadingDeg: Double) {
        headingOffsetDeg = wrap(mapHeadingDeg - rawHeadingDeg)
        pendingLargeOffsetDeg = null
        headingAligned = true
    }

    /**
     * The venue config's map north: compass bearing of the map's +Y axis. A rough first
     * offset, used only while nothing better is known — [alignHeading] (pressing "I'm at
     * start") or a walked stretch replaces it. Applied now when the raw heading already
     * points to magnetic north ([rawIsNorthReferenced]), otherwise at [onNorthCalibrated]:
     * before that the raw heading's zero is arbitrary.
     */
    fun setMapNorth(compassBearingOfMapNorth: Double, rawIsNorthReferenced: Boolean) {
        mapNorthBearingDeg = compassBearingOfMapNorth
        if (rawIsNorthReferenced && !headingAligned) {
            headingOffsetDeg = wrap(-compassBearingOfMapNorth)
            pendingLargeOffsetDeg = null
        }
    }

    /**
     * The phone found north: the raw heading jumped by [rawJumpDeg] and now points to
     * magnetic north. Without a user / walked alignment, take the map north outright;
     * otherwise keep the map heading ([absorbRawHeadingJump]). Absorbing the jump on top of
     * a map-north offset set too early cancelled it.
     */
    fun onNorthCalibrated(rawJumpDeg: Double) {
        val north = mapNorthBearingDeg
        if (north != null && !headingAligned) {
            headingOffsetDeg = wrap(-north)
            pendingLargeOffsetDeg = null
        } else {
            absorbRawHeadingJump(rawJumpDeg)
        }
    }

    /** The raw heading itself jumped by [rawJumpDeg] (the phone found north): keep the map heading. */
    fun absorbRawHeadingJump(rawJumpDeg: Double) {
        headingOffsetDeg = wrap(headingOffsetDeg - rawJumpDeg)
        pendingLargeOffsetDeg = pendingLargeOffsetDeg?.let { wrap(it - rawJumpDeg) }
    }

    /** A new raw PDR sample. Moves the map position by the corrected step. */
    fun ingest(raw: Vec2) {
        val prev = lastRaw
        lastRaw = raw
        if (prev == null) return
        val d = (raw - prev).length
        if (d <= 0) return
        rawPathSinceBase += d
        val pos = mapPosition ?: return
        val step = rotate(raw - prev, headingOffsetDeg)
        mapPosition = pos + step * stepScale
        distanceSinceFixM += d * stepScale
    }

    /** The raw frame jumped without the user moving: continue from [raw] without integrating the jump. */
    fun rebase(raw: Vec2) {
        lastRaw = raw
    }

    /**
     * The server localized a photo to [node] (map frame) with [confidence]. [raw] is the PDR
     * reading when the photo was taken — the answer arrives seconds later, and the steps
     * walked since then are replayed by the next [ingest].
     */
    fun applyFix(node: Vec2, confidence: Double, raw: Vec2): FixResult {
        val needed = if (mapPosition == null) config.minFirstFixConfidence else config.minConfidence
        if (confidence < needed) return FixResult.RejectedLowConfidence
        // PDR walked since the shutter (straight line; the wait is a few seconds).
        val sinceCapture = lastRaw?.let { it - raw } ?: Vec2(0.0, 0.0)
        mapPosition?.let { pos ->
            // Compare with where dead reckoning had the user when the photo was taken.
            val atCapture = pos - rotate(sinceCapture, headingOffsetDeg) * stepScale
            val jump = (node - atCapture).length
            if (jump > config.maxJumpM && confidence < config.jumpOverrideConfidence) {
                return FixResult.RejectedJump(jump)
            }
        }

        val pathToCapture = maxOf(rawPathSinceBase - sinceCapture.length, 0.0)
        var keepBase = false
        val bm = baseMap
        val br = baseRaw
        if (bm != null && br != null) {
            val mapVec = node - bm
            val rawVec = raw - br
            val mapDist = mapVec.length
            val rawDist = rawVec.length
            // Only a long, fairly straight stretch says anything about heading / step length.
            val straight = pathToCapture > 0 && rawDist / pathToCapture > 0.8
            // Photos taken close together: keep measuring from the same start.
            keepBase = straight && (mapDist < config.minCalibrationDistanceM || rawDist < config.minCalibrationDistanceM)
            if (mapDist >= config.minCalibrationDistanceM && rawDist >= config.minCalibrationDistanceM && straight) {
                val observed = wrap(bearing(mapVec) - bearing(rawVec))
                if (abs(wrap(observed - headingOffsetDeg)) <= config.maxHeadingCorrectionDeg) {
                    headingOffsetDeg = wrap(headingOffsetDeg + config.learningRate * wrap(observed - headingOffsetDeg))
                    pendingLargeOffsetDeg = null
                    headingAligned = true
                } else {
                    // One stretch far off may be a wrong fix; two that agree mean the
                    // compass is off (90° in the 2026-10-17 store test): take their mean.
                    val prev = pendingLargeOffsetDeg
                    if (prev != null && abs(wrap(observed - prev)) <= config.largeCorrectionAgreementDeg) {
                        headingOffsetDeg = wrap(prev + wrap(observed - prev) / 2)
                        pendingLargeOffsetDeg = null
                        headingAligned = true
                    } else {
                        pendingLargeOffsetDeg = observed
                    }
                }
                val ratio = (mapDist / rawDist).coerceIn(config.stepScaleMin, config.stepScaleMax)
                stepScale = (stepScale + config.learningRate * (ratio - stepScale))
                    .coerceIn(config.stepScaleMin, config.stepScaleMax)
            }
        }

        mapPosition = node
        if (keepBase) {
            rawPathSinceBase = pathToCapture
        } else {
            baseMap = node
            baseRaw = raw
            rawPathSinceBase = 0.0
        }
        lastRaw = raw
        distanceSinceFixM = 0.0
        acceptedFixes++
        return FixResult.Accepted
    }

    /** Raw PDR heading → map heading, in [0, 360). */
    fun mapHeading(rawHeadingDeg: Double): Double {
        val h = (rawHeadingDeg + headingOffsetDeg) % 360.0
        return if (h < 0) h + 360.0 else h
    }

    companion object {
        /** Heading of a vector: 0° = +y, clockwise. */
        fun bearing(v: Vec2): Double = Math.toDegrees(atan2(v.x, v.y))

        /** Rotate clockwise by [deg] (a heading h becomes h + deg). */
        fun rotate(v: Vec2, deg: Double): Vec2 {
            val r = Math.toRadians(deg)
            return Vec2(v.x * cos(r) + v.y * sin(r), -v.x * sin(r) + v.y * cos(r))
        }

        fun wrap(deg: Double): Double {
            var d = deg % 360.0
            if (d > 180) d -= 360.0
            if (d <= -180) d += 360.0
            return d
        }
    }
}
