package com.example.shopping.navigation

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Everything the screen needs about following the route right now. */
data class FollowState(
    val position: Vec2,              // map metres
    val headingDegrees: Double,
    val progressM: Double,           // distance walked along the route
    val remainingM: Double,          // distance still to go
    val offRouteM: Double,           // perpendicular distance from the route
    val nextTurnDirection: String?,  // e.g. "right"; null when only straight remains
    val nextTurnInM: Double?,        // distance until that turn
    val prompt: String,              // what to tell the user right now
    val needsReplan: Boolean,        // drifted too far from the route
    val arrived: Boolean,            // within reach of the target
    val lookAhead: Vec2?,            // a point a few metres further along the route (AR arrow aim)
)

/**
 * Walks a planned route (map metres, +y = heading 0°) with the phone's position.
 * Port of the iOS PathFollower (APPNAV_ios/Navigation/PathFollower.swift).
 *
 * Keeps an anchor: the map point the user was last known to be at paired with the PDR
 * reading at that moment; map position = anchor + (PDR now − PDR at anchor). Once the
 * MapPositionEstimator has a fix the caller feeds map positions directly and anchors
 * with identity (mapPoint == pdr). Pure logic — unit tested.
 */
class PathFollower {
    companion object {
        /** Beyond this cross-track distance the route no longer describes where the user is. */
        const val OFF_ROUTE_THRESHOLD_M = 6.0
        /** Within this distance of the end the target counts as reached. */
        const val ARRIVAL_RADIUS_M = 3.0
        /** A turn is "now" inside this distance, "soon" up to ANNOUNCE_TURN_M. */
        const val TURN_NOW_M = 3.0
        const val ANNOUNCE_TURN_M = 12.0
        /** How far ahead along the route the AR arrow aims. */
        const val LOOK_AHEAD_M = 5.0

        fun zh(direction: String): String = when (direction) {
            "right" -> "右轉"
            "left" -> "左轉"
            "slight_right" -> "稍微右轉"
            "slight_left" -> "稍微左轉"
            "sharp_right" -> "大幅右轉"
            "sharp_left" -> "大幅左轉"
            "behind" -> "迴轉"
            else -> "直走"
        }

        private fun arrivalPrompt(remaining: Double) =
            "目標就在附近（約 ${max(1, remaining.roundToInt())} 公尺），請環顧四周"

        /** Projection of p onto segment a–b: (t clamped to 0..1, perpendicular distance). */
        fun project(p: Vec2, a: Vec2, b: Vec2): Pair<Double, Double> {
            val v = b - a
            val len2 = v.x * v.x + v.y * v.y
            if (len2 <= 0) return 0.0 to (p - a).length
            val t = (((p.x - a.x) * v.x + (p.y - a.y) * v.y) / len2).coerceIn(0.0, 1.0)
            return t to (p - (a + v * t)).length
        }

        fun nearestVertexIndex(p: Vec2, line: List<Vec2>): Int =
            line.indices.minByOrNull { (line[it] - p).length } ?: 0
    }

    var polyline: List<Vec2> = emptyList(); private set
    private var cumulative: List<Double> = emptyList()
    private var turnsAlongRoute: List<Pair<Double, String>> = emptyList()
    private var segmentIndex = 0
    private var anchorMap = Vec2(0.0, 0.0)
    private var anchorPdr = Vec2(0.0, 0.0)
    var hasRoute = false; private set

    val totalLengthM: Double get() = cumulative.lastOrNull() ?: 0.0

    /**
     * Install a route. [turns] are (map point where the step starts, direction). The user
     * is assumed to be at the first vertex unless [keepAnchor] is set.
     */
    fun setRoute(polyline: List<Vec2>, turns: List<Pair<Vec2, String>>, pdr: Vec2, keepAnchor: Boolean = false) {
        this.polyline = polyline
        var acc = 0.0
        cumulative = polyline.mapIndexed { i, p -> if (i > 0) acc += (p - polyline[i - 1]).length; acc }
        turnsAlongRoute = turns
            .filter { it.second != "straight" }
            .map { (at, dir) -> cumulative[nearestVertexIndex(at, polyline)] to dir }
            .sortedBy { it.first }
        segmentIndex = 0
        hasRoute = polyline.isNotEmpty()
        if (!keepAnchor && polyline.isNotEmpty()) anchor(polyline.first(), pdr)
    }

    /** Pin the user's map position to the PDR reading at this moment. */
    fun anchor(mapPoint: Vec2, pdr: Vec2) {
        anchorMap = mapPoint
        anchorPdr = pdr
        segmentIndex = 0
    }

    fun update(pdr: Vec2, headingDegrees: Double): FollowState? {
        if (!hasRoute) return null
        val pos = anchorMap + (pdr - anchorPdr)

        if (polyline.size < 2) {
            val d = (polyline[0] - pos).length
            return FollowState(pos, headingDegrees, 0.0, d, 0.0, null, null, arrivalPrompt(d),
                needsReplan = false, arrived = d <= ARRIVAL_RADIUS_M, lookAhead = polyline[0])
        }

        // Project onto the route, looking only a few segments ahead so a route that
        // doubles back on itself can't teleport the marker backwards.
        val lo = max(0, segmentIndex - 1)
        val hi = min(polyline.size - 2, segmentIndex + 3)
        var bestSeg = lo
        var bestT = 0.0
        var bestCross = Double.MAX_VALUE
        for (seg in lo..hi) {
            val (t, cross) = project(pos, polyline[seg], polyline[seg + 1])
            if (cross < bestCross) { bestSeg = seg; bestT = t; bestCross = cross }
        }
        segmentIndex = bestSeg
        val segLen = cumulative[bestSeg + 1] - cumulative[bestSeg]
        val progress = cumulative[bestSeg] + bestT * segLen
        val toTarget = (polyline.last() - pos).length
        val remaining = max(totalLengthM - progress, toTarget)

        val next = turnsAlongRoute.firstOrNull { it.first > progress - 1.0 }
        val turnIn = next?.let { max(0.0, it.first - progress) }

        val arrived = toTarget <= ARRIVAL_RADIUS_M
        val offRoute = bestCross > OFF_ROUTE_THRESHOLD_M && !arrived
        val prompt = when {
            arrived -> arrivalPrompt(toTarget)
            offRoute -> "似乎偏離路線（偏差約 ${bestCross.roundToInt()} 公尺），重新規劃中…"
            next != null && turnIn != null -> {
                val verb = zh(next.second)
                when {
                    turnIn <= TURN_NOW_M -> "現在$verb"
                    turnIn <= ANNOUNCE_TURN_M -> "${turnIn.roundToInt()} 公尺後$verb"
                    else -> "直走約 ${turnIn.roundToInt()} 公尺後$verb"
                }
            }
            else -> "直走約 ${remaining.roundToInt()} 公尺就到"
        }

        return FollowState(pos, headingDegrees, progress, remaining, bestCross,
            next?.second, turnIn, prompt, offRoute, arrived, pointAt(progress + LOOK_AHEAD_M))
    }

    /** The point [distanceM] along the route (clamped to its ends). */
    fun pointAt(distanceM: Double): Vec2? {
        if (polyline.isEmpty()) return null
        if (distanceM <= 0) return polyline.first()
        for (i in 1 until polyline.size) {
            if (cumulative[i] >= distanceM) {
                val seg = cumulative[i] - cumulative[i - 1]
                val t = if (seg > 0) (distanceM - cumulative[i - 1]) / seg else 0.0
                return polyline[i - 1] + (polyline[i] - polyline[i - 1]) * t
            }
        }
        return polyline.last()
    }
}
