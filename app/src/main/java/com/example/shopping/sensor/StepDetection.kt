package com.example.shopping.sensor

import kotlin.math.exp

/**
 * Steps from the accelerometer, for the stretches where the phone's own step detector goes
 * quiet. Pure logic — unit tested; [PdrTracker] feeds it every accelerometer event.
 *
 * |a| minus its ~1 s average, lightly smoothed (~3 Hz): a step is a peak above
 * [THRESHOLD] that falls back below the average. Times are the sensor's event times, so
 * events delivered late in a burst (main thread busy with the camera) still count.
 * The first peak after a pause is held until a second one follows within [MAX_STEP_MS]
 * and then both count: a single bump is not a walk, but a 3-step walk keeps its first step.
 *
 * Measured on a hand-held Sensor Lab walk (0819全聯1, ~189 steps by cadence): the old
 * fallback (low-pass, 1.2 m/s² over a slow average) found 10; this finds ~154, and no
 * 4-second stretch of walking is left without a step (the hardware detector left 8 of 32).
 */
class SoftwareStepDetector {

    companion object {
        /** m/s² above the average. */
        const val THRESHOLD = 0.6
        const val MIN_STEP_MS = 280L
        const val MAX_STEP_MS = 2000L
        private const val BASE_TAU_S = 1.0
        private const val SMOOTH_TAU_S = 0.06
    }

    private var base = 0.0
    private var smooth = 0.0
    private var prevMs = -1L
    private var inPeak = false
    private var peak = 0.0
    private var peakMs = 0L
    private var lastStepMs = -1L
    private var pendingMs = -1L

    /** One sample: [tMs] its event time, [magnitude] |a| in m/s². Returns the event times of
     *  the steps it completes: usually none, one, or two when a walk starts. */
    fun ingest(tMs: Long, magnitude: Float): List<Long> {
        val m = magnitude.toDouble()
        if (prevMs < 0) {
            base = m; smooth = m; prevMs = tMs
            return emptyList()
        }
        val dt = (tMs - prevMs).coerceAtLeast(1) / 1000.0
        prevMs = tMs
        base += (1 - exp(-dt / BASE_TAU_S)) * (m - base)
        smooth += (1 - exp(-dt / SMOOTH_TAU_S)) * (m - smooth)
        val d = smooth - base
        if (d > THRESHOLD) {
            if (!inPeak || d > peak) { peak = d; peakMs = tMs }
            inPeak = true
            return emptyList()
        }
        if (!inPeak || d >= 0) return emptyList()
        inPeak = false

        if (lastStepMs >= 0 && peakMs - lastStepMs < MIN_STEP_MS) return emptyList()
        if (lastStepMs >= 0 && peakMs - lastStepMs <= MAX_STEP_MS) {
            lastStepMs = peakMs
            return listOf(peakMs)
        }
        // First peak after a pause: wait for a second one.
        if (pendingMs >= 0 && peakMs - pendingMs in MIN_STEP_MS..MAX_STEP_MS) {
            val first = pendingMs
            pendingMs = -1L
            lastStepMs = peakMs
            return listOf(first, peakMs)
        }
        if (pendingMs < 0 || peakMs - pendingMs > MAX_STEP_MS) pendingMs = peakMs
        return emptyList()
    }

    fun reset() {
        prevMs = -1L
        inPeak = false
        lastStepMs = -1L
        pendingMs = -1L
    }
}

/**
 * One step stream from the hardware step detector and [SoftwareStepDetector], by event time
 * (ms): hardware steps count; software steps count only once the hardware has been quiet
 * for [HW_SILENT_MS]; either is dropped within [SAME_STEP_MS] of a counted step (the other
 * source's report of the same footfall, or a late event for a step already counted).
 */
class StepMerger {

    companion object {
        const val HW_SILENT_MS = 1500L
        const val SAME_STEP_MS = 280L
    }

    private var lastStepMs = Long.MIN_VALUE / 2
    private var lastHwMs = Long.MIN_VALUE / 2

    /** A hardware step detector event; true = count it. */
    fun hardware(tMs: Long): Boolean {
        hardwareAlive(tMs)
        return count(tMs)
    }

    /** The hardware is counting (a step counter update), without a step to place. */
    fun hardwareAlive(tMs: Long) {
        if (tMs > lastHwMs) lastHwMs = tMs
    }

    /** A software step; true = count it. */
    fun software(tMs: Long): Boolean = tMs - lastHwMs > HW_SILENT_MS && count(tMs)

    private fun count(tMs: Long): Boolean {
        if (tMs - lastStepMs < SAME_STEP_MS) return false
        lastStepMs = tMs
        return true
    }

    fun reset() {
        lastStepMs = Long.MIN_VALUE / 2
        lastHwMs = Long.MIN_VALUE / 2
    }
}
