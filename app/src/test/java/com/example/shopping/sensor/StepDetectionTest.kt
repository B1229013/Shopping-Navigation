package com.example.shopping.sensor

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

class StepDetectionTest {

    private val g = 9.81f

    /** |a| at 50 Hz: [stillS] standing, then [steps] steps at [hz] of ±[amp] m/s², then standing again. */
    private fun walk(steps: Int, hz: Double = 1.8, amp: Double = 1.0, stillS: Double = 1.0,
                     noise: Double = 0.05, period: () -> Long = { 20L }): List<Pair<Long, Float>> {
        val rnd = Random(7)
        val walkS = steps / hz
        val out = mutableListOf<Pair<Long, Float>>()
        var t = 0L
        while (t <= ((stillS * 2 + walkS) * 1000).toLong()) {
            val s = t / 1000.0 - stillS
            val step = if (s in 0.0..walkS) amp * sin(2 * PI * hz * s) else 0.0
            out += t to (g + step + noise * (rnd.nextDouble() * 2 - 1)).toFloat()
            t += period()
        }
        return out
    }

    private fun count(samples: List<Pair<Long, Float>>): List<Long> {
        val d = SoftwareStepDetector()
        return samples.flatMap { (t, m) -> d.ingest(t, m) }
    }

    @Test
    fun gentleHandHeldWalkIsCounted() {
        // ±1 m/s² is a phone held up in front while walking slowly; the old fallback
        // (low-pass + 1.2 m/s² over the average) counted 10 of ~189 such steps in a real log.
        assertEquals(18, count(walk(steps = 18)).size)
    }

    @Test
    fun aShortWalkKeepsItsFirstStep() {
        // 2–3 m between test points: every step matters, including the first.
        assertEquals(4, count(walk(steps = 4)).size)
    }

    @Test
    fun standingStillCountsNothing() {
        assertEquals(0, count(walk(steps = 0, stillS = 10.0, noise = 0.3)).size)
    }

    @Test
    fun oneBumpIsNotAWalk() {
        val samples = walk(steps = 0, stillS = 3.0).map { (t, m) -> t to if (t in 2000L..2080L) m + 3f else m }
        assertEquals(0, count(samples).size)
    }

    @Test
    fun usesEventTimesNotArrivalRate() {
        // Uneven sampling (27–45 ms, as in the Sensor Lab logs) gives the same count.
        val rnd = Random(3)
        assertEquals(12, count(walk(steps = 12, period = { 27L + rnd.nextInt(19) })).size)
    }

    @Test
    fun stepsAreAtLeastMinIntervalApart() {
        val times = count(walk(steps = 20, hz = 2.2))
        assertTrue(times.zipWithNext().all { (a, b) -> b - a >= SoftwareStepDetector.MIN_STEP_MS })
    }

    // ── StepMerger: hardware detector first, software while it is quiet ──

    @Test
    fun softwareStepsWaitWhileHardwareIsReporting() {
        val m = StepMerger()
        assertTrue(m.hardware(1000))
        assertFalse(m.software(1500))
        assertTrue(m.hardware(1550))
    }

    @Test
    fun softwareStepsCountOnceHardwareIsQuiet() {
        val m = StepMerger()
        assertTrue(m.hardware(1000))
        assertTrue(m.software(1000 + StepMerger.HW_SILENT_MS + 1))
        assertTrue(m.software(1000 + StepMerger.HW_SILENT_MS + 551))
    }

    @Test
    fun aHardwareStepNextToACountedStepIsTheSameStep() {
        val m = StepMerger()
        assertTrue(m.software(5000))
        assertFalse(m.hardware(5100))           // the detector catching up on the same footfall
        assertFalse(m.hardware(4700))           // a late event for a step already counted
        assertTrue(m.hardware(5600))
    }

    @Test
    fun stepCounterEventsKeepSoftwareQuiet() {
        val m = StepMerger()
        m.hardwareAlive(1000)
        assertFalse(m.software(1400))
        m.reset()
        assertTrue(m.software(1400))
    }
}
