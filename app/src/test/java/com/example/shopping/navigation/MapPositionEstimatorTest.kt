package com.example.shopping.navigation

import com.example.shopping.navigation.MapPositionEstimator.FixResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class MapPositionEstimatorTest {

    private fun walk(e: MapPositionEstimator, from: Vec2, to: Vec2, steps: Int = 10) {
        for (i in 1..steps) e.ingest(from + (to - from) * (i.toDouble() / steps))
    }

    @Test
    fun noMapPositionUntilFirstFix() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0)); e.ingest(Vec2(0.0, 5.0))
        assertNull(e.mapPosition)
        assertFalse(e.isCalibrated)
    }

    @Test
    fun firstFixSnapsAndLaterStepsMoveTheMapPosition() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        assertEquals(FixResult.Accepted, e.applyFix(Vec2(10.0, 20.0), 0.4, raw = Vec2(0.0, 0.0)))
        e.ingest(Vec2(0.0, 3.0))
        assertEquals(10.0, e.mapPosition!!.x, 1e-9)
        assertEquals(23.0, e.mapPosition!!.y, 1e-9)
        assertEquals(3.0, e.distanceSinceFixM, 1e-9)
    }

    @Test
    fun lowConfidenceFixIsRejected() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        assertEquals(FixResult.RejectedLowConfidence, e.applyFix(Vec2(1.0, 1.0), 0.2, Vec2(0.0, 0.0)))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        // after the first fix the bar is higher
        assertEquals(FixResult.RejectedLowConfidence, e.applyFix(Vec2(1.0, 1.0), 0.5, Vec2(0.0, 0.0)))
    }

    @Test
    fun implausibleJumpIsRejectedUnlessVeryConfident() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        assertTrue(e.applyFix(Vec2(30.0, 0.0), 0.7, Vec2(0.0, 0.0)) is FixResult.RejectedJump)
        assertEquals(FixResult.Accepted, e.applyFix(Vec2(30.0, 0.0), 0.9, Vec2(0.0, 0.0)))
    }

    @Test
    fun jumpIsMeasuredAtCaptureTimeNotAfterWalkingOn() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        walk(e, Vec2(0.0, 0.0), Vec2(0.0, 20.0))          // photo taken at raw y=20
        walk(e, Vec2(0.0, 20.0), Vec2(0.0, 30.0))         // walked 10 m more while waiting
        // node at y=21 is 1 m from where the photo was taken (but 9 m from "now")
        assertEquals(FixResult.Accepted, e.applyFix(Vec2(0.0, 21.0), 0.7, Vec2(0.0, 20.0)))
        e.ingest(Vec2(0.0, 30.0))                         // replay the steps since the shutter
        // the 20 m straight stretch also calibrated the step scale: 1 + 0.3·(21/20 − 1)
        assertEquals(1.015, e.stepScale, 1e-9)
        assertEquals(21.0 + 10.0 * 1.015, e.mapPosition!!.y, 1e-6)
    }

    @Test
    fun straightStretchBetweenFixesLearnsHeadingOffsetAndStepScale() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        // PDR thinks it walked 10 m north, the map says 12 m east-ish (rotated +10°)
        walk(e, Vec2(0.0, 0.0), Vec2(0.0, 10.0))
        val trueVec = MapPositionEstimator.rotate(Vec2(0.0, 12.0), 10.0)
        e.applyFix(trueVec, 0.9, Vec2(0.0, 10.0))
        assertEquals(0.3 * 10.0, e.headingOffsetDeg, 1e-6)
        assertEquals(1.0 + 0.3 * (1.2 - 1.0), e.stepScale, 1e-6)
    }

    @Test
    fun stepScaleIsClamped() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        walk(e, Vec2(0.0, 0.0), Vec2(0.0, 10.0))
        e.applyFix(Vec2(0.0, 40.0), 0.9, Vec2(0.0, 10.0))  // ratio 4 → clamped to 1.3
        assertEquals(1.0 + 0.3 * (1.3 - 1.0), e.stepScale, 1e-6)
    }

    @Test
    fun alignHeadingMakesTheCurrentRawHeadingTheGivenMapHeading() {
        val e = MapPositionEstimator()
        e.alignHeading(rawHeadingDeg = 93.7, mapHeadingDeg = 0.0)   // compass said east, user faces into the store
        assertEquals(0.0, e.mapHeading(93.7), 1e-9)
        assertEquals(90.0, e.mapHeading(183.7), 1e-9)
    }

    @Test
    fun alignHeadingAlsoTurnsLaterSteps() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        e.alignHeading(rawHeadingDeg = 90.0, mapHeadingDeg = 0.0)
        e.ingest(Vec2(5.0, 0.0))                                     // raw "east" = map +y
        assertEquals(0.0, e.mapPosition!!.x, 1e-9)
        assertEquals(5.0, e.mapPosition!!.y, 1e-9)
    }

    @Test
    fun rawHeadingJumpKeepsTheMapHeading() {
        val e = MapPositionEstimator()
        e.alignHeading(rawHeadingDeg = 10.0, mapHeadingDeg = 0.0)
        e.absorbRawHeadingJump(75.0)                                 // phone found north: raw 10° → 85°
        assertEquals(0.0, e.mapHeading(85.0), 1e-9)
    }

    @Test
    fun mapNorthWaitsForTheRawHeadingToFindNorth() {
        val e = MapPositionEstimator()
        e.setMapNorth(76.0, rawIsNorthReferenced = false)            // raw zero still arbitrary
        assertEquals(0.0, e.headingOffsetDeg, 1e-9)
        e.onNorthCalibrated(40.0)                                    // raw now magnetic: the jump is not absorbed
        assertEquals(-76.0, e.headingOffsetDeg, 1e-9)
        assertEquals(0.0, e.mapHeading(76.0), 1e-9)                  // facing compass 76° = map +y
    }

    @Test
    fun mapNorthAppliesAtOnceWhenNorthIsAlreadyKnown() {
        val e = MapPositionEstimator()
        e.setMapNorth(330.0, rawIsNorthReferenced = true)
        assertEquals(0.0, e.mapHeading(330.0), 1e-9)
    }

    @Test
    fun userAlignmentOutranksMapNorth() {
        val e = MapPositionEstimator()
        e.setMapNorth(76.0, rawIsNorthReferenced = false)
        e.alignHeading(rawHeadingDeg = 10.0, mapHeadingDeg = 0.0)    // "I'm at the entrance"
        e.onNorthCalibrated(75.0)                                    // raw 10° → 85°: keep the user's heading
        assertEquals(0.0, e.mapHeading(85.0), 1e-9)
        e.setMapNorth(76.0, rawIsNorthReferenced = true)             // a later session's map north doesn't override it
        assertEquals(0.0, e.mapHeading(85.0), 1e-9)
    }

    @Test
    fun walkedCorrectionOutranksMapNorth() {
        val e = MapPositionEstimator()
        e.setMapNorth(76.0, rawIsNorthReferenced = false)
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        walk(e, Vec2(0.0, 0.0), Vec2(0.0, 10.0))
        e.applyFix(Vec2(1.0, 10.0), 0.9, Vec2(0.0, 10.0))            // small offset learned from walking
        val learned = e.headingOffsetDeg
        e.onNorthCalibrated(20.0)
        assertEquals(learned - 20.0, e.headingOffsetDeg, 1e-9)
    }

    /** Fix at the origin, then two 10 m stretches PDR calls north that the map says go east. */
    private fun twoStretchesOff(e: MapPositionEstimator) {
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        walk(e, Vec2(0.0, 0.0), Vec2(0.0, 10.0))
        assertEquals(FixResult.Accepted, e.applyFix(Vec2(10.0, 0.0), 0.9, Vec2(0.0, 10.0)))
        assertEquals(0.0, e.headingOffsetDeg, 1e-9)                  // one stretch 90° off: could be a wrong fix
        walk(e, Vec2(0.0, 10.0), Vec2(0.0, 20.0))
        assertEquals(FixResult.Accepted, e.applyFix(Vec2(20.0, 0.0), 0.9, Vec2(0.0, 20.0)))
    }

    @Test
    fun twoAgreeingStretchesFarOffReplaceTheOffset() {
        val e = MapPositionEstimator()
        twoStretchesOff(e)
        assertEquals(90.0, e.headingOffsetDeg, 1e-6)
    }

    @Test
    fun twoDisagreeingStretchesFarOffChangeNothing() {
        val e = MapPositionEstimator()
        e.ingest(Vec2(0.0, 0.0))
        e.applyFix(Vec2(0.0, 0.0), 1.0, Vec2(0.0, 0.0))
        walk(e, Vec2(0.0, 0.0), Vec2(0.0, 10.0))
        e.applyFix(Vec2(10.0, 0.0), 0.9, Vec2(0.0, 10.0))             // 90° off
        walk(e, Vec2(0.0, 10.0), Vec2(0.0, 20.0))
        e.applyFix(Vec2(0.0, -10.0), 0.9, Vec2(0.0, 20.0))            // then 180°-ish off: no agreement
        assertEquals(0.0, e.headingOffsetDeg, 1e-6)
    }

    @Test
    fun mapHeadingAddsOffsetAndWraps() {
        val e = MapPositionEstimator()
        assertEquals(350.0, e.mapHeading(-10.0), 1e-9)
        assertEquals(10.0, e.mapHeading(370.0), 1e-9)
    }

    @Test
    fun geometryHelpers() {
        assertEquals(0.0, MapPositionEstimator.bearing(Vec2(0.0, 1.0)), 1e-9)
        assertEquals(90.0, MapPositionEstimator.bearing(Vec2(1.0, 0.0)), 1e-9)
        val r = MapPositionEstimator.rotate(Vec2(0.0, 1.0), 90.0)   // north rotated clockwise → east
        assertEquals(1.0, r.x, 1e-9); assertEquals(0.0, r.y, 1e-9)
        assertEquals(-170.0, MapPositionEstimator.wrap(190.0), 1e-9)
        assertEquals(180.0, MapPositionEstimator.wrap(-180.0), 1e-9)
    }
}
