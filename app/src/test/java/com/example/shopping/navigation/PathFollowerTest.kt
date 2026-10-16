package com.example.shopping.navigation

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PathFollowerTest {
    // L-shaped route: north 20 m, then turn right and go east 10 m
    private val route = listOf(Vec2(0.0, 0.0), Vec2(0.0, 20.0), Vec2(10.0, 20.0))
    private val turns = listOf(Vec2(0.0, 0.0) to "straight", Vec2(0.0, 20.0) to "right")

    private fun follower(pdrAtStart: Vec2 = Vec2(0.0, 0.0)) =
        PathFollower().apply { setRoute(route, turns, pdrAtStart) }

    @Test
    fun noRouteNoState() {
        assertNull(PathFollower().update(Vec2(0.0, 0.0), 0.0))
    }

    @Test
    fun anchorsAtRouteStartAndTracksPdrOffsets() {
        val f = follower(pdrAtStart = Vec2(5.0, 5.0))       // raw PDR started elsewhere
        val s = f.update(Vec2(5.0, 10.0), 0.0)!!           // walked 5 m north
        assertEquals(0.0, s.position.x, 1e-9)
        assertEquals(5.0, s.position.y, 1e-9)
        assertEquals(5.0, s.progressM, 1e-9)
        assertEquals(25.0, s.remainingM, 1e-9)
    }

    @Test
    fun announcesTheTurnByDistance() {
        val f = follower()
        assertEquals("直走約 15 公尺後右轉", f.update(Vec2(0.0, 5.0), 0.0)!!.prompt)
        assertEquals("10 公尺後右轉", f.update(Vec2(0.0, 10.0), 0.0)!!.prompt)
        val near = f.update(Vec2(0.0, 18.0), 0.0)!!
        assertEquals("現在右轉", near.prompt)
        assertEquals("right", near.nextTurnDirection)
    }

    @Test
    fun afterTheTurnCountsDownToTheTarget() {
        val f = follower()
        f.update(Vec2(0.0, 20.0), 90.0)
        val s = f.update(Vec2(4.0, 20.0), 90.0)!!
        assertEquals("直走約 6 公尺就到", s.prompt)
        assertFalse(s.arrived)
    }

    @Test
    fun arrivesNearTheEnd() {
        val f = follower()
        f.update(Vec2(0.0, 20.0), 90.0)
        val s = f.update(Vec2(8.0, 20.0), 90.0)!!
        assertTrue(s.arrived)
        assertTrue(s.prompt.startsWith("目標就在附近"))
    }

    @Test
    fun offRouteAsksForReplan() {
        val s = follower().update(Vec2(9.0, 8.0), 0.0)!!
        assertTrue(s.needsReplan)
        assertTrue(s.prompt.contains("偏離路線"))
    }

    @Test
    fun keepAnchorAndIdentityAnchorForMapPositions() {
        val f = follower()
        f.anchor(Vec2(0.0, 12.0), Vec2(0.0, 12.0))         // map-frame fix: identity anchor
        val s = f.update(Vec2(0.0, 14.0), 0.0)!!
        assertEquals(14.0, s.position.y, 1e-9)
        assertEquals(14.0, s.progressM, 1e-9)
    }

    @Test
    fun lookAheadPointsFurtherAlongTheRoute() {
        val f = follower()
        val s = f.update(Vec2(0.0, 18.0), 0.0)!!
        // 5 m ahead of progress 18 = 3 m past the corner on the east leg
        assertEquals(3.0, s.lookAhead!!.x, 1e-9)
        assertEquals(20.0, s.lookAhead!!.y, 1e-9)
        assertEquals(10.0, f.pointAt(100.0)!!.x, 1e-9)       // clamped to the end
    }
}
