import Foundation

/// The five heading-computation methods being A/B tested against each other in
/// `SensorComboTracker` — see that file's doc comment for the Android-terminology mapping
/// (Rotation Vector / Game Rotation Vector / raw sensor combinations) this project is
/// replicating.
enum HeadingMethodID: String, CaseIterable, Codable {
    case accelMag = "加速度＋磁力"
    case accelGyro = "加速度＋陀螺儀"
    case accelMagGyro = "加速度＋磁力＋陀螺儀"
    case rotationVector = "旋轉向量"
    case gameRotationVector = "Game Rotation Vector"
}

/// Tilt-independent heading/yaw-rate formulas built from raw gravity/magnetic/gyro vectors.
///
/// Both formulas here are deliberately **not** the rectified (√ of squares, always ≥ 0)
/// style used by the earlier Holding/Pocket/Swing gyro formulas in `MotionPathTracker`,
/// which turned out to amplify sensor noise into phantom turns (see that file's history).
/// A cross product and a dot product are both *signed*, physically meaningful quantities
/// that don't have that rectification problem, so this is a more robust foundation —
/// though still unverified on real hardware until tested via `SensorComboTestView`.
enum TiltCompensatedCompass {
    /// Standard "cross product" tilt-compensated compass heading (the same technique
    /// Android's own `SensorManager.getRotationMatrix()` uses internally, and what the
    /// `nju-aml2022/Pedestrian-Dead-Reckoning-PDR` reference project's `direction_predictor.py`
    /// does): geometric, so it works at any device tilt without needing roll/pitch angles or
    /// their sign conventions.
    ///
    /// `east = normalize(magnetic × gravity)`, `north = normalize(gravity × east)` — these
    /// give the East/North directions expressed in the phone's own local axes. The heading of
    /// the phone's own +Y axis ("top of phone", the reference direction for this app's
    /// Holding posture) relative to North is then `atan2(east·(0,1,0), north·(0,1,0))` =
    /// `atan2(east.y, north.y)`.
    ///
    /// Returns `nil` in the degenerate case where magnetic and gravity vectors are
    /// (near-)parallel (cross product ≈ 0) — physically means the phone is oriented so its
    /// magnetic-field reading can't be used to resolve a horizontal direction right now.
    ///
    /// This exact cross-product derivation (`east = magnetic × up`, `north = up × east`)
    /// is copied from Android's `SensorManager.getRotationMatrix()` / the nju-aml2022 PDR
    /// reference project, both of which feed it an accelerometer/gravity vector that points
    /// *away* from the earth (the reaction-force convention). iOS's own
    /// `CMDeviceMotion.gravity` points the other way — *toward* the earth (Apple's docs:
    /// lying flat screen-up gives `gravity.z == -1`) — so it has to be negated here before
    /// the formula sees it. Skipping this negation was an actual bug that shipped for a
    /// while: it flips the `east` cross product but not `north` (two negations cancel
    /// there), so the resulting heading came out as `-trueHeading` — a walked path mirrored
    /// left/right around the north–south line relative to GPS ground truth. See
    /// `SensorComboTestView`'s comparison chart, which is what surfaced this.
    static func heading(magnetic: (x: Double, y: Double, z: Double), gravity: (x: Double, y: Double, z: Double)) -> Double? {
        let up = (x: -gravity.x, y: -gravity.y, z: -gravity.z)
        let east = cross(magnetic, up)
        let eastNorm = norm(east)
        guard eastNorm > 0.01 else { return nil }
        let eastUnit = (x: east.x / eastNorm, y: east.y / eastNorm, z: east.z / eastNorm)

        let north = cross(up, eastUnit)
        let northNorm = norm(north)
        guard northNorm > 0.01 else { return nil }
        let northUnit = (x: north.x / northNorm, y: north.y / northNorm, z: north.z / northNorm)

        var heading = atan2(eastUnit.y, northUnit.y) * 180 / .pi
        if heading < 0 { heading += 360 }
        return heading
    }

    /// The component of the gyro's angular velocity along the (world) vertical axis —
    /// i.e. the true yaw rate, regardless of how the phone is tilted — obtained by
    /// projecting the raw gyro vector onto the (unit) gravity vector: `ω_yaw = ω · ĝ`.
    /// This works at any tilt without decomposing into roll/pitch/yaw Euler angles (and
    /// their attendant sign/axis-order pitfalls); it's the same principle used in
    /// various AHRS implementations for isolating yaw rate from a tilted IMU.
    ///
    /// Deliberately left using iOS's `gravity` as-is (not negated like `heading(_:_:)`
    /// above) — a dot product only flips sign under negation, it doesn't have that
    /// function's two-cross-products-cancel-one-negation asymmetry, so the direction this
    /// was actually validated against (turning right on a real walk should increase
    /// `accelGyro`'s heading, matching the now-fixed `heading(_:_:)` convention) needs
    /// re-checking on a device now that that fix landed — if a straight-line test starts
    /// curving the wrong way after the fix, this is the term to negate next.
    static func verticalAxisYawRate(gyro: (x: Double, y: Double, z: Double), gravity: (x: Double, y: Double, z: Double)) -> Double {
        let gn = norm(gravity)
        guard gn > 0.01 else { return 0 }
        let g = (x: gravity.x / gn, y: gravity.y / gn, z: gravity.z / gn)
        return gyro.x * g.x + gyro.y * g.y + gyro.z * g.z
    }

    private static func cross(_ a: (x: Double, y: Double, z: Double), _ b: (x: Double, y: Double, z: Double)) -> (x: Double, y: Double, z: Double) {
        (x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x)
    }

    private static func norm(_ v: (x: Double, y: Double, z: Double)) -> Double {
        (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    }
}
