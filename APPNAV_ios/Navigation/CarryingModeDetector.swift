import Foundation

/// Four of the six phone-carrying postures from 劉嘉心〈基於行人航位推算之室內定位研究〉
/// (國立臺北科技大學電機工程系碩士論文, 2020) §3.2, plus a catch-all for readings that don't
/// land clearly in any of them. Calling Right/Left were dropped — this app's navigation
/// feature never has a reason to be in a phone-call posture while navigating.
enum CarryingMode: String, CaseIterable {
    case holding = "手持直立"
    case pocket = "口袋"
    case swingRight = "右手自然擺動"
    case swingLeft = "左手自然擺動"
    case unknown = "無法判斷"
}

/// Classifies the current `CarryingMode` from the gravity vector alone, per 劉嘉心 (2020)
/// §3.4.1/§3.4.3/§3.4.4 — this is only the paper's **first** detection stage ("Mode1", the
/// conditional-threshold classifier), not the full four-stage pipeline:
///
/// 1. ✅ Implemented here — conditional classifier on `degTx`/`degP`.
/// 2. ❌ Not implemented — Sliding Window Algorithm (§3.4.5) that resolves the overlap
///    regions between modes (the paper's own analysis notes `degTx` overlaps heavily across
///    all six modes, and even `degP` — the better discriminator — still overlaps at the
///    edges between adjacent modes).
/// 3. ❌ Not implemented — mode-switch debounce (§3.4.6) that suppresses false switches from
///    momentary readings.
/// 4. ❌ Not implemented — output/flag update stage (§3.4.7).
///
/// Deliberately scoped this way: stages 2-4 add real complexity (and, per this project's
/// track record with hand-rolled sensor formulas, real bug risk) on top of a classifier
/// that hasn't been validated on real hardware yet. Test stage 1 alone first (see the live
/// readout in `SensorTestView`) before deciding whether stages 2-4 are worth the added risk.
///
/// Also note: the paper's exact threshold cut points (`Rth1`-`Rth5`, `Pth1`-`Pth5`) are shown
/// only in figures in the source PDF, not as transcribed numbers in the text — the ranges
/// used in `classify(degTx:degP:)` below are this implementation's best-effort derivation
/// from the paper's *descriptive* angle ranges (§3.4.3), not transcribed constants, so they
/// should be treated as a starting point to tune against real readings, not ground truth.
final class CarryingModeDetector {
    private(set) var degTx: Double = 0
    private(set) var degP: Double = 0
    private(set) var mode: CarryingMode = .unknown

    /// `gravity` is the gravity vector in the phone's own local frame (e.g. from
    /// `CMDeviceMotion.gravity` — same axis convention as Android's gravity sensor, which
    /// is what the paper's `gx`/`gy`/`gz` refer to).
    func ingest(gravity: (x: Double, y: Double, z: Double)) {
        let gn = (gravity.x * gravity.x + gravity.y * gravity.y + gravity.z * gravity.z).squareRoot()
        guard gn > 0.01 else { return }

        degTx = Self.tiltAngle(gravity: gravity, gn: gn)
        degP = Self.tiltPlaneAttitude(gravity: gravity, gn: gn) ?? degP
        mode = Self.classify(degTx: degTx, degP: degP)
    }

    /// §3.4.1 eq. (3.1): tilt angle from rotation about the x-axis. 0° = lying flat
    /// screen-up; ±180° = lying flat screen-down; ±90° = standing on its side.
    private static func tiltAngle(gravity: (x: Double, y: Double, z: Double), gn: Double) -> Double {
        let gzRatio = abs(gravity.z) / gn
        if gravity.z >= 0 {
            return gravity.y >= 0 ? 90 * (1 - gzRatio) : -90 * (1 - gzRatio)
        } else {
            return gravity.y >= 0 ? 90 * (1 + gzRatio) : -90 * (1 + gzRatio)
        }
    }

    /// §3.4.1 eq. (3.4): attitude angle within the tilt plane found above — this is what
    /// actually discriminates between the six modes (see class doc comment). Returns `nil`
    /// in the degenerate case where the phone is lying exactly flat (no tilt plane to
    /// measure an angle within), so the caller keeps the last valid reading instead of
    /// snapping to a meaningless value.
    private static func tiltPlaneAttitude(gravity: (x: Double, y: Double, z: Double), gn: Double) -> Double? {
        let denom = (gn * gn - gravity.z * gravity.z).squareRoot()
        guard denom > 0.01 else { return nil }
        let ratio = gravity.y / denom
        if gravity.x <= 0 {
            return ratio * 90
        } else if gravity.y >= 0 {
            return 180 - ratio * 90
        } else {
            return -180 - ratio * 90
        }
    }

    /// §3.4.3's descriptive ranges (standing + walking cases unioned together, since this
    /// stage doesn't yet distinguish them — see class doc comment). Checked in an order that
    /// prioritizes the modes this app actually cares about (Holding is the only one the real
    /// navigation feature ever uses; the rest matter only for making sure a *test* walk was
    /// actually held the way it should be).
    private static func classify(degTx: Double, degP: Double) -> CarryingMode {
        if degTx >= 0, degTx <= 90, degP >= 45, degP <= 135 {
            return .holding
        }
        if abs(degTx) >= 15, abs(degTx) <= 165 {
            if (degP <= -60 && degP >= -180) || (degP >= 150 && degP <= 180) {
                return .swingRight
            }
            if degP >= -120, degP <= 30 {
                return .swingLeft
            }
        }
        if abs(degTx) >= 30, abs(degTx) <= 150 {
            if (degP >= 45 && degP <= 135) || (degP <= -45 && degP >= -135) {
                return .pocket
            }
        }
        return .unknown
    }

    func reset() {
        degTx = 0
        degP = 0
        mode = .unknown
    }
}
