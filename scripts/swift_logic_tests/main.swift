// Unit tests for the UI-free navigation logic, runnable without Xcode:
//   scripts/swift_logic_tests/run.sh
// (The app has no XCTest target; these files are compiled together with the
// production sources they test.)
import CoreGraphics
import Foundation

var failures = 0
var passed = 0
func check(_ cond: Bool, _ msg: String, file: String = #file, line: Int = #line) {
    if cond { passed += 1 } else { failures += 1; print("FAIL line \(line): \(msg)") }
}
func approx(_ a: Double, _ b: Double, _ tol: Double = 1e-6) -> Bool { abs(a - b) <= tol }

// MARK: - MapPositionEstimator

do {  // no position before the first fix; integrates steps after it
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0)); e.ingest(raw: (0, 5))
    check(e.mapPosition == nil, "uncalibrated has no map position")
    check(e.applyFix(node: (10, 20), confidence: 0.9, raw: (0, 5)) == .accepted, "first fix accepted")
    e.ingest(raw: (0, 8))
    check(approx(e.mapPosition!.x, 10) && approx(e.mapPosition!.y, 23), "moves 3 m north after fix")
    check(approx(e.distanceSinceFixM, 3), "distance since fix = 3")
}

do {  // low-confidence and jump rejection
    let e = MapPositionEstimator()
    check(e.applyFix(node: (0, 0), confidence: 0.2, raw: (0, 0)) == .rejectedLowConfidence, "low conf rejected")
    e.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    if case .rejectedJump = e.applyFix(node: (40, 0), confidence: 0.7, raw: (0, 0)) {
        check(true, "")
    } else { check(false, "40 m jump at conf 0.7 should be rejected") }
    check(e.applyFix(node: (40, 0), confidence: 0.9, raw: (0, 0)) == .accepted, "very confident jump accepted")
}

do {  // the first fix needs less confidence than later ones
    let e = MapPositionEstimator()
    check(e.applyFix(node: (0, 0), confidence: 0.35, raw: (0, 0)) == .accepted, "first fix at 0.35 accepted")
    check(e.applyFix(node: (1, 0), confidence: 0.35, raw: (0, 0)) == .rejectedLowConfidence,
          "later fix at 0.35 rejected")
    check(e.applyFix(node: (1, 0), confidence: 0.65, raw: (0, 0)) == .accepted, "later fix at 0.65 accepted")
}

do {  // starting at the entrance = a fix at the map origin
    let e = MapPositionEstimator()
    e.ingest(raw: (3, 4))
    check(e.applyFix(node: (0, 0), confidence: 1, raw: (3, 4)) == .accepted, "entrance start accepted")
    e.ingest(raw: (3, 9))
    check(approx(e.mapPosition!.x, 0) && approx(e.mapPosition!.y, 5), "walks on from the entrance")
}

do {  // heading offset learned: PDR thinks it walked north, map says east
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0))
    e.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...10 { e.ingest(raw: (0, Double(i))) }          // 10 m "north" by PDR
    e.applyFix(node: (10, 0), confidence: 0.9, raw: (0, 10)) // really 10 m east → +90° error
    // 90° exceeds the 45° per-fix plausibility gate → not learned
    check(approx(e.headingOffsetDeg, 0), "90° correction rejected as implausible")

    let f = MapPositionEstimator()
    f.ingest(raw: (0, 0))
    f.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...10 { f.ingest(raw: (0, Double(i))) }
    let east20 = (x: 10 * sin(20 * Double.pi / 180), y: 10 * cos(20 * Double.pi / 180))
    f.applyFix(node: east20, confidence: 0.9, raw: (0, 10))   // 20° clockwise error
    check(approx(f.headingOffsetDeg, 0.3 * 20, 1e-6), "heading offset moves 30% toward 20° (got \(f.headingOffsetDeg))")
    // next steps are rotated by the learned offset
    f.ingest(raw: (0, 11))
    let step = (x: f.mapPosition!.x - east20.x, y: f.mapPosition!.y - east20.y)
    check(approx(MapPositionEstimator.bearing(step), 6, 1e-6), "new step bearing = learned 6°")
}

do {  // step scale learned and clamped
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0))
    e.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...10 { e.ingest(raw: (0, Double(i))) }
    e.applyFix(node: (0, 12), confidence: 0.9, raw: (0, 10))   // walked 12 m, PDR said 10
    check(approx(e.stepScale, 1 + 0.3 * 0.2), "step scale 1.06 (got \(e.stepScale))")
    let g = MapPositionEstimator()
    g.ingest(raw: (0, 0))
    g.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...10 { g.ingest(raw: (0, Double(i))) }
    g.applyFix(node: (0, 30), confidence: 0.9, raw: (0, 10))   // absurd 3× ratio → clamped to 1.3
    check(approx(g.stepScale, 1 + 0.3 * 0.3), "ratio clamped before smoothing (got \(g.stepScale))")
}

do {  // short or wandering stretches don't calibrate
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0))
    e.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...4 { e.ingest(raw: (0, Double(i))) }
    e.applyFix(node: (0, 6), confidence: 0.9, raw: (0, 4))
    check(e.stepScale == 1 && e.headingOffsetDeg == 0, "< 8 m does not calibrate")

    let w = MapPositionEstimator()
    w.ingest(raw: (0, 0))
    w.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    // walk 10 m out and 6 m back sideways: raw path 20+ m, straight-line 10 m
    for i in 1...10 { w.ingest(raw: (Double(i), 0)) }
    for i in 1...10 { w.ingest(raw: (10, Double(i))) }
    w.applyFix(node: (12, 12), confidence: 0.9, raw: (10, 10))
    check(w.stepScale == 1 && w.headingOffsetDeg == 0, "L-shaped path does not calibrate")
}

do {  // rebase ignores a raw-frame jump
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0))
    e.applyFix(node: (5, 5), confidence: 0.9, raw: (0, 0))
    e.rebase(raw: (100, 100))
    e.ingest(raw: (100, 101))
    check(approx(e.mapPosition!.x, 5) && approx(e.mapPosition!.y, 6), "rebase does not integrate the jump")
}

do {  // geometry helpers
    check(approx(MapPositionEstimator.bearing((1, 0)), 90), "east = 90°")
    let r = MapPositionEstimator.rotate((0, 1), byDegrees: 90)
    check(approx(r.x, 1) && approx(r.y, 0, 1e-9), "rotate north by 90° → east")
    check(approx(MapPositionEstimator.wrap(270), -90) && approx(MapPositionEstimator.wrap(-190), 170), "wrap")
    let e = MapPositionEstimator()
    check(approx(e.mapHeading(fromRaw: 350), 350), "no offset")
}

// MARK: - PhotoQualityChecker

func makeImage(width: Int, height: Int, pixel: (Int, Int) -> UInt8) -> CGImage {
    var px = [UInt8](repeating: 0, count: width * height)
    for y in 0..<height { for x in 0..<width { px[y * width + x] = pixel(x, y) } }
    let provider = CGDataProvider(data: Data(px) as CFData)!
    return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                   space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
}

do {
    var rng = SystemRandomNumberGenerator()
    let noisy = makeImage(width: 300, height: 400) { _, _ in UInt8.random(in: 0...255, using: &rng) }
    let m = PhotoQualityChecker.measure(noisy)!
    check(m.sharpness > PhotoQualityChecker.minSharpness, "textured image is sharp (\(m.sharpness))")
    check(PhotoQualityChecker.hint(for: m) == nil, "textured image passes")

    let flat = makeImage(width: 300, height: 400) { _, _ in 128 }
    check(PhotoQualityChecker.hint(for: PhotoQualityChecker.measure(flat)!) == PhotoQualityChecker.blurHint,
          "flat image is 'blurred'")

    let dark = makeImage(width: 300, height: 400) { _, _ in UInt8.random(in: 0...40, using: &rng) }
    check(PhotoQualityChecker.hint(for: PhotoQualityChecker.measure(dark)!) == PhotoQualityChecker.darkHint,
          "dark image asks for flash")

    let gradient = makeImage(width: 1000, height: 800) { x, _ in UInt8(x * 255 / 999) }
    check(PhotoQualityChecker.measure(gradient)!.sharpness < 5, "smooth gradient has ~no edges")
}

do {  // tilt
    check(approx(PhotoQualityChecker.tiltDegrees(gravity: (0, -1, 0)), 0), "upright = 0°")
    check(approx(PhotoQualityChecker.tiltDegrees(gravity: (0, 0, -1)), 90), "flat on table = 90°")
    let s = sin(20 * Double.pi / 180), c = cos(20 * Double.pi / 180)
    check(approx(PhotoQualityChecker.tiltDegrees(gravity: (0, -c, -s)), 20, 1e-6), "20° forward tilt")
    check(PhotoQualityChecker.isUpright(gravity: (0, -c, -s)), "20° is acceptable")
    check(!PhotoQualityChecker.isUpright(gravity: (0, -0.5, -0.87)), "60° (pointing at floor) is not")
}

print("\(passed) passed, \(failures) failed")
exit(failures == 0 ? 0 : 1)
