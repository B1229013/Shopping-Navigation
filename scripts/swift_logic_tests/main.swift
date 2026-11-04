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

do {  // the jump gate compares the node with the position when the photo was taken
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0))
    e.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...20 { e.ingest(raw: (0, Double(i))) }        // kept walking while the server answered
    check(e.applyFix(node: (0, 3), confidence: 0.7, raw: (0, 2)) == .accepted,
          "photo at 2 m localized 1 m off is accepted though the user is now 17 m further")
    e.ingest(raw: (0, 20))                                    // replay the steps since the shutter
    check(approx(e.mapPosition!.x, 0) && approx(e.mapPosition!.y, 21), "position = node + 18 m walked since")
    if case .rejectedJump = e.applyFix(node: (30, 21), confidence: 0.7, raw: (0, 20)) {
        check(true, "")
    } else { check(false, "a real 30 m jump is still rejected") }
}

do {  // fixes closer than 8 m extend the calibration stretch instead of restarting it
    let e = MapPositionEstimator()
    e.ingest(raw: (0, 0))
    e.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...5 { e.ingest(raw: (0, Double(i))) }
    e.applyFix(node: (0, 6), confidence: 0.9, raw: (0, 5))   // 6 m: too short, no calibration yet
    check(e.stepScale == 1, "6 m stretch does not calibrate")
    for i in 6...10 { e.ingest(raw: (0, Double(i))) }
    e.applyFix(node: (0, 12), confidence: 0.9, raw: (0, 10))  // 12 m from the first fix, PDR 10
    check(approx(e.stepScale, 1 + 0.3 * 0.2), "two short stretches calibrate together (got \(e.stepScale))")

    let w = MapPositionEstimator()                           // ...but not across a turn
    w.ingest(raw: (0, 0))
    w.applyFix(node: (0, 0), confidence: 0.9, raw: (0, 0))
    for i in 1...6 { w.ingest(raw: (0, Double(i))) }
    for i in 1...6 { w.ingest(raw: (Double(i), 6)) }
    w.applyFix(node: (6, 6), confidence: 0.9, raw: (6, 6))   // L-shaped: restart from here
    for i in 7...16 { w.ingest(raw: (Double(i), 6)) }
    w.applyFix(node: (18, 6), confidence: 0.9, raw: (16, 6)) // straight 12 m map / 10 m PDR east
    check(approx(w.stepScale, 1.06) && approx(w.headingOffsetDeg, 0), "restarted after the turn (got \(w.stepScale))")
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

// MARK: - PathFollower look-ahead (#3: the arrow on a route that doubles back)

do {  // U-shaped route: out along x = 0, back along x = 1.5
    let f = PathFollower()
    let route = [CGPoint(x: 0, y: 0), CGPoint(x: 0, y: 10), CGPoint(x: 1.5, y: 10), CGPoint(x: 1.5, y: 0)]
    f.setRoute(polyline: route, turns: [], pdr: .zero, keepAnchor: false)
    f.anchor(mapPoint: .zero, pdr: .zero)
    var st: FollowState?
    for y in stride(from: 0.0, through: 3.0, by: 0.5) {   // walk out, 0.8 m right of the line
        st = f.update(pdr: CGPoint(x: 0.8, y: y), headingDegrees: 0)
    }
    let ahead = f.point(atProgress: st!.progressM + 5)!
    check(approx(ahead.x, 0) && approx(ahead.y, 8), "look-ahead stays on the outbound leg (got \(ahead))")
    check(f.point(atProgress: 11.5 + 4)! == CGPoint(x: 1.5, y: 6), "4 m into the return leg")
    check(f.point(atProgress: 99)! == route.last!, "past the end clamps to the last point")
    check(f.point(atProgress: -3)! == route.first!, "before the start clamps to the first point")
}

do {  // walking the whole U still advances through both turns to the end
    let f = PathFollower()
    let route = [CGPoint(x: 0, y: 0), CGPoint(x: 0, y: 10), CGPoint(x: 1.5, y: 10), CGPoint(x: 1.5, y: 0)]
    f.setRoute(polyline: route, turns: [], pdr: .zero, keepAnchor: false)
    var walk: [CGPoint] = stride(from: 0.0, through: 10.0, by: 0.5).map { CGPoint(x: 0, y: $0) }
    walk += [CGPoint(x: 0.75, y: 10), CGPoint(x: 1.5, y: 10)]
    walk += stride(from: 9.5, through: 0.0, by: -0.5).map { CGPoint(x: 1.5, y: $0) }
    var st: FollowState?
    var monotonic = true
    for p in walk {
        let prev = st?.progressM ?? 0
        st = f.update(pdr: p, headingDegrees: 0)
        if st!.progressM + 1e-9 < prev { monotonic = false }
    }
    check(monotonic, "progress never goes backwards on the way round")
    check(approx(st!.progressM, 21.5) && st!.arrived, "reaches the end (progress \(st!.progressM))")
}

// MARK: - Budget tab expense edits (#4)

do {  // editing a shopping-list item keeps it on the list
    var listItem = ShoppingItem(name: "牛奶", qty: 1, price: 0, isChecked: true, createdAt: 1000)
    listItem.dueDate = 5000
    let edited = ShoppingItem.expense(editing: listItem, name: "鮮奶", qty: 2, price: 95,
                                      store: "", purchasedAt: 3000, category: "Other")
    check(edited.source == nil && edited.id == listItem.id, "still a shopping-list item")
    check(edited.dueDate == 5000 && edited.createdAt == 1000, "keeps due date and creation time")
    check(edited.name == "鮮奶" && edited.qty == 2 && edited.price == 95 && edited.purchasedAt == 3000,
          "takes the edited fields")
    check(edited.location == "Other", "the user's own category, Other included")
    let new = ShoppingItem.expense(editing: nil, name: "餅乾", qty: 1, price: 30,
                                   store: "全聯", purchasedAt: 3000, category: "Food")
    check(new.source == "budget" && new.storeName == "全聯" && new.createdAt == 3000, "a new expense is a budget item")
}

// MARK: - CategoryClassifier (#5: the head of the name decides, not any substring)

do {
    let cases: [(String, String)] = [
        ("watermelon", CAT_FOOD), ("茶葉蛋", CAT_FOOD), ("抹茶餅乾", CAT_FOOD), ("可樂果", CAT_FOOD),
        ("牛奶糖", CAT_FOOD), ("鮮奶油", CAT_FOOD),
        ("菜瓜布", CAT_GROCERIES), ("麵粉", CAT_GROCERIES), ("番茄醬", CAT_GROCERIES), ("沙拉油", CAT_GROCERIES),
        ("牛奶", CAT_BEVERAGES), ("珍珠奶茶", CAT_BEVERAGES), ("香蕉牛奶", CAT_BEVERAGES), ("蘋果汁", CAT_BEVERAGES),
        ("礦泉水", CAT_BEVERAGES), ("water", CAT_BEVERAGES), ("麥茶", CAT_BEVERAGES),
        ("牛奶餅乾", CAT_FOOD), ("雞肉便當", CAT_FOOD), ("衛生紙", CAT_GROCERIES), ("螺絲起子", CAT_OTHER),
    ]
    for (name, want) in cases {
        check(guessCategory(from: name) == want, "\(name) → \(want), got \(guessCategory(from: name))")
    }
    // the receipt LLM's category stands unless the name really reads as a drink
    check(normalizeCategory("Food", "茶葉蛋") == CAT_FOOD, "LLM Food kept for 茶葉蛋")
    check(normalizeCategory("Food", "鮮奶") == CAT_BEVERAGES, "milk filed as Food is still a drink")
}

// MARK: - LLM errors the user can read (#6)

do {
    UserDefaults.standard.removeObject(forKey: "groqApiKey")
    let noKey = GroqService.userMessage(for: URLError(.userAuthenticationRequired))
    check(noKey.contains("設定"), "no key → points to Settings (\(noKey))")
    UserDefaults.standard.set("test-key", forKey: "groqApiKey")
    let failed = GroqService.userMessage(for: NSError(domain: "LLM", code: 429,
                                                     userInfo: [NSLocalizedDescriptionKey: "HTTP 429"]))
    check(failed.contains("HTTP 429"), "other errors say what happened (\(failed))")
    UserDefaults.standard.removeObject(forKey: "groqApiKey")
    // a refusal comes back with "content": null — it must still decode
    let json = #"{"choices":[{"message":{"role":"assistant","content":null}}]}"#.data(using: .utf8)!
    check((try? JSONDecoder().decode(GroqResponse.self, from: json)) != nil, "null content decodes")
}

print("\(passed) passed, \(failures) failed")
exit(failures == 0 ? 0 : 1)
