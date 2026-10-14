// Simulated walk through MapPositionEstimator (the phone's real correction code).
// Build: swiftc -O -o walk_sim walk_sim.swift ../../APPNAV_ios/Navigation/MapPositionEstimator.swift
// (rename to main.swift first if swiftc complains about top-level code; used by
// backend/eval/report_figures.py --walk-sim)
// stdin: route vertices "x,y" per line, then "FIX i" lines = photo fixes at vertex i.
// args: headingBiasDeg stepScale seed
// stdout CSV: step,true_x,true_y,raw_x,raw_y,est_x,est_y,fix,heading_offset,step_scale
import Foundation

struct LCG { var s: UInt64; mutating func next() -> Double { s = s &* 6364136223846793005 &+ 1442695040888963407; return Double(s >> 11) / Double(1 << 53) } 
    mutating func gauss() -> Double { let u = max(next(), 1e-12), v = next(); return (-2 * log(u)).squareRoot() * cos(2 * .pi * v) } }

let args = CommandLine.arguments
let bias = Double(args.count > 1 ? args[1] : "10")!
let scale = Double(args.count > 2 ? args[2] : "0.9")!
var rng = LCG(s: UInt64(args.count > 3 ? args[3] : "7")!)
var verts: [(Double, Double)] = []
var fixes = Set<Int>()
while let line = readLine() {
    if line.hasPrefix("FIX ") { fixes.insert(Int(line.dropFirst(4))!); continue }
    let p = line.split(separator: ",").map { Double($0)! }
    verts.append((p[0], p[1]))
}
let est = MapPositionEstimator()
var raw = (x: 0.0, y: 0.0)
est.ingest(raw: raw)
print("step,true_x,true_y,raw_x,raw_y,est_x,est_y,fix,heading_offset,step_scale")
var step = 0
func emit(_ t: (Double, Double), fix: Bool) {
    let e = est.mapPosition
    print("\(step),\(t.0),\(t.1),\(raw.x),\(raw.y),\(e.map { "\($0.x)" } ?? ""),\(e.map { "\($0.y)" } ?? ""),\(fix ? 1 : 0),\(est.headingOffsetDeg),\(est.stepScale)")
}
if fixes.contains(0) { est.applyFix(node: verts[0], confidence: 0.9, raw: raw) }
emit(verts[0], fix: fixes.contains(0))
for i in 1..<verts.count {
    let a = verts[i - 1], b = verts[i]
    let dx = b.0 - a.0, dy = b.1 - a.1, len = (dx * dx + dy * dy).squareRoot()
    let n = max(1, Int((len / 0.7).rounded()))
    let heading = atan2(dx, dy) * 180 / .pi
    for k in 1...n {
        let t = (a.0 + dx * Double(k) / Double(n), a.1 + dy * Double(k) / Double(n))
        let h = (heading - bias + rng.gauss() * 3) * .pi / 180   // PDR heading is off by `bias`
        let L = len / Double(n) * scale * (1 + rng.gauss() * 0.05)
        raw = (raw.x + L * sin(h), raw.y + L * cos(h))
        est.ingest(raw: raw)
        step += 1
        let fixHere = k == n && fixes.contains(i)
        if fixHere { est.applyFix(node: b, confidence: 0.9, raw: raw) }
        emit(t, fix: fixHere)
    }
}
