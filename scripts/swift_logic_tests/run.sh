#!/bin/bash
# Compile the UI-free navigation logic together with its tests and run them (no Xcode needed).
set -e
cd "$(dirname "$0")"
SRC=../../APPNAV_ios/Navigation
OUT=$(mktemp -d)/swift_logic_tests
swiftc -O -o "$OUT" main.swift "$SRC/MapPositionEstimator.swift" "$SRC/PhotoQualityChecker.swift"
"$OUT"
