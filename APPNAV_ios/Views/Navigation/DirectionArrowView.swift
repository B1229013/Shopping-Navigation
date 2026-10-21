import SwiftUI

/// Floating AR arrow that points toward the target node.
/// Gray when not localized, cyan/blue gradient when localized.
struct DirectionArrowView: View {
    let relativeBearingDegrees: Double
    let isLocalized: Bool

    var body: some View {
        Image(systemName: "location.north.fill")
            .font(.system(size: 36, weight: .bold))
            .foregroundStyle(
                isLocalized
                ? AnyShapeStyle(.linearGradient(
                    colors: [.cyan, .blue],
                    startPoint: .top,
                    endPoint: .bottom
                ))
                : AnyShapeStyle(Color.white.opacity(0.5))
            )
            .shadow(color: isLocalized ? .cyan.opacity(0.6) : .clear, radius: 8)
            .rotationEffect(.degrees(relativeBearingDegrees))
            .animation(.easeInOut(duration: 0.3), value: relativeBearingDegrees)
    }
}
