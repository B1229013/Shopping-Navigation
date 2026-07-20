import SwiftUI
import Foundation

// MARK: - Color helpers
extension Color {
    static let appBg = Color(hex: 0xECFDF5)
    static let appAccent = Color(hex: 0x059669)
    static let appSurface = Color.white
    static let appBorder = Color(hex: 0xE1F2ED)
    static let appTextPrimary = Color(hex: 0x0F172A)
    static let appTextSecondary = Color(hex: 0x475569)
    static let appTextTertiary = Color(hex: 0x94A3B8)
    static let appGold = Color(hex: 0xD97706)
    static let appSuccess = Color(hex: 0x059669)
    static let appDanger = Color(hex: 0xEF4444)
    static let chartBlue = Color(hex: 0x3B82F6)
    static let chartGreen = Color(hex: 0x10B981)
    static let chartAmber = Color(hex: 0xF59E0B)
    static let chartViolet = Color(hex: 0x8B5CF6)

    init(hex: UInt, alpha: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xff) / 255,
            green: Double((hex >> 08) & 0xff) / 255,
            blue: Double((hex >> 00) & 0xff) / 255,
            opacity: alpha
        )
    }
}

// MARK: - Date helpers
extension Date {
    var millisecondsSince1970: Double { timeIntervalSince1970 * 1000 }

    static func from(milliseconds: Double) -> Date {
        Date(timeIntervalSince1970: milliseconds / 1000)
    }

    var shortDateString: String {
        let f = DateFormatter()
        f.dateFormat = "yyyy/MM/dd"
        return f.string(from: self)
    }

    var monthDayString: String {
        let f = DateFormatter()
        f.dateFormat = "M月d日"
        f.locale = Locale(identifier: "zh_TW")
        return f.string(from: self)
    }

    var yearMonthString: String {
        let f = DateFormatter()
        f.dateFormat = "yyyy年M月"
        f.locale = Locale(identifier: "zh_TW")
        return f.string(from: self)
    }

    var monthKey: String {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM"
        return f.string(from: self)
    }

    var greeting: String {
        let hour = Calendar.current.component(.hour, from: self)
        if hour < 12 { return "早安" }
        if hour < 18 { return "午安" }
        return "晚安"
    }
}

// MARK: - Number formatting
extension Int {
    var formattedWithCommas: String {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f.string(from: NSNumber(value: self)) ?? "\(self)"
    }
}

// MARK: - View modifiers
struct CardStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(Color.appSurface)
            .cornerRadius(16)
            .shadow(color: .black.opacity(0.04), radius: 8, x: 0, y: 2)
    }
}

extension View {
    func cardStyle() -> some View { modifier(CardStyle()) }

    /// Caps content width and centers it on wide/regular-width screens (iPad, iPhone landscape)
    /// while leaving compact-width devices (iPhone portrait) completely unaffected.
    func adaptiveWidth(_ maxWidth: CGFloat = 700) -> some View {
        self
            .frame(maxWidth: maxWidth)
            .frame(maxWidth: .infinity)
    }
}
