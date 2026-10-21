import Foundation

// Read API keys from Info.plist (set via Xcode build settings or xcconfig)
enum AppConfig {
    static var groqApiKey: String {
        Bundle.main.object(forInfoDictionaryKey: "GROQ_API_KEY") as? String ?? ""
    }

    static var paddleOcrApiUrl: String {
        Bundle.main.object(forInfoDictionaryKey: "PADDLEOCR_API_URL") as? String ?? ""
    }

    static var paddleOcrToken: String {
        Bundle.main.object(forInfoDictionaryKey: "PADDLEOCR_ACCESS_TOKEN") as? String ?? ""
    }

    static var mapsApiKey: String {
        Bundle.main.object(forInfoDictionaryKey: "MAPS_API_KEY") as? String ?? ""
    }

    static var navigationBackendURL: String {
        Bundle.main.object(forInfoDictionaryKey: "NAVIGATION_BACKEND_URL") as? String ?? ""
    }

    /// Run DINOv2 on the phone (Core ML) and upload its top matches. Off unless the
    /// ON_DEVICE_DINO build setting is YES: the server runs DINOv2 on every uploaded photo,
    /// the same path as Android — better on the lab1019 test walk (Top-1 44.7 % vs 37.5 %),
    /// in every store, and with no reference data to bundle into the app.
    static var onDeviceDINO: Bool {
        let v = (Bundle.main.object(forInfoDictionaryKey: "ON_DEVICE_DINO") as? String ?? "").uppercased()
        return v == "YES" || v == "1" || v == "TRUE"
    }
}
