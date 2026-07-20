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
}
