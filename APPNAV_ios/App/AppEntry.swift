import SwiftUI
import FirebaseCore

@main
struct AppNAV_iOSApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate

    var body: some Scene {
        WindowGroup {
            RootView()
        }
    }
}

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        FirebaseApp.configure()
        return true
    }
}

struct RootView: View {
    @StateObject private var auth = AuthService.shared
    @State private var isLoggedIn: Bool = false

    var body: some View {
        Group {
            if isLoggedIn {
                MainContainerView()
                    .transition(.opacity)
            } else {
                LoginView {
                    withAnimation(.easeInOut(duration: 0.3)) {
                        isLoggedIn = true
                    }
                }
                .transition(.opacity)
            }
        }
        .onReceive(auth.$isLoggedIn) { loggedIn in
            if !loggedIn { isLoggedIn = false }
        }
    }
}
