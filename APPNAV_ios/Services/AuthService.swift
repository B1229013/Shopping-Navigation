import Foundation
import FirebaseAuth
import Combine

@MainActor
final class AuthService: ObservableObject {
    static let shared = AuthService()
    @Published var currentUser: User? = Auth.auth().currentUser
    @Published var isLoggedIn: Bool = Auth.auth().currentUser != nil

    private var handle: AuthStateDidChangeListenerHandle?

    init() {
        handle = Auth.auth().addStateDidChangeListener { [weak self] _, user in
            Task { @MainActor in
                self?.currentUser = user
                self?.isLoggedIn = user != nil
            }
        }
    }

    func signIn(email: String, password: String) async throws {
        let result = try await Auth.auth().signIn(withEmail: email, password: password)
        self.currentUser = result.user
        self.isLoggedIn = true
    }

    func signUp(email: String, password: String) async throws {
        let result = try await Auth.auth().createUser(withEmail: email, password: password)
        self.currentUser = result.user
        self.isLoggedIn = true
    }

    func signOut() {
        try? Auth.auth().signOut()
        currentUser = nil
        isLoggedIn = false
    }

    var userEmail: String {
        currentUser?.email ?? "使用者"
    }

    var displayName: String {
        userEmail.components(separatedBy: "@").first ?? "使用者"
    }
}
