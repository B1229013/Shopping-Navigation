import SwiftUI

struct LoginView: View {
    let onLoginSuccess: () -> Void

    @StateObject private var auth = AuthService.shared
    @State private var email = ""
    @State private var password = ""
    @State private var isPasswordVisible = false
    @State private var isSignUpMode = false
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var appeared = false

    var body: some View {
        ZStack {
            Color.appBg.ignoresSafeArea()

            ScrollView {
                VStack(spacing: 0) {
                    Spacer(minLength: 40)

                    // Hero image placeholder
                    ZStack {
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.appAccent.opacity(0.1))
                        Image(systemName: "cart.fill")
                            .resizable()
                            .scaledToFit()
                            .frame(width: 80)
                            .foregroundColor(.appAccent)
                    }
                    .frame(height: 160)
                    .padding(.horizontal, 40)
                    .opacity(appeared ? 1 : 0)
                    .offset(y: appeared ? 0 : -20)
                    .animation(.spring(response: 0.6, dampingFraction: 0.8).delay(0.1), value: appeared)

                    VStack(spacing: 8) {
                        Text(isSignUpMode ? "建立帳號" : "智慧購物助理")
                            .font(.title.bold())
                            .foregroundColor(.appTextPrimary)
                        Text("AI 驅動的購物體驗")
                            .font(.subheadline)
                            .foregroundColor(.appTextSecondary)
                    }
                    .padding(.top, 24)
                    .opacity(appeared ? 1 : 0)
                    .animation(.spring(response: 0.6).delay(0.2), value: appeared)

                    VStack(spacing: 12) {
                        // Email field
                        HStack {
                            Image(systemName: "envelope")
                                .foregroundColor(.appTextSecondary)
                                .frame(width: 20)
                            TextField("Email", text: $email)
                                .keyboardType(.emailAddress)
                                .autocapitalization(.none)
                                .autocorrectionDisabled()
                        }
                        .padding(14)
                        .background(Color.appSurface)
                        .cornerRadius(14)

                        // Password field
                        HStack {
                            Image(systemName: "lock")
                                .foregroundColor(.appTextSecondary)
                                .frame(width: 20)
                            if isPasswordVisible {
                                TextField("密碼", text: $password)
                            } else {
                                SecureField("密碼", text: $password)
                            }
                            Button {
                                isPasswordVisible.toggle()
                            } label: {
                                Image(systemName: isPasswordVisible ? "eye.slash" : "eye")
                                    .foregroundColor(.appTextTertiary)
                            }
                        }
                        .padding(14)
                        .background(Color.appSurface)
                        .cornerRadius(14)

                        if let error = errorMessage {
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.appDanger)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        // Primary action button
                        Button {
                            Task { await handleAuth() }
                        } label: {
                            HStack {
                                if isLoading {
                                    ProgressView().tint(.white)
                                } else {
                                    Text(isSignUpMode ? "建立帳號" : "登入")
                                        .fontWeight(.bold)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 52)
                            .background(Color.appAccent)
                            .foregroundColor(.white)
                            .cornerRadius(14)
                        }
                        .disabled(isLoading)

                        // Skip button (debug)
                        Button {
                            onLoginSuccess()
                        } label: {
                            Text("測試用：直接進入 (跳過登入)")
                                .font(.subheadline)
                                .foregroundColor(.appTextSecondary)
                                .frame(maxWidth: .infinity)
                                .frame(height: 52)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 14)
                                        .stroke(Color.appAccent.opacity(0.3), lineWidth: 1)
                                )
                        }

                        Button {
                            withAnimation { isSignUpMode.toggle() }
                            errorMessage = nil
                        } label: {
                            Text(isSignUpMode ? "已有帳號？立即登入" : "沒有帳號？立即註冊")
                                .foregroundColor(.appAccent)
                                .font(.subheadline)
                        }
                        .padding(.top, 4)
                    }
                    .padding(.horizontal, 32)
                    .padding(.top, 32)
                    .opacity(appeared ? 1 : 0)
                    .animation(.spring(response: 0.6).delay(0.3), value: appeared)

                    Spacer(minLength: 40)
                }
                .adaptiveWidth(480)
            }
        }
        .onAppear { appeared = true }
    }

    private func handleAuth() async {
        guard !email.isEmpty, !password.isEmpty else {
            errorMessage = "請填寫所有欄位"
            return
        }
        isLoading = true
        errorMessage = nil
        do {
            if isSignUpMode {
                try await auth.signUp(email: email, password: password)
                isSignUpMode = false
                errorMessage = nil
            } else {
                try await auth.signIn(email: email, password: password)
                onLoginSuccess()
            }
        } catch {
            errorMessage = "失敗: \(error.localizedDescription)"
        }
        isLoading = false
    }
}
