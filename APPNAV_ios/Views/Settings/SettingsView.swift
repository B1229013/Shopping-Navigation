import SwiftUI

struct SettingsView: View {
    @ObservedObject private var auth = AuthService.shared
    @Environment(\.dismiss) private var dismiss
    @AppStorage("groqApiKey") private var groqApiKey = AppConfig.groqApiKey
    @AppStorage("paddleOcrApiUrl") private var paddleOcrApiUrl = AppConfig.paddleOcrApiUrl
    @AppStorage("paddleOcrToken") private var paddleOcrToken = AppConfig.paddleOcrToken
    @AppStorage("navigationBackendURL") private var navigationBackendURL = AppConfig.navigationBackendURL
    @AppStorage("profileGender") private var profileGender = "其他"
    @AppStorage("profileHeight") private var profileHeight = ""
    @AppStorage("profileWeight") private var profileWeight = ""
    @AppStorage("profileAllergies") private var profileAllergies = ""
    @State private var showSignOutAlert = false
    @State private var showSensorTest = false
    @State private var showTopoMapTest = false
    @State private var showSensorComboTest = false
    @State private var showGPSLogger = false

    var body: some View {
        NavigationView {
            List {
                // User info
                Section {
                    HStack(spacing: 14) {
                        ZStack {
                            Circle()
                                .fill(Color.appAccent.opacity(0.15))
                                .frame(width: 56, height: 56)
                            Text(auth.displayName.prefix(1).uppercased())
                                .font(.title2.bold())
                                .foregroundColor(.appAccent)
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            Text(auth.displayName)
                                .font(.headline)
                                .foregroundColor(.appTextPrimary)
                            Text(auth.userEmail)
                                .font(.caption)
                                .foregroundColor(.appTextSecondary)
                        }
                    }
                    .padding(.vertical, 6)
                }

                // Personal profile (used by 食物分析: 過敏原提醒等)
                Section("個人資料（選填）") {
                    Picker("性別", selection: $profileGender) {
                        Text("男").tag("男")
                        Text("女").tag("女")
                        Text("其他").tag("其他")
                    }
                    HStack {
                        Text("身高")
                        Spacer()
                        TextField("cm", text: $profileHeight)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                        Text("cm").foregroundColor(.appTextTertiary)
                    }
                    HStack {
                        Text("體重")
                        Spacer()
                        TextField("kg", text: $profileWeight)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                        Text("kg").foregroundColor(.appTextTertiary)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("過敏原（用逗號分隔，如：牛奶, 花生）")
                            .font(.caption)
                            .foregroundColor(.appTextSecondary)
                        TextField("過敏原", text: $profileAllergies, axis: .vertical)
                            .lineLimit(2...4)
                    }
                }

                // API Configuration
                Section("API 設定") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("OpenAI API 金鑰")
                            .font(.caption)
                            .foregroundColor(.appTextSecondary)
                        SecureField("輸入 OpenAI API 金鑰", text: $groqApiKey)
                            .font(.caption)
                            .autocapitalization(.none)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("PaddleOCR API URL")
                            .font(.caption)
                            .foregroundColor(.appTextSecondary)
                        TextField("輸入 PaddleOCR URL", text: $paddleOcrApiUrl)
                            .font(.caption)
                            .autocapitalization(.none)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("PaddleOCR Token")
                            .font(.caption)
                            .foregroundColor(.appTextSecondary)
                        SecureField("輸入 PaddleOCR Token", text: $paddleOcrToken)
                            .font(.caption)
                            .autocapitalization(.none)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("導航後端網址")
                            .font(.caption)
                            .foregroundColor(.appTextSecondary)
                        TextField("例如 http://192.168.1.5:8000", text: $navigationBackendURL)
                            .font(.caption)
                            .autocapitalization(.none)
                            .keyboardType(.URL)
                    }
                }

                // Debug tools
                Section("除錯工具") {
                    Button {
                        showSensorTest = true
                    } label: {
                        HStack {
                            Image(systemName: "figure.walk")
                            Text("計步器測試")
                        }
                    }
                    Button {
                        showTopoMapTest = true
                    } label: {
                        HStack {
                            Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                            Text("拓樸地圖測試")
                        }
                    }
                    Button {
                        showSensorComboTest = true
                    } label: {
                        HStack {
                            Image(systemName: "arrow.triangle.branch")
                            Text("感測器組合測試")
                        }
                    }
                    Button {
                        showGPSLogger = true
                    } label: {
                        HStack {
                            Image(systemName: "location.fill")
                            Text("GPS 記錄")
                        }
                    }
                }

                // App info
                Section("關於") {
                    LabeledContent("版本", value: "1.0.0 (iOS)")
                    LabeledContent("OCR 引擎", value: "iOS Vision + PaddleOCR")
                    LabeledContent("AI 模型", value: AppConfig.llmModel)
                }

                // Sign out
                Section {
                    Button(role: .destructive) {
                        showSignOutAlert = true
                    } label: {
                        HStack {
                            Image(systemName: "rectangle.portrait.and.arrow.right")
                            Text("登出")
                        }
                    }
                }
            }
            .navigationTitle("設定")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("完成") { dismiss() }
                }
            }
            .alert("確認登出", isPresented: $showSignOutAlert) {
                Button("登出", role: .destructive) {
                    auth.signOut()
                    dismiss()
                }
                Button("取消", role: .cancel) {}
            } message: {
                Text("確定要登出帳號嗎？")
            }
            .sheet(isPresented: $showSensorTest) {
                SensorTestView()
            }
            .sheet(isPresented: $showTopoMapTest) {
                TopoMapTestView()
            }
            .sheet(isPresented: $showSensorComboTest) {
                SensorComboTestView()
            }
            .sheet(isPresented: $showGPSLogger) {
                GPSLoggerView()
            }
        }
    }
}
