import SwiftUI

struct AIView: View {
    let shoppingItems: [ShoppingItem]
    let dietRecords: [DietRecord]
    let budgetTotal: Int

    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isLoading = false

    private let groqApiKey = UserDefaults.standard.string(forKey: "groqApiKey") ?? AppConfig.groqApiKey

    struct ChatMessage: Identifiable {
        let id = UUID()
        let role: String
        let content: String
        let timestamp = Date()
    }

    private let suggestions = [
        "分析我的購物習慣",
        "本月預算使用情況",
        "建議省錢的購物方式",
        "今天應該吃什麼？"
    ]

    var body: some View {
        VStack(spacing: 0) {
            headerView
            chatArea
            inputArea
        }
        .adaptiveWidth(760)
        .background(Color.appBg)
    }

    // MARK: - Header
    private var headerView: some View {
        HStack(spacing: 12) {
            ZStack {
                Circle().fill(Color.appAccent.opacity(0.15)).frame(width: 44, height: 44)
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.appAccent)
                    .font(.title3)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text("AI 購物助理")
                    .font(.headline.bold())
                    .foregroundColor(.appTextPrimary)
                Text("Powered by Groq LLaMA")
                    .font(.caption2)
                    .foregroundColor(.appTextTertiary)
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 12)
    }

    // MARK: - Chat Area
    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if messages.isEmpty {
                        welcomeSection
                    }
                    ForEach(messages) { msg in
                        MessageBubble(message: msg)
                            .id(msg.id)
                    }
                    if isLoading {
                        TypingIndicator()
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 8)
                .padding(.bottom, 20)
            }
            .onChange(of: messages.count) { _, _ in
                withAnimation {
                    proxy.scrollTo(messages.last?.id, anchor: .bottom)
                }
            }
        }
    }

    // MARK: - Welcome Section
    private var welcomeSection: some View {
        VStack(spacing: 16) {
            Spacer(minLength: 20)
            VStack(spacing: 8) {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 40))
                    .foregroundColor(.appAccent.opacity(0.6))
                Text("你好！我是 AI 購物助理")
                    .font(.headline)
                    .foregroundColor(.appTextPrimary)
                Text("我可以幫你分析購物習慣、建議省錢方法，或回答任何購物相關問題")
                    .font(.caption)
                    .foregroundColor(.appTextSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
            }

            VStack(spacing: 8) {
                Text("快速提問")
                    .font(.caption.bold())
                    .foregroundColor(.appTextTertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                ForEach(suggestions, id: \.self) { suggestion in
                    Button {
                        inputText = suggestion
                        sendMessage()
                    } label: {
                        HStack {
                            Image(systemName: "sparkles")
                                .font(.caption)
                                .foregroundColor(.appAccent)
                            Text(suggestion)
                                .font(.subheadline)
                                .foregroundColor(.appTextPrimary)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .foregroundColor(.appTextTertiary)
                        }
                        .padding(12)
                        .background(Color.appSurface)
                        .cornerRadius(12)
                    }
                }
            }
        }
    }

    // MARK: - Input Area
    private var inputArea: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 12) {
                TextField("輸入問題...", text: $inputText, axis: .vertical)
                    .lineLimit(1...4)
                    .padding(12)
                    .background(Color.appSurface)
                    .cornerRadius(12)
                    .foregroundColor(.appTextPrimary)

                Button {
                    sendMessage()
                } label: {
                    Image(systemName: isLoading ? "ellipsis" : "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundColor(inputText.trimmingCharacters(in: .whitespaces).isEmpty ? .appTextTertiary : .appAccent)
                }
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || isLoading)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(Color.appBg)
        }
    }

    // MARK: - Send Message
    private func sendMessage() {
        let trimmed = inputText.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, !isLoading else { return }

        let userMsg = ChatMessage(role: "user", content: trimmed)
        messages.append(userMsg)
        inputText = ""
        isLoading = true

        Task {
            do {
                let systemPrompt = buildSystemPrompt()
                var groqMessages: [GroqMessage] = [GroqMessage(role: "system", content: systemPrompt)]
                for msg in messages {
                    groqMessages.append(GroqMessage(role: msg.role, content: msg.content))
                }

                let reply = try await GroqService.shared.chat(messages: groqMessages, apiKey: groqApiKey)
                await MainActor.run {
                    messages.append(ChatMessage(role: "assistant", content: reply))
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    messages.append(ChatMessage(role: "assistant", content: "抱歉，發生錯誤：\(error.localizedDescription)"))
                    isLoading = false
                }
            }
        }
    }

    private func buildSystemPrompt() -> String {
        let pendingCount = shoppingItems.filter { !$0.isChecked }.count
        let purchasedCount = shoppingItems.filter { $0.isChecked }.count
        let totalSpent = shoppingItems.filter { $0.isChecked }.reduce(0) { $0 + $1.price * $1.qty }
        let pendingNames = shoppingItems.filter { !$0.isChecked }.prefix(10).map { $0.name }.joined(separator: ", ")

        return """
        你是智慧購物助理，幫助用戶管理購物清單和預算。請用繁體中文回答。

        用戶當前狀態：
        - 待購商品：\(pendingCount) 項 (\(pendingNames))
        - 已購買：\(purchasedCount) 項
        - 本月已支出：$\(totalSpent)
        - 月預算：$\(budgetTotal)
        - 飲食紀錄：\(dietRecords.count) 筆

        請提供實用、具體的建議。回答要簡潔清晰。
        """
    }
}

// MARK: - Sub Views

struct MessageBubble: View {
    let message: AIView.ChatMessage
    private var isUser: Bool { message.role == "user" }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isUser { Spacer(minLength: 60) }
            if !isUser {
                ZStack {
                    Circle().fill(Color.appAccent.opacity(0.1)).frame(width: 32, height: 32)
                    Image(systemName: "brain.head.profile").font(.caption).foregroundColor(.appAccent)
                }
            }
            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .font(.subheadline)
                    .foregroundColor(isUser ? .white : .appTextPrimary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(isUser ? Color.appAccent : Color.appSurface)
                    .cornerRadius(18)
                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundColor(.appTextTertiary)
            }
            if !isUser { Spacer(minLength: 60) }
        }
    }
}

struct TypingIndicator: View {
    @State private var phase = 0

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            ZStack {
                Circle().fill(Color.appAccent.opacity(0.1)).frame(width: 32, height: 32)
                Image(systemName: "brain.head.profile").font(.caption).foregroundColor(.appAccent)
            }
            HStack(spacing: 4) {
                ForEach(0..<3) { i in
                    Circle()
                        .fill(Color.appTextTertiary)
                        .frame(width: 8, height: 8)
                        .scaleEffect(phase == i ? 1.3 : 0.8)
                        .animation(.easeInOut(duration: 0.5).repeatForever().delay(Double(i) * 0.15), value: phase)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 14)
            .background(Color.appSurface)
            .cornerRadius(18)
            Spacer()
        }
        .onAppear { phase = 1 }
    }
}
