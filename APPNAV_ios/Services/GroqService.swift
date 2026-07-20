import Foundation

struct GroqMessage: Codable {
    let role: String
    let content: String
}

struct GroqRequest: Codable {
    let model: String
    let messages: [GroqMessage]
    let response_format: GroqResponseFormat?

    struct GroqResponseFormat: Codable {
        let type: String
    }
}

struct GroqResponse: Codable {
    let choices: [Choice]
    struct Choice: Codable {
        let message: GroqMessage
    }
}

struct TidiedReceiptItem: Codable {
    let name: String
    let cat: String
    let qty: Int
    let total_price: Double?
}

struct BudgetEntry: Codable {
    let store: String
    let timestamp: String
    let total_amount: Double
    let line_items: [TidiedReceiptItem]
}

struct TidiedReceiptResponse: Codable {
    let budget_entry: BudgetEntry
}

final class GroqService {
    static let shared = GroqService()
    private let baseURL = URL(string: "https://api.groq.com/openai/v1/chat/completions")!

    func parseReceipt(ocrText: String, apiKey: String) async throws -> TidiedReceiptResponse {
        let systemPrompt = """
            You are a specialized Data Extraction and Budget Logic Engine for the "Smart AI Shopping Assistant" app.
            Your sole responsibility is to process receipt/recipe images and output structured data for the Budget module.
            The input text is in Traditional Chinese (繁體中文), recognized via OCR.

            KNOWN RECEIPT TEMPLATES:
            A) 全聯福利中心 (PX Mart) - "TX" prefix = taxable, strip it. Date: YYYY/MM/DD
            B) 家樂福 (Carrefour) - "N" suffix = non-taxable. Minguo year: add 1911
            C) Generic receipts

            EXTRACTION RULES:
            - Date: Convert Minguo year (e.g. 114) to Western (2025) by adding 1911. Format: YYYY/MM/DD.
            - Prices: Strip "TX", "$", "N" symbols.
            - Multiplier: Look for * or × symbol. Calculate total_price = unit_price × qty.
            - Keep item names in Traditional Chinese.
            - Do NOT include subtotal, tax, change, or payment method lines.

            CATEGORIZATION: Food, Beverages, Groceries, Other

            Return ONLY:
            {"budget_entry":{"store":"Store Name","timestamp":"YYYY/MM/DD","total_amount":0.0,"line_items":[{"name":"Item","cat":"Food","qty":1,"total_price":0.0}]}}
            """

        let body = GroqRequest(
            model: "llama-3.3-70b-versatile",
            messages: [
                GroqMessage(role: "system", content: systemPrompt),
                GroqMessage(role: "user", content: "OCR Text (Traditional Chinese):\n\(ocrText)")
            ],
            response_format: GroqRequest.GroqResponseFormat(type: "json_object")
        )

        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        request.timeoutInterval = 30

        let (data, _) = try await URLSession.shared.data(for: request)
        let groqResponse = try JSONDecoder().decode(GroqResponse.self, from: data)
        guard let content = groqResponse.choices.first?.message.content,
              let contentData = content.data(using: .utf8) else {
            throw URLError(.badServerResponse)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys
        return try decoder.decode(TidiedReceiptResponse.self, from: contentData)
    }

    func chat(messages: [GroqMessage], apiKey: String) async throws -> String {
        let body = GroqRequest(
            model: "llama-3.3-70b-versatile",
            messages: messages,
            response_format: nil
        )

        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        request.timeoutInterval = 30

        let (data, _) = try await URLSession.shared.data(for: request)
        let response = try JSONDecoder().decode(GroqResponse.self, from: data)
        return response.choices.first?.message.content ?? ""
    }
}
