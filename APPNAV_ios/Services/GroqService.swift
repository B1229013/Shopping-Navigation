import Foundation

struct GroqMessage: Codable {
    let role: String
    /// nil when the model refuses ("content": null); requests always set it.
    let content: String?
}

struct GroqRequest: Codable {
    let model: String
    let messages: [GroqMessage]
    let response_format: GroqResponseFormat?
    struct GroqResponseFormat: Codable { let type: String }
}

struct GroqResponse: Codable {
    let choices: [Choice]
    struct Choice: Codable { let message: GroqMessage }
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

/// Structured result of reading a food-package photo: ingredients text separated
/// from the nutrition-facts numbers (per serving where available).
struct NutritionParse: Codable {
    var name: String?
    var ingredients: String?
    var calorie: Double?
    var protein: Double?
    var fat: Double?
    var saturatedFat: Double?
    var transFat: Double?
    var carbs: Double?
    var sugar: Double?
    var sodium: Double?
    var sugarAlcohol: Double?
}

/// All LLM calls go through an OpenAI-compatible /chat/completions endpoint,
/// configured in AppConfig (currently OpenAI).
final class GroqService {
    /// A short message for a failed LLM call — shown instead of failing silently.
    static func userMessage(for error: Error) -> String {
        if AppConfig.groqApiKey.isEmpty { return "尚未設定 API 金鑰，請到「設定」輸入" }
        return "AI 呼叫失敗：\(error.localizedDescription)"
    }

    static let shared = GroqService()

    private var endpoint: URL { URL(string: AppConfig.llmBaseURL)! }
    private var model: String { AppConfig.llmModel }

    private func send(_ messages: [GroqMessage], apiKey: String) async throws -> String {
        let body = GroqRequest(model: model, messages: messages, response_format: nil)
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        request.timeoutInterval = 60

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, !(200...299).contains(http.statusCode) {
            let detail = String(data: data, encoding: .utf8) ?? ""
            throw NSError(domain: "LLM", code: http.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "伺服器回應 \(http.statusCode)：\(detail.prefix(180))"])
        }
        let decoded = try JSONDecoder().decode(GroqResponse.self, from: data)
        return decoded.choices.first?.message.content ?? ""
    }

    /// Pull the outermost {...} JSON object out of a reply that may be fenced or explained.
    private func extractJSON(_ text: String) -> Data? {
        guard let start = text.firstIndex(of: "{"),
              let end = text.lastIndex(of: "}"),
              start < end else { return nil }
        return String(text[start...end]).data(using: .utf8)
    }

    func chat(messages: [GroqMessage], apiKey: String) async throws -> String {
        try await send(messages, apiKey: apiKey)
    }

    func parseReceipt(ocrText: String, apiKey: String) async throws -> TidiedReceiptResponse {
        let systemPrompt = """
            You are a Data Extraction and Budget Logic Engine for a shopping assistant app.
            Process receipt OCR text (Traditional Chinese) into structured budget data.

            KNOWN RECEIPT TEMPLATES:
            A) 全聯福利中心 (PX Mart) - "TX" prefix = taxable, strip it. Date: YYYY/MM/DD
            B) 家樂福 (Carrefour) - "N" suffix = non-taxable. Minguo year: add 1911
            C) Generic receipts

            RULES:
            - Date: convert Minguo year (e.g. 114) to Western by adding 1911. Format YYYY/MM/DD.
            - Strip "TX", "$", "N" symbols from prices.
            - Multiplier: look for * or × ; total_price = unit_price × qty.
            - Keep item names in Traditional Chinese.
            - Do NOT include subtotal, tax, change, or payment-method lines.
            CATEGORIZATION: Food, Beverages, Groceries, Other

            Return ONLY this JSON (no markdown, no explanation):
            {"budget_entry":{"store":"Store Name","timestamp":"YYYY/MM/DD","total_amount":0.0,"line_items":[{"name":"Item","cat":"Food","qty":1,"total_price":0.0}]}}
            """
        let content = try await send([
            GroqMessage(role: "system", content: systemPrompt),
            GroqMessage(role: "user", content: "OCR Text (Traditional Chinese):\n\(ocrText)")
        ], apiKey: apiKey)
        guard let data = extractJSON(content) else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(TidiedReceiptResponse.self, from: data)
    }

    /// Reads OCR text from a food package photo (which mixes 成分/原料, 營養標示 and noise)
    /// and separates the ingredients from the per-serving nutrition numbers.
    func parseNutritionLabel(ocrText: String, apiKey: String) async throws -> NutritionParse {
        let systemPrompt = """
            你是食品包裝資料擷取引擎。輸入是對食品包裝拍照後 OCR 得到的繁體中文文字，
            可能同時包含「成分（原料）」、「營養標示」與其他雜訊文字。
            請判斷哪些屬於成分、哪些是營養標示數值，分別整理，不要把全部文字塞在一起。
            "ingredients" 只放成分內容本身（例如：水、麵粉、糖、鹽），不要包含「成分」「成份」「原料」等標題字或冒號；產品名稱放在 "name"。
            營養標示取「每份」的數值（若只有每100克就用每100克）。
            單位：熱量 kcal(大卡)；蛋白質/脂肪/飽和脂肪/反式脂肪/碳水化合物/糖/糖醇 用公克(g)；鈉 用毫克(mg)。
            找不到的欄位給 null。只輸出 JSON，不要任何解釋或 markdown：
            {"name":"產品名稱或null","ingredients":"成分文字","calorie":0,"protein":0,"fat":0,"saturatedFat":0,"transFat":0,"carbs":0,"sugar":0,"sodium":0,"sugarAlcohol":0}
            """
        let content = try await send([
            GroqMessage(role: "system", content: systemPrompt),
            GroqMessage(role: "user", content: ocrText)
        ], apiKey: apiKey)
        guard let data = extractJSON(content) else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(NutritionParse.self, from: data)
    }

    /// Classify a single item name into one of the four category ids. Used as a
    /// fallback when local keyword matching returns "Other".
    func classifyItem(name: String, apiKey: String) async throws -> String {
        let sys = """
            把商品分到四類之一，只回其中一個英文字，不要任何其他文字：
            Food（食物：肉、菜、蛋、魚、水果、米麵、零食等），
            Beverages（飲料），
            Groceries（生活用品、清潔用品、調味料），
            Other（都不是）。
            """
        let content = try await send([
            GroqMessage(role: "system", content: sys),
            GroqMessage(role: "user", content: name)
        ], apiKey: apiKey)
        return normalizeCategory(content, name)
    }
}
