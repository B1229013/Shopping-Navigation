import Foundation

final class PersistenceService {
    static let shared = PersistenceService()
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private func fileURL(name: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(name)
    }

    func load<T: Decodable>(_ type: T.Type, from fileName: String) -> T? {
        let url = fileURL(name: fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(type, from: data)
    }

    func save<T: Encodable>(_ value: T, to fileName: String) {
        guard let data = try? encoder.encode(value) else { return }
        try? data.write(to: fileURL(name: fileName), options: .atomic)
    }

    func loadString(from fileName: String) -> String? {
        let url = fileURL(name: fileName)
        return try? String(contentsOf: url, encoding: .utf8)
    }

    func saveString(_ value: String, to fileName: String) {
        let url = fileURL(name: fileName)
        try? value.write(to: url, atomically: true, encoding: .utf8)
    }
}
