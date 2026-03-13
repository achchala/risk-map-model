//
//  RouteService.swift
//  RiskMapApp
//
//  Calls backend /api/routes/safety-aware and /api/risk-definition
//

import Foundation
import CoreLocation

class RouteService: ObservableObject {
    @Published var safetyAwareResponse: SafetyAwareResponse?
    @Published var riskDefinition: RiskDefinitionResponse?
    @Published var isLoading = false
    @Published var errorMessage: String?

    #if targetEnvironment(simulator)
    private let baseURL = "http://localhost:8000/api"
    #else
    private let baseURL = "http://10.10.8.221:8000/api" // Replace with your Mac's IP (ipconfig getifaddr en0)
    #endif

    func fetchSafetyAwareRoutes(
        origin: CLLocationCoordinate2D,
        destination: CLLocationCoordinate2D,
        beta: Double,
        weather: [String: Any]? = nil,
        timeOfDay: [String: Any]? = nil
    ) async throws -> SafetyAwareResponse {
        await MainActor.run { isLoading = true; errorMessage = nil }
        defer { Task { @MainActor in isLoading = false } }

        let url = URL(string: "\(baseURL)/routes/safety-aware")!
        var body: [String: Any] = [
            "origin": ["latitude": origin.latitude, "longitude": origin.longitude],
            "destination": ["latitude": destination.latitude, "longitude": destination.longitude],
            "beta": beta,
        ]
        if let w = weather { body["weather"] = w }
        if let t = timeOfDay { body["time_of_day"] = t }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.timeoutInterval = 60

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIError.serverError("Invalid response")
        }
        guard http.statusCode == 200 else {
            let msg = String(data: data, encoding: .utf8) ?? "Status \(http.statusCode)"
            await MainActor.run { errorMessage = msg }
            throw APIError.serverError(msg)
        }

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(SafetyAwareResponse.self, from: data)
        await MainActor.run {
            safetyAwareResponse = decoded
            errorMessage = nil
        }
        return decoded
    }

    func fetchRiskDefinition() async throws -> RiskDefinitionResponse {
        let url = URL(string: "\(baseURL)/risk-definition")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 10

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.serverError("Risk definition unavailable")
        }
        let decoded = try JSONDecoder().decode(RiskDefinitionResponse.self, from: data)
        await MainActor.run { riskDefinition = decoded }
        return decoded
    }
}
