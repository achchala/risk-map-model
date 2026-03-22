//
//  RiskService.swift
//  RiskMapApp
//
//  service for fetching risk predictions from backend API
//

import Foundation
import CoreLocation
import Combine

class RiskService: ObservableObject {
    @Published var roadSegments: [RoadSegment] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    /// Feature importance from model (for ranking risk drivers). Fetched with risk-definition.
    @Published var featureImportance: [String: Double]? = nil
    
    // Backend API URL: Simulator uses localhost; device must use your Mac's IP on the same Wi‑Fi.
    #if targetEnvironment(simulator)
    private let baseURL = "http://localhost:8000/api"
    #else
    // Your Mac's IP for device testing (same Wi‑Fi as iPhone)
    private let baseURL = "http://10.36.143.251:8000/api"
    #endif
    
    /// Parse backend JSON error body: { "error": "message" }
    private static func parseServerError(from data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let message = json["error"] as? String, !message.isEmpty else { return nil }
        return message
    }

    // fetch risk predictions for region (pass weather for real-time risk adjustment)
    func fetchRiskPredictions(for region: MKCoordinateRegion, weather: WeatherData? = nil) async throws -> [RoadSegment] {
        await MainActor.run {
            self.isLoading = true
        }
        
        do {
            let urlString = "\(baseURL)/risk-predictions"
            guard let url = URL(string: urlString) else {
                await MainActor.run {
                    self.isLoading = false
                }
                throw APIError.invalidURL
            }
            
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 30.0  // 30 second timeout
            
            var requestBody: [String: Any] = [
                "north": region.center.latitude + region.span.latitudeDelta / 2,
                "south": region.center.latitude - region.span.latitudeDelta / 2,
                "east": region.center.longitude + region.span.longitudeDelta / 2,
                "west": region.center.longitude - region.span.longitudeDelta / 2
            ]
            if let weather = weather {
                var weatherDict: [String: Any] = ["condition": weather.condition.rawValue]
                weatherDict["temperature"] = weather.temperature
                if let v = weather.visibility { weatherDict["visibility"] = v }
                if let p = weather.precipitation { weatherDict["precipitation"] = p }
                requestBody["weather"] = weatherDict
            }
            let cal = Calendar.current
            let now = Date()
            requestBody["time_of_day"] = [
                "hour": cal.component(.hour, from: now),
                "is_weekend": [1, 7].contains(cal.component(.weekday, from: now))
            ]
            
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                await MainActor.run {
                    self.errorMessage = "Invalid response"
                    self.isLoading = false
                }
                throw APIError.serverError("Invalid response")
            }
            
            guard httpResponse.statusCode == 200 else {
                let serverMessage = Self.parseServerError(from: data) ?? "Status code: \(httpResponse.statusCode)"
                await MainActor.run {
                    self.errorMessage = serverMessage
                    self.isLoading = false
                }
                throw APIError.serverError(serverMessage)
            }
            
            let segments = try JSONDecoder().decode([RoadSegment].self, from: data)
            
            await MainActor.run {
                self.roadSegments = segments
                self.errorMessage = nil
                self.isLoading = false
            }
            
            return segments
        } catch let error as DecodingError {
            await MainActor.run {
                self.errorMessage = "Failed to decode response: \(error.localizedDescription)"
                self.isLoading = false
            }
            throw APIError.decodingError
        } catch let urlError as URLError {
            let hint: String
            switch urlError.code {
            case .cannotConnectToHost, .cannotFindHost:
                #if targetEnvironment(simulator)
                hint = "Backend not running. Start it: cd backend-api && python app.py"
                #else
                hint = "Cannot reach backend. Check: (1) Backend running on Mac? (2) iPhone and Mac on same Wi‑Fi? (3) Mac IP in RiskService.swift still correct? (Run: ipconfig getifaddr en0)"
                #endif
            case .notConnectedToInternet, .networkConnectionLost:
                hint = "No network. Use Wi‑Fi (same network as your Mac) for device testing."
            case .timedOut:
                hint = "Request timed out. Is the backend running and reachable at the configured IP?"
            default:
                hint = urlError.localizedDescription
            }
            await MainActor.run {
                self.errorMessage = hint
                self.isLoading = false
            }
            throw APIError.networkError(urlError)
        } catch {
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
            }
            throw error
        }
    }
    
    // MARK: - get risk prediction for location
    func getRiskPrediction(for location: CLLocationCoordinate2D) async throws -> RiskPredictionResponse {
        let urlString = "\(baseURL)/risk-prediction"
        guard let url = URL(string: urlString) else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "latitude": location.latitude,
            "longitude": location.longitude
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.serverError("Invalid response")
        }
        
        return try JSONDecoder().decode(RiskPredictionResponse.self, from: data)
    }
    
    // MARK: - get high risk roads
    func getHighRiskRoads() -> [RoadSegment] {
        return roadSegments.filter { $0.riskLevel == .high }
            .sorted { $0.numTotalCrashes > $1.numTotalCrashes }
    }

    /// Fetch risk definition (percentile thresholds) from backend
    func fetchRiskDefinition() async throws -> RiskDefinitionResponse {
        let urlString = "\(baseURL)/risk-definition"
        guard let url = URL(string: urlString) else { throw APIError.invalidURL }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 15.0

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw APIError.serverError("Failed to load risk definition")
        }

        let def = try JSONDecoder().decode(RiskDefinitionResponse.self, from: data)
        await MainActor.run {
            self.featureImportance = def.featureImportance
        }
        return def
    }

    /// Fetch safety-aware routes from backend (fastest vs safer - genuinely different routes from graph-based routing).
    /// - Parameter beta: Balance between speed and safety. Higher = more weight on avoiding risk (0.25 speed, 1 balanced, 3 safety).
    func fetchSafetyAwareRoutes(
        origin: CLLocationCoordinate2D,
        destination: CLLocationCoordinate2D,
        weather: WeatherData? = nil,
        beta: Double = 1.0
    ) async throws -> SafetyAwareResponse {
        let urlString = "\(baseURL)/routes/safety-aware"
        guard let url = URL(string: urlString) else { throw APIError.invalidURL }

        var requestBody: [String: Any] = [
            "origin": ["latitude": origin.latitude, "longitude": origin.longitude],
            "destination": ["latitude": destination.latitude, "longitude": destination.longitude],
            "beta": beta
        ]
        if let weather = weather {
            var weatherDict: [String: Any] = ["condition": weather.condition.rawValue]
            weatherDict["temperature"] = weather.temperature
            if let v = weather.visibility { weatherDict["visibility"] = v }
            if let p = weather.precipitation { weatherDict["precipitation"] = p }
            requestBody["weather"] = weatherDict
        }
        let cal = Calendar.current
        let now = Date()
        requestBody["time_of_day"] = [
            "hour": cal.component(.hour, from: now),
            "is_weekend": [1, 7].contains(cal.component(.weekday, from: now))
        ]

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 25.0
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.serverError("Invalid response")
        }

        guard httpResponse.statusCode == 200 else {
            let serverMessage = Self.parseServerError(from: data) ?? "Status code: \(httpResponse.statusCode)"
            throw APIError.serverError(serverMessage)
        }

        return try JSONDecoder().decode(SafetyAwareResponse.self, from: data)
    }

    func fetchGoogleMapsETAs(namedURLs: [String: String]) async throws -> GoogleMapsETAResponse {
        let urlString = "\(baseURL)/debug/google-maps-eta"
        guard let url = URL(string: urlString) else { throw APIError.invalidURL }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 20.0
        request.httpBody = try JSONSerialization.data(withJSONObject: ["urls": namedURLs])

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.serverError("Invalid response")
        }

        guard httpResponse.statusCode == 200 else {
            let serverMessage = Self.parseServerError(from: data) ?? "Status code: \(httpResponse.statusCode)"
            throw APIError.serverError(serverMessage)
        }

        let decoded = try JSONDecoder().decode(GoogleMapsETAResponse.self, from: data)
        print("[google-eta] backend response etasSeconds=\(decoded.etasSeconds) failures=\(decoded.failures) sources=\(decoded.sources ?? [:])")
        return decoded
    }
}

// mapkit import
import MapKit

extension MKCoordinateRegion {
    // helper for creating region from bounds
    static func toronto() -> MKCoordinateRegion {
        MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
            span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
        )
    }
}

