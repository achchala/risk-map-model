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
    /// Total segments in last map fetch bbox (before cap). Nil when unknown. If > 200, map should show "zoom in" overlay.
    @Published var totalSegmentsInView: Int?
    /// All segments for Road Details list (full Toronto, higher limit). Fetched separately from map.
    @Published var allSegmentsForList: [RoadSegment] = []
    @Published var isLoading = false
    @Published var isLoadingList = false
    @Published var errorMessage: String?
    /// Feature importance from model (for ranking risk drivers). Fetched with risk-definition.
    @Published var featureImportance: [String: Double]? = nil
    
    // Backend API URL. Use Settings to set your Mac's IP when testing on a physical device (same Wi‑Fi).
    private var baseURL: String {
        if let custom = UserDefaults.standard.string(forKey: "backendAPIURL"), !custom.trimmingCharacters(in: .whitespaces).isEmpty {
            var trimmed = custom.trimmingCharacters(in: .whitespaces)
            if trimmed.hasSuffix("/") { trimmed = String(trimmed.dropLast()) }
            return trimmed.hasSuffix("/api") ? trimmed : "\(trimmed)/api"
        }
        #if targetEnvironment(simulator)
        return "http://localhost:8000/api"
        #else
        return "http://10.36.169.232:8000/api"  // fallback; set in Settings for your Mac's IP
        #endif
    }
    
    /// Add ngrok bypass header when using ngrok tunnel (skips browser interstitial)
    private func addNgrokBypass(to request: inout URLRequest) {
        if baseURL.contains("ngrok") {
            request.setValue("true", forHTTPHeaderField: "ngrok-skip-browser-warning")
        }
    }

    /// Parse backend JSON error body: { "error": "message", "code": "OUT_OF_BOUNDS" }
    private static func parseServerError(from data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let message = json["error"] as? String, !message.isEmpty else { return nil }
        return message
    }

    /// Parse structured error: returns (message, code) for 400 responses.
    private static func parseStructuredError(from data: Data) -> (message: String, code: String?)? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let message = json["error"] as? String else { return nil }
        let code = json["code"] as? String
        return (message, code)
    }

    // fetch risk predictions for region (pass weather for real-time risk adjustment)
    /// - Parameter limit: Max segments to return. Default 200 for map.
    /// - Parameter includeTotal: When true, backend returns total_in_bbox; we set totalSegmentsInView. Use for map zoom logic.
    func fetchRiskPredictions(for region: MKCoordinateRegion, weather: WeatherData? = nil, limit: Int? = nil, includeTotal: Bool = false) async throws -> [RoadSegment] {
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
            addNgrokBypass(to: &request)
            
            var requestBody: [String: Any] = [
                "north": region.center.latitude + region.span.latitudeDelta / 2,
                "south": region.center.latitude - region.span.latitudeDelta / 2,
                "east": region.center.longitude + region.span.longitudeDelta / 2,
                "west": region.center.longitude - region.span.longitudeDelta / 2
            ]
            if let limit = limit { requestBody["limit"] = limit }
            if includeTotal { requestBody["include_total"] = true }
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
            
            let segments: [RoadSegment]
            if includeTotal {
                struct Wrapper: Decodable {
                    let segments: [RoadSegment]
                    let total_in_bbox: Int
                }
                let wrapper = try JSONDecoder().decode(Wrapper.self, from: data)
                segments = wrapper.segments
                await MainActor.run {
                    self.roadSegments = segments
                    self.totalSegmentsInView = wrapper.total_in_bbox
                    self.errorMessage = nil
                    self.isLoading = false
                }
            } else {
                segments = try JSONDecoder().decode([RoadSegment].self, from: data)
                await MainActor.run {
                    self.roadSegments = segments
                    self.totalSegmentsInView = nil
                    self.errorMessage = nil
                    self.isLoading = false
                }
            }
            
            return segments
        } catch let error as DecodingError {
            await MainActor.run {
                self.errorMessage = "Failed to decode response: \(error.localizedDescription)"
                self.isLoading = false
            }
            throw APIError.decodingError
        } catch let urlError as URLError {
            await MainActor.run {
                self.isLoading = false
            }
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

    /// Fetch all segments for Road Details list (full Toronto bbox, higher limit, lite payload for speed).
    func fetchAllRiskPredictionsForList() async throws {
        await MainActor.run {
            self.isLoadingList = true
        }
        defer {
            Task { @MainActor in self.isLoadingList = false }
        }
        let urlString = "\(baseURL)/risk-predictions"
        guard let url = URL(string: urlString) else { throw APIError.invalidURL }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 90.0  // Longer timeout for full load
        addNgrokBypass(to: &request)
        // Full Toronto bbox; limit 5000; lite=true; prioritize_risk=false for geographic mix
        let requestBody: [String: Any] = [
            "north": 43.85,
            "south": 43.58,
            "east": -79.11,
            "west": -79.64,
            "limit": 5000,
            "lite": true,
            "prioritize_risk": false,
        ]
        let cal = Calendar.current
        let now = Date()
        var body = requestBody
        body["time_of_day"] = [
            "hour": cal.component(.hour, from: now),
            "is_weekend": [1, 7].contains(cal.component(.weekday, from: now)),
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let msg = Self.parseServerError(from: data) ?? "Failed to load road segments"
            await MainActor.run { self.errorMessage = msg }
            throw APIError.serverError(msg)
        }
        let segments = try JSONDecoder().decode([RoadSegment].self, from: data)
        await MainActor.run {
            self.allSegmentsForList = segments
            self.errorMessage = nil
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
        addNgrokBypass(to: &request)
        
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

    /// Test backend connectivity (for Settings). Returns (success, message).
    func testConnection() async -> (Bool, String) {
        let urlString = "\(baseURL)/health"
        guard let url = URL(string: urlString) else {
            return (false, "Invalid URL: \(urlString)")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10.0
        addNgrokBypass(to: &request)
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                return (false, "Invalid response")
            }
            if http.statusCode == 200 {
                return (true, "Connected successfully")
            }
            return (false, "HTTP \(http.statusCode) at \(urlString)")
        } catch {
            return (false, "\(error.localizedDescription)\nURL: \(urlString)")
        }
    }

    /// Fetch risk definition (percentile thresholds) from backend
    func fetchRiskDefinition() async throws -> RiskDefinitionResponse {
        let urlString = "\(baseURL)/risk-definition"
        guard let url = URL(string: urlString) else { throw APIError.invalidURL }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 15.0
        addNgrokBypass(to: &request)

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
    /// - Parameter weather: Optional weather for risk adjustment. Pass nil to use default.
    /// - Parameter timeOfDay: Optional (hour, isWeekend) override. Pass nil to use current time.
    func fetchSafetyAwareRoutes(
        origin: CLLocationCoordinate2D,
        destination: CLLocationCoordinate2D,
        weather: WeatherData? = nil,
        timeOfDay: (hour: Int, isWeekend: Bool)? = nil,
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
        if let tod = timeOfDay {
            requestBody["time_of_day"] = ["hour": tod.hour, "is_weekend": tod.isWeekend]
        } else {
            let cal = Calendar.current
            let now = Date()
            requestBody["time_of_day"] = [
                "hour": cal.component(.hour, from: now),
                "is_weekend": [1, 7].contains(cal.component(.weekday, from: now))
            ]
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 25.0
        addNgrokBypass(to: &request)
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.serverError("Invalid response")
        }

        guard httpResponse.statusCode == 200 else {
            if httpResponse.statusCode == 400,
               let parsed = Self.parseStructuredError(from: data),
               parsed.code == "OUT_OF_BOUNDS" {
                throw APIError.outOfBounds(parsed.message)
            }
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
        request.timeoutInterval = 35.0
        addNgrokBypass(to: &request)
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

