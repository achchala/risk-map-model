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
    
    // backend API URL - update this to your Mac's IP address when testing on device
    // for iOS Simulator, use localhost. For physical device, use your Mac's IP
    private let baseURL = "http://localhost:8000/api"
    // For Simulator: "http://localhost:8000/api"
    // For Device: "http://10.10.11.47:8000/api" (update IP if needed)
    
    // fetch risk predictions for region (pass weather for real-time risk adjustment)
    func fetchRiskPredictions(for region: MKCoordinateRegion, weather: WeatherData? = nil) async throws -> [RoadSegment] {
        await MainActor.run {
            self.isLoading = true
        }

        do {
            let urlString = "\(baseURL)/risk-predictions"
            guard let url = URL(string: urlString) else {
                await MainActor.run { self.isLoading = false }
                throw APIError.invalidURL
            }

            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.timeoutInterval = 30.0

            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            formatter.timeZone = TimeZone.current
            let asOf = formatter.string(from: Date())

            var requestBody: [String: Any] = [
                "north": region.center.latitude + region.span.latitudeDelta / 2,
                "south": region.center.latitude - region.span.latitudeDelta / 2,
                "east": region.center.longitude + region.span.longitudeDelta / 2,
                "west": region.center.longitude - region.span.longitudeDelta / 2,
                "as_of": asOf
            ]

            if let weather = weather {
                var weatherDict: [String: Any] = ["condition": weather.condition.rawValue]
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
                await MainActor.run {
                    self.errorMessage = "Status code: \(httpResponse.statusCode)"
                    self.isLoading = false
                }
                throw APIError.serverError("Status code: \(httpResponse.statusCode)")
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
        } catch {
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
            }
            throw error
        }
    }
    
    // MARK: - get risk prediction for location (pass weather for real-time adjustment)
    func getRiskPrediction(for location: CLLocationCoordinate2D, weather: WeatherData? = nil) async throws -> RiskPredictionResponse {
        let urlString = "\(baseURL)/risk-prediction"
        guard let url = URL(string: urlString) else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone.current

        var requestBody: [String: Any] = [
            "latitude": location.latitude,
            "longitude": location.longitude,
            "as_of": formatter.string(from: Date())
        ]
        if let weather = weather {
            var w: [String: Any] = ["condition": weather.condition.rawValue]
            if let v = weather.visibility { w["visibility"] = v }
            if let p = weather.precipitation { w["precipitation"] = p }
            requestBody["weather"] = w
        }
        
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

