//
//  RiskModels.swift
//  RiskMapApp
//
//  data models for road segments and risk predictions
//

import Foundation
import CoreLocation
import MapKit

// risk level enum
enum RiskLevel: String, Codable, CaseIterable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    
    var displayName: String {
        switch self {
        case .low: return "Low Risk"
        case .medium: return "Medium Risk"
        case .high: return "High Risk"
        }
    }
    
    var color: String {
        switch self {
        case .low: return "#2E8B57"      // green
        case .medium: return "#FFA500"   // orange
        case .high: return "#DC143C"     // crimson red
        }
    }
    
    var systemImage: String {
        switch self {
        case .low: return "checkmark.circle.fill"
        case .medium: return "exclamationmark.circle.fill"
        case .high: return "xmark.circle.fill"
        }
    }
}

// MARK: - Road Segment Model
struct RoadSegment: Identifiable, Codable {
    let id: String
    let linearName: String
    let roadClass: String
    let segmentLength: Double
    let riskLevel: RiskLevel
    let riskScore: Int?
    let riskExplanation: String?
    let confidence: Double
    let numTotalCrashes: Int
    let numKSICrashes: Int
    let fatalityCount: Int
    let coordinates: [Coordinate]
    let probabilities: RiskProbabilities?
    let weatherMult: Double?
    let timeMult: Double?

    struct Coordinate: Codable {
        let latitude: Double
        let longitude: Double
    }

    struct RiskProbabilities: Codable {
        let low: Double
        let medium: Double
        let high: Double
    }

    var displayRiskScore: Int {
        riskScore ?? Int(confidence * 100)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case linearName = "LINEAR_NAME"
        case roadClass = "ROAD_CLASS"
        case segmentLength = "segment_length"
        case riskLevel = "risk_label"
        case riskScore = "risk_score"
        case riskExplanation = "risk_explanation"
        case confidence
        case numTotalCrashes = "num_total_crashes"
        case numKSICrashes = "num_ksi_crashes"
        case fatalityCount = "fatality_count"
        case coordinates
        case probabilities
        case weatherMult = "weather_mult"
        case timeMult = "time_mult"
    }
}

// risk prediction response
struct RiskPredictionResponse: Codable {
    let riskLevel: RiskLevel
    let riskScore: Int?
    let riskExplanation: String?
    let confidence: Double
    let probabilities: RiskProbabilities
    let segmentInfo: RoadSegment?
    
    struct RiskProbabilities: Codable {
        let low: Double
        let medium: Double
        let high: Double
    }
}

// MARK: - Weather Data (for real-time risk adjustment)
struct WeatherData: Codable {
    let condition: WeatherCondition
    let temperature: Double
    let visibility: Double? // in km
    let windSpeed: Double? // in km/h
    let precipitation: Double? // in mm

    enum WeatherCondition: String, Codable {
        case clear = "clear"
        case cloudy = "cloudy"
        case rain = "rain"
        case heavyRain = "heavy_rain"
        case snow = "snow"
        case heavySnow = "heavy_snow"
        case fog = "fog"
        case mist = "mist"
        case thunderstorm = "thunderstorm"
        case sleet = "sleet"

        var displayName: String {
            switch self {
            case .clear: return "Clear"
            case .cloudy: return "Cloudy"
            case .mist: return "Mist"
            case .rain: return "Rain"
            case .heavyRain: return "Heavy Rain"
            case .snow: return "Snow"
            case .heavySnow: return "Heavy Snow"
            case .fog: return "Fog"
            case .thunderstorm: return "Thunderstorm"
            case .sleet: return "Sleet"
            }
        }
    }
}

// MARK: - Route Model (for RouteService)
struct Route: Identifiable {
    let id = UUID()
    let route: MKRoute
    let riskScore: Double
    let highRiskSegments: Int
    let mediumRiskSegments: Int
    let lowRiskSegments: Int
    let routeType: RouteType

    enum RouteType {
        case safer
        case optimal
    }

    var estimatedTime: TimeInterval {
        route.expectedTravelTime
    }

    var distance: CLLocationDistance {
        route.distance
    }

    var polyline: MKPolyline {
        route.polyline
    }

    var steps: [MKRoute.Step] {
        route.steps
    }

    var detailedCoordinates: [CLLocationCoordinate2D] {
        route.detailedCoordinates
    }

    func safetyExplanation(comparedTo optimalRoute: Route?) -> String {
        var reasons: [String] = []
        if let optimal = optimalRoute {
            let highRiskDiff = optimal.highRiskSegments - self.highRiskSegments
            let riskScoreDiff = optimal.riskScore - self.riskScore
            if highRiskDiff > 0 {
                reasons.append("Avoids \(highRiskDiff) additional high-risk segment\(highRiskDiff > 1 ? "s" : "")")
            }
            if self.highRiskSegments == 0 {
                reasons.append("Completely avoids high-risk roads")
            } else if self.highRiskSegments < optimal.highRiskSegments {
                reasons.append("Reduces high-risk exposure by \(highRiskDiff) segment\(highRiskDiff > 1 ? "s" : "")")
            }
            if riskScoreDiff > 0.3 {
                reasons.append("Lower overall risk score (\(String(format: "%.1f", self.riskScore)) vs \(String(format: "%.1f", optimal.riskScore)))")
            }
            if self.lowRiskSegments > optimal.lowRiskSegments {
                let diff = self.lowRiskSegments - optimal.lowRiskSegments
                reasons.append("Uses \(diff) more low-risk segment\(diff > 1 ? "s" : "")")
            }
        }
        return reasons.isEmpty ? "Optimized for safety" : reasons.joined(separator: ", ")
    }
}

struct RouteComparison {
    let saferRoute: Route
    let optimalRoute: Route

    var timeDifference: TimeInterval {
        abs(saferRoute.estimatedTime - optimalRoute.estimatedTime)
    }

    var saferRouteSlower: Bool {
        saferRoute.estimatedTime > optimalRoute.estimatedTime
    }

    var safetyImprovement: String {
        let highRiskDiff = optimalRoute.highRiskSegments - saferRoute.highRiskSegments
        let riskScoreDiff = optimalRoute.riskScore - saferRoute.riskScore
        var improvements: [String] = []
        if highRiskDiff > 0 {
            improvements.append("\(highRiskDiff) fewer high-risk segment\(highRiskDiff > 1 ? "s" : "")")
        }
        if riskScoreDiff > 0.3 {
            improvements.append("\(String(format: "%.1f", riskScoreDiff)) points lower risk score")
        }
        if saferRoute.lowRiskSegments > optimalRoute.lowRiskSegments {
            let diff = saferRoute.lowRiskSegments - optimalRoute.lowRiskSegments
            improvements.append("\(diff) more low-risk segment\(diff > 1 ? "s" : "")")
        }
        return improvements.isEmpty ? "Similar safety profile with optimized route planning" : improvements.joined(separator: ", ")
    }

    var detailedExplanation: String {
        var explanation = "The safer route was selected because it "
        let highRiskDiff = optimalRoute.highRiskSegments - saferRoute.highRiskSegments
        let riskScoreDiff = optimalRoute.riskScore - saferRoute.riskScore
        var reasons: [String] = []
        if highRiskDiff > 0 {
            reasons.append("avoids \(highRiskDiff) high-risk road segment\(highRiskDiff > 1 ? "s" : "") that the fastest route would take")
        }
        if saferRoute.highRiskSegments == 0 && optimalRoute.highRiskSegments > 0 {
            reasons.append("completely eliminates high-risk road exposure")
        }
        if riskScoreDiff > 0.5 {
            reasons.append("has a significantly lower overall risk score (\(String(format: "%.1f", saferRoute.riskScore)) vs \(String(format: "%.1f", optimalRoute.riskScore)))")
        }
        if saferRoute.lowRiskSegments > optimalRoute.lowRiskSegments + 2 {
            let diff = saferRoute.lowRiskSegments - optimalRoute.lowRiskSegments
            reasons.append("uses \(diff) more low-risk road segments")
        }
        if reasons.isEmpty {
            return "The safer route offers similar safety with optimized planning."
        }
        explanation += reasons.joined(separator: " and ")
        return explanation + "."
    }
}

// MARK: - MKRoute detailedCoordinates (used by Route)
extension MKRoute {
    var detailedCoordinates: [CLLocationCoordinate2D] {
        var allCoords: [CLLocationCoordinate2D] = []
        let polylineCoords = polyline.coordinates
        if !steps.isEmpty {
            for step in steps {
                for coord in step.polyline.coordinates {
                    if allCoords.isEmpty || !allCoords.contains(where: {
                        abs($0.latitude - coord.latitude) < 0.00001 && abs($0.longitude - coord.longitude) < 0.00001
                    }) {
                        allCoords.append(coord)
                    }
                }
            }
        }
        return allCoords.count >= 10 ? allCoords : polylineCoords
    }
}

// API error
enum APIError: Error, LocalizedError {
    case invalidURL
    case noData
    case decodingError
    case serverError(String)
    case networkError(Error)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .noData:
            return "No data received"
        case .decodingError:
            return "Failed to decode response"
        case .serverError(let message):
            return "Server error: \(message)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}






