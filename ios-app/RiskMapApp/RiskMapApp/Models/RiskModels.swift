//
//  RiskModels.swift
//  RiskMapApp
//
//  data models for road segments and risk predictions
//

import Foundation
import CoreLocation

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
    
    struct Coordinate: Codable {
        let latitude: Double
        let longitude: Double
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






