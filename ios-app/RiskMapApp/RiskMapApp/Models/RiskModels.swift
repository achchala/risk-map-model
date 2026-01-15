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
    let confidence: Double
    let numTotalCrashes: Int
    let numKSICrashes: Int
    let fatalityCount: Int
    let coordinates: [Coordinate]
    
    struct Coordinate: Codable {
        let latitude: Double
        let longitude: Double
    }
    
    enum CodingKeys: String, CodingKey {
        case id
        case linearName = "LINEAR_NAME"
        case roadClass = "ROAD_CLASS"
        case segmentLength = "segment_length"
        case riskLevel = "risk_label"
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
    let confidence: Double
    let probabilities: RiskProbabilities
    let segmentInfo: RoadSegment?
    
    struct RiskProbabilities: Codable {
        let low: Double
        let medium: Double
        let high: Double
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





