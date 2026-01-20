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

// MARK: - Route Models
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
    
    /// Get detailed coordinates following the actual road path
    var detailedCoordinates: [CLLocationCoordinate2D] {
        route.detailedCoordinates
    }
    
    // Generate explanation for why this route is safer
    func safetyExplanation(comparedTo optimalRoute: Route?) -> String {
        var reasons: [String] = []
        
        if let optimal = optimalRoute {
            // Compare to optimal route
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
        } else {
            // Standalone explanation
            if highRiskSegments == 0 {
                reasons.append("Avoids all high-risk road segments")
            } else if highRiskSegments < 3 {
                reasons.append("Minimizes high-risk segments (\(highRiskSegments))")
            }
            
            if lowRiskSegments > mediumRiskSegments + highRiskSegments {
                reasons.append("Primarily uses low-risk roads (\(lowRiskSegments) segments)")
            }
            
            if riskScore < 1.5 {
                reasons.append("Overall low risk score (\(String(format: "%.1f", riskScore)))")
            }
        }
        
        if reasons.isEmpty {
            return "This route balances safety with travel time."
        }
        
        return reasons.joined(separator: ". ") + "."
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
        
        if improvements.isEmpty {
            return "Similar safety profile with optimized route planning"
        }
        
        return improvements.joined(separator: ", ")
    }
    
    // Detailed explanation of why safer route was chosen
    var detailedExplanation: String {
        var explanation = "The safer route was selected because it "
        
        let highRiskDiff = optimalRoute.highRiskSegments - saferRoute.highRiskSegments
        let riskScoreDiff = optimalRoute.riskScore - saferRoute.riskScore
        let timeDiff = saferRoute.estimatedTime - optimalRoute.estimatedTime
        
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
            reasons.append("prioritizes low-risk roads (\(saferRoute.lowRiskSegments) vs \(optimalRoute.lowRiskSegments) segments)")
        }
        
        if reasons.isEmpty {
            return "The safer route provides better safety planning while maintaining reasonable travel time."
        }
        
        explanation += reasons.joined(separator: ", ")
        
        if timeDiff > 0 {
            let minutes = Int(timeDiff / 60)
            explanation += ". This adds approximately \(minutes) minute\(minutes > 1 ? "s" : "") to your journey but significantly improves safety."
        } else {
            explanation += " while maintaining similar travel time."
        }
        
        return explanation
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






