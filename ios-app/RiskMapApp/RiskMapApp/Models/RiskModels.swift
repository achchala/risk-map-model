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
struct RoadSegment: Identifiable, Codable, Equatable {
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
    let segmentLocation: String?
    let riskDrivers: [String: Double]?
    let riskExplanation: String?
    
    struct Coordinate: Codable, Equatable {
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
        case segmentLocation = "segment_location"
        case riskDrivers
        case riskExplanation = "risk_explanation"
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

// MARK: - Safety-aware routing API

struct RouteSegment: Codable {
    let segmentId: Int
    let coordinates: [Coordinate]
    let LINEAR_NAME: String
    let ROAD_CLASS: String
    let lambdaPerHour: Double
    let expectedCrashes: Double
    let risk_label: String?

    struct Coordinate: Codable, Equatable {
        let latitude: Double
        let longitude: Double
    }

    var riskLevel: RiskLevel {
        RiskLevel(rawValue: risk_label ?? "low") ?? .low
    }
}

struct RouteSummary: Codable {
    let totalTravelTimeHours: Double
    let expectedCrashes: Double
    let routeProbability: Double
    let highRiskSegments: Int?
    let mediumRiskSegments: Int?
    let lowRiskSegments: Int?
}

struct RouteOption: Codable {
    let nodes: [Int]
    let segmentIds: [Int]
    let segments: [RouteSegment]
    let summary: RouteSummary

    /// Flattened coordinates for the full route (one polyline instead of per-segment).
    var fullRouteCoordinates: [CLLocationCoordinate2D] {
        var result: [CLLocationCoordinate2D] = []
        for seg in segments {
            let coords = seg.coordinates.map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) }
            guard !coords.isEmpty else { continue }
            if result.isEmpty {
                result = coords
            } else {
                let last = result.last!
                let first = coords[0]
                let same = abs(last.latitude - first.latitude) < 1e-9 && abs(last.longitude - first.longitude) < 1e-9
                result.append(contentsOf: same ? Array(coords.dropFirst()) : coords)
            }
        }
        return result.filter { $0.latitude.isFinite && $0.longitude.isFinite }
    }
}

struct AvoidedSegment: Codable, Identifiable, Equatable {
    var id: Int { segmentId }
    let segmentId: Int
    let lambdaPerHour: Double
    let riskDrivers: [String: Double]?
    let riskExplanation: String?
    let segmentLocation: String?
    let coordinates: [RouteSegment.Coordinate]
    let LINEAR_NAME: String
    let ROAD_CLASS: String
    let risk_label: String
    let numTotalCrashes: Int?
    let numKSICrashes: Int?
    let fatalityCount: Int?

    enum CodingKeys: String, CodingKey {
        case segmentId, lambdaPerHour, riskDrivers, coordinates, LINEAR_NAME, ROAD_CLASS, risk_label
        case riskExplanation = "risk_explanation"
        case segmentLocation = "segment_location"
        case numTotalCrashes = "num_total_crashes"
        case numKSICrashes = "num_ksi_crashes"
        case fatalityCount = "fatality_count"
    }
    
    var riskLevel: RiskLevel {
        RiskLevel(rawValue: risk_label) ?? .medium
    }
}

struct SafetyAwareResponse: Codable {
    let fastest: RouteOption
    let safer: RouteOption
    let avoidedSegments: [AvoidedSegment]
    let betaHoursPerExpectedCrash: Double
}

struct GoogleMapsETAResponse: Codable {
    let etasSeconds: [String: Double]
    let failures: [String: String]
}

struct RiskDefinitionResponse: Codable {
    let p70: Double
    let p90: Double
    let description: String
    let low: String
    let medium: String
    let high: String
    /// Feature importance from model (for ranking risk drivers). Optional; backend sends when available.
    let featureImportance: [String: Double]?
}

// MARK: - Route (MapKit or backend-based, for safer vs optimal comparison)
struct Route: Identifiable {
    let id = UUID()
    let polyline: MKPolyline
    var estimatedTime: TimeInterval?
    let distance: CLLocationDistance
    let riskScore: Double
    let highRiskSegments: Int
    let mediumRiskSegments: Int
    let lowRiskSegments: Int
    let routeType: RouteType
    private let _steps: [MKRoute.Step]?

    enum RouteType {
        case safer
        case optimal
    }

    /// Create Route from MapKit MKRoute (after risk analysis)
    init(mkRoute: MKRoute, riskScore: Double, highRiskSegments: Int, mediumRiskSegments: Int, lowRiskSegments: Int, routeType: RouteType) {
        self.polyline = mkRoute.polyline
        self.estimatedTime = mkRoute.expectedTravelTime
        self.distance = mkRoute.distance
        self.riskScore = riskScore
        self.highRiskSegments = highRiskSegments
        self.mediumRiskSegments = mediumRiskSegments
        self.lowRiskSegments = lowRiskSegments
        self.routeType = routeType
        self._steps = mkRoute.steps
    }

    /// Create Route from backend safety-aware API response (produces genuinely different fastest vs safer).
    /// - Parameter estimatedTimeOverride: When provided (e.g. from MapKit's effective speed), overrides backend's optimistic free-flow estimate.
    init(routeOption: RouteOption, routeType: RouteType, estimatedTimeOverride: TimeInterval? = nil) {
        let coords = routeOption.fullRouteCoordinates
        self.polyline = MKPolyline(coordinates: coords, count: coords.count)
        self.distance = Self.computeDistance(coords: coords)
        self.estimatedTime = estimatedTimeOverride
        let high = routeOption.summary.highRiskSegments ?? 0
        let med = routeOption.summary.mediumRiskSegments ?? 0
        let low = routeOption.summary.lowRiskSegments ?? 0
        let total = high + med + low
        self.riskScore = total > 0 ? (Double(high) * 3.0 + Double(med) * 2.0 + Double(low) * 1.0) / Double(total) : 0.0
        self.highRiskSegments = high
        self.mediumRiskSegments = med
        self.lowRiskSegments = low
        self.routeType = routeType
        self._steps = nil
    }

    private static func computeDistance(coords: [CLLocationCoordinate2D]) -> CLLocationDistance {
        computeDistanceStatic(coords: coords)
    }

    static func computeDistanceStatic(coords: [CLLocationCoordinate2D]) -> CLLocationDistance {
        guard coords.count >= 2 else { return 0 }
        var total: CLLocationDistance = 0
        for i in 0..<coords.count - 1 {
            total += CLLocation(latitude: coords[i].latitude, longitude: coords[i].longitude)
                .distance(from: CLLocation(latitude: coords[i + 1].latitude, longitude: coords[i + 1].longitude))
        }
        return total
    }

    var steps: [MKRoute.Step] {
        _steps ?? []
    }

    var detailedCoordinates: [CLLocationCoordinate2D] {
        polyline.coordinates
    }

    func shapingWaypoints(maxWaypoints: Int) -> [CLLocationCoordinate2D] {
        Self.shapingWaypointsForCoordinates(detailedCoordinates, maxWaypoints: maxWaypoints)
    }

    func googleMapsURL(
        origin: CLLocationCoordinate2D,
        destination: CLLocationCoordinate2D,
        maxWaypoints: Int = 8
    ) -> String? {
        let waypoints = shapingWaypoints(maxWaypoints: maxWaypoints)
            .map { "\($0.latitude),\($0.longitude)" }
        var url = "https://www.google.com/maps/dir/?api=1&origin=\(origin.latitude),\(origin.longitude)&destination=\(destination.latitude),\(destination.longitude)&travelmode=driving"
        if !waypoints.isEmpty {
            url += "&waypoints=\(waypoints.joined(separator: "|"))"
        }
        return url.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)
    }

    static func shapingWaypointsForCoordinates(_ coordinates: [CLLocationCoordinate2D], maxWaypoints: Int) -> [CLLocationCoordinate2D] {
        let coords = coordinates.filter { $0.latitude.isFinite && $0.longitude.isFinite }
        guard coords.count >= 3, maxWaypoints > 0 else { return [] }

        let minSpacingMeters: CLLocationDistance = 350
        var selectedIndices: [Int] = []
        var lastChosen = CLLocation(latitude: coords[0].latitude, longitude: coords[0].longitude)

        // Prefer meaningful bends/turns first so external navigation apps are nudged
        // toward the selected route shape rather than arbitrary evenly spaced points.
        for index in 1..<(coords.count - 1) {
            let prev = coords[index - 1]
            let curr = coords[index]
            let next = coords[index + 1]

            let turnAngle = abs(routeHeadingDelta(
                routeHeading(from: prev, to: curr),
                routeHeading(from: curr, to: next)
            ))
            let currLocation = CLLocation(latitude: curr.latitude, longitude: curr.longitude)
            let spacing = currLocation.distance(from: lastChosen)

            if turnAngle >= 25, spacing >= minSpacingMeters {
                selectedIndices.append(index)
                lastChosen = currLocation
                if selectedIndices.count >= maxWaypoints { break }
            }
        }

        if selectedIndices.count < maxWaypoints {
            let evenlySpacedIndices = evenlySpacedWaypointIndices(
                coordinates: coords,
                count: maxWaypoints - selectedIndices.count
            )
            for index in evenlySpacedIndices {
                if selectedIndices.count >= maxWaypoints { break }
                let coord = coords[index]
                let existingCoords = selectedIndices.map { coords[$0] }
                if !containsNearbyCoordinate(existingCoords, coord: coord, thresholdMeters: 120) {
                    selectedIndices.append(index)
                }
            }
        }

        return selectedIndices
            .prefix(maxWaypoints)
            .sorted()
            .map { coords[$0] }
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

private func evenlySpacedWaypointIndices(coordinates: [CLLocationCoordinate2D], count: Int) -> [Int] {
    guard coordinates.count >= 3, count > 0 else { return [] }
    let step = Double(coordinates.count - 1) / Double(count + 1)
    return (1...count).compactMap { i in
        let idx = Int(round(step * Double(i)))
        guard idx > 0 && idx < coordinates.count - 1 else { return nil }
        return idx
    }
}

private func containsNearbyCoordinate(_ coordinates: [CLLocationCoordinate2D], coord: CLLocationCoordinate2D, thresholdMeters: CLLocationDistance) -> Bool {
    let location = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
    return coordinates.contains {
        CLLocation(latitude: $0.latitude, longitude: $0.longitude).distance(from: location) < thresholdMeters
    }
}

private func routeHeading(from start: CLLocationCoordinate2D, to end: CLLocationCoordinate2D) -> Double {
    let dy = end.latitude - start.latitude
    let dx = end.longitude - start.longitude
    return atan2(dy, dx) * 180 / .pi
}

private func routeHeadingDelta(_ a: Double, _ b: Double) -> Double {
    var delta = b - a
    while delta > 180 { delta -= 360 }
    while delta < -180 { delta += 360 }
    return delta
}

struct RouteComparison {
    let saferRoute: Route
    let optimalRoute: Route

    var timeDifference: TimeInterval? {
        guard let saferETA = saferRoute.estimatedTime, let fastestETA = optimalRoute.estimatedTime else {
            return nil
        }
        return abs(saferETA - fastestETA)
    }

    var saferRouteSlower: Bool? {
        guard let saferETA = saferRoute.estimatedTime, let fastestETA = optimalRoute.estimatedTime else {
            return nil
        }
        return saferETA > fastestETA
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
        return "The safer route was selected because it " + reasons.joined(separator: " and ") + "."
    }
}

// MARK: - MKRoute detailedCoordinates
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






