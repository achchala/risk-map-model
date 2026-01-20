//
//  RouteService.swift
//  RiskMapApp
//
//  Service for calculating safer and optimal routes using MapKit
//

import Foundation
import MapKit
import CoreLocation
import Combine

class RouteService: ObservableObject {
    @Published var saferRoute: Route?
    @Published var optimalRoute: Route?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let riskService: RiskService
    
    init(riskService: RiskService) {
        self.riskService = riskService
    }
    
    // Calculate both safer and optimal routes
    func calculateRoutes(from start: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D) async {
        await MainActor.run {
            isLoading = true
            errorMessage = nil
        }
        
        do {
            // Calculate optimal route (fastest) - with timeout
            let optimalMKRoute = try await withTimeout(seconds: 30) {
                try await self.calculateOptimalRoute(from: start, to: destination)
            }
            
            // Calculate safer route (avoiding high-risk segments) - with timeout
            let saferMKRoute = try await withTimeout(seconds: 30) {
                try await self.calculateSaferRoute(from: start, to: destination)
            }
            
            // Analyze routes for risk - with timeout and error handling
            let optimalRoute = try await withTimeout(seconds: 20) {
                try await self.analyzeRoute(optimalMKRoute, type: .optimal)
            }
            
            let saferRoute = try await withTimeout(seconds: 20) {
                try await self.analyzeRoute(saferMKRoute, type: .safer)
            }
            
            await MainActor.run {
                self.optimalRoute = optimalRoute
                self.saferRoute = saferRoute
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
                print("Route calculation error: \(error.localizedDescription)")
            }
        }
    }
    
    // Helper: Add timeout to async operations
    private func withTimeout<T>(seconds: TimeInterval, operation: @escaping () async throws -> T) async throws -> T {
        return try await withThrowingTaskGroup(of: T.self) { group in
            // Add the actual operation
            group.addTask {
                try await operation()
            }
            
            // Add timeout task
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw RouteError.routeCalculationFailed
            }
            
            // Get the first completed task
            defer { group.cancelAll() }
            return try await group.next()!
        }
    }
    
    // Calculate optimal route (fastest/shortest)
    private func calculateOptimalRoute(from start: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D) async throws -> MKRoute {
        let request = MKDirections.Request()
        request.source = MKMapItem(placemark: MKPlacemark(coordinate: start))
        request.destination = MKMapItem(placemark: MKPlacemark(coordinate: destination))
        request.transportType = .automobile
        request.requestsAlternateRoutes = false
        
        let directions = MKDirections(request: request)
        let response = try await directions.calculate()
        
        guard let route = response.routes.first else {
            throw RouteError.noRouteFound
        }
        
        return route
    }
    
    // Calculate safer route by avoiding high-risk segments
    private func calculateSaferRoute(from start: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D) async throws -> MKRoute {
        // Request multiple alternate routes
        let request = MKDirections.Request()
        request.source = MKMapItem(placemark: MKPlacemark(coordinate: start))
        request.destination = MKMapItem(placemark: MKPlacemark(coordinate: destination))
        request.transportType = .automobile
        request.requestsAlternateRoutes = true
        
        let directions = MKDirections(request: request)
        let response = try await directions.calculate()
        
        // Get risk data for the route area
        let routes = response.routes
        guard !routes.isEmpty else {
            throw RouteError.noRouteFound
        }
        
        // If we only have one route, use it
        if routes.count == 1 {
            return routes[0]
        }
        
        // Get risk segments for all routes (limit to first 3 routes for performance)
        var routeRiskScores: [(route: MKRoute, score: Double)] = []
        
        // Use the first route's region to fetch risk data once (more efficient)
        let firstRouteRegion = createRegionForRoute(routes[0])
        let riskSegments: [RoadSegment]
        
        do {
            riskSegments = try await riskService.fetchRiskPredictions(for: firstRouteRegion)
        } catch {
            // If risk data fetch fails, just return the first route
            print("Warning: Could not fetch risk data, using first route: \(error.localizedDescription)")
            return routes[0]
        }
        
        // Limit risk segments to prevent memory issues
        let maxRiskSegments = 5000
        let limitedRiskSegments = Array(riskSegments.prefix(maxRiskSegments))
        
        // Calculate risk scores for each route using the same risk data
        for route in routes.prefix(3) { // Limit to 3 routes for performance
            let riskScore = calculateRouteRiskScore(route: route, riskSegments: limitedRiskSegments)
            routeRiskScores.append((route: route, score: riskScore))
        }
        
        // Sort by risk score (lower is better) and return the safest route
        routeRiskScores.sort { $0.score < $1.score }
        
        return routeRiskScores.first?.route ?? routes[0]
    }
    
    // Calculate risk score for a route (lower is better)
    private func calculateRouteRiskScore(route: MKRoute, riskSegments: [RoadSegment]) -> Double {
        let routePoints = samplePointsAlongRoute(route)
        guard !routePoints.isEmpty else { return 0.0 }
        
        var totalRiskScore = 0.0
        var pointCount = 0
        
        for point in routePoints {
            if let nearestSegment = findNearestSegment(to: point, in: riskSegments) {
                let riskWeight: Double
                switch nearestSegment.riskLevel {
                case .high:
                    riskWeight = 3.0
                case .medium:
                    riskWeight = 2.0
                case .low:
                    riskWeight = 1.0
                }
                totalRiskScore += riskWeight
                pointCount += 1
            }
        }
        
        return pointCount > 0 ? totalRiskScore / Double(pointCount) : 0.0
    }
    
    // Analyze route for risk levels
    private func analyzeRoute(_ route: MKRoute, type: Route.RouteType) async throws -> Route {
        // Get risk segments for the route area
        let routeRegion = createRegionForRoute(route)
        let riskSegments: [RoadSegment]
        
        do {
            riskSegments = try await riskService.fetchRiskPredictions(for: routeRegion)
        } catch {
            // If risk data fetch fails, return route with default risk values
            print("Warning: Could not fetch risk data for route analysis: \(error.localizedDescription)")
            return Route(
                route: route,
                riskScore: 0.0,
                highRiskSegments: 0,
                mediumRiskSegments: 0,
                lowRiskSegments: 0,
                routeType: type
            )
        }
        
        // Sample points along the route (limit sampling for performance)
        let routePoints = samplePointsAlongRoute(route)
        let sampledPoints = Array(routePoints.prefix(50)) // Limit to 50 points for performance
        
        // Limit risk segments to prevent memory issues
        let maxRiskSegments = 5000
        let limitedRiskSegments = Array(riskSegments.prefix(maxRiskSegments))
        
        // Count risk segments along the route
        var highRiskCount = 0
        var mediumRiskCount = 0
        var lowRiskCount = 0
        var totalRiskScore = 0.0
        var matchedPoints = 0
        
        for point in sampledPoints {
            // Find nearest risk segment
            if let nearestSegment = findNearestSegment(to: point, in: limitedRiskSegments) {
                matchedPoints += 1
                switch nearestSegment.riskLevel {
                case .high:
                    highRiskCount += 1
                    totalRiskScore += 3.0
                case .medium:
                    mediumRiskCount += 1
                    totalRiskScore += 2.0
                case .low:
                    lowRiskCount += 1
                    totalRiskScore += 1.0
                }
            }
        }
        
        let averageRiskScore = matchedPoints > 0 ? totalRiskScore / Double(matchedPoints) : 0.0
        
        return Route(
            route: route,
            riskScore: averageRiskScore,
            highRiskSegments: highRiskCount,
            mediumRiskSegments: mediumRiskCount,
            lowRiskSegments: lowRiskCount,
            routeType: type
        )
    }
    
    // Helper: Create region for route
    private func createRegionForRoute(_ route: MKRoute) -> MKCoordinateRegion {
        let coordinates = route.polyline.coordinates
        guard !coordinates.isEmpty else {
            return MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
                span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
            )
        }
        
        // Single pass to find min/max (more memory efficient)
        var minLat = coordinates[0].latitude
        var maxLat = coordinates[0].latitude
        var minLon = coordinates[0].longitude
        var maxLon = coordinates[0].longitude
        
        for coord in coordinates {
            minLat = min(minLat, coord.latitude)
            maxLat = max(maxLat, coord.latitude)
            minLon = min(minLon, coord.longitude)
            maxLon = max(maxLon, coord.longitude)
        }
        
        let center = CLLocationCoordinate2D(
            latitude: (minLat + maxLat) / 2,
            longitude: (minLon + maxLon) / 2
        )
        
        let span = MKCoordinateSpan(
            latitudeDelta: (maxLat - minLat) * 1.5,
            longitudeDelta: (maxLon - minLon) * 1.5
        )
        
        return MKCoordinateRegion(center: center, span: span)
    }
    
    // Helper: Sample points along route (optimized to prevent memory issues)
    private func samplePointsAlongRoute(_ route: MKRoute) -> [CLLocationCoordinate2D] {
        let coordinates = route.polyline.coordinates
        guard !coordinates.isEmpty else { return [] }
        
        // Limit total coordinates to prevent memory issues
        let maxCoordinates = 500
        if coordinates.count <= maxCoordinates {
            // For short routes, sample every 100 meters
            return samplePointsEvery(coordinates: coordinates, distance: 100, maxPoints: 100)
        } else {
            // For long routes, just take evenly spaced points
            let step = max(1, coordinates.count / 100)
            return stride(from: 0, to: coordinates.count, by: step).map { coordinates[$0] }
        }
    }
    
    // Helper: Sample points at regular intervals
    private func samplePointsEvery(coordinates: [CLLocationCoordinate2D], distance: CLLocationDistance, maxPoints: Int) -> [CLLocationCoordinate2D] {
        var sampledPoints: [CLLocationCoordinate2D] = []
        var accumulatedDistance: CLLocationDistance = 0
        var lastSampledIndex = 0
        
        for i in 0..<coordinates.count - 1 {
            let start = coordinates[i]
            let end = coordinates[i + 1]
            
            let segmentDistance = CLLocation(latitude: start.latitude, longitude: start.longitude)
                .distance(from: CLLocation(latitude: end.latitude, longitude: end.longitude))
            
            // Sample points along this segment
            var segmentAccumulated: CLLocationDistance = 0
            while segmentAccumulated + distance <= segmentDistance && sampledPoints.count < maxPoints {
                let ratio = (segmentAccumulated + distance) / segmentDistance
                let lat = start.latitude + (end.latitude - start.latitude) * ratio
                let lon = start.longitude + (end.longitude - start.longitude) * ratio
                sampledPoints.append(CLLocationCoordinate2D(latitude: lat, longitude: lon))
                segmentAccumulated += distance
                accumulatedDistance += distance
            }
            
            accumulatedDistance += segmentDistance
        }
        
        // Always include the last point
        if !sampledPoints.contains(where: { abs($0.latitude - coordinates.last!.latitude) < 0.0001 && abs($0.longitude - coordinates.last!.longitude) < 0.0001 }) {
            sampledPoints.append(coordinates.last!)
        }
        
        return sampledPoints.isEmpty ? Array(coordinates.prefix(maxPoints)) : sampledPoints
    }
    
    // Helper: Find nearest segment to a point (optimized with early exit)
    private func findNearestSegment(to point: CLLocationCoordinate2D, in segments: [RoadSegment]) -> RoadSegment? {
        let pointLocation = CLLocation(latitude: point.latitude, longitude: point.longitude)
        var nearestSegment: RoadSegment?
        var minDistance: CLLocationDistance = 50.0 // Start with threshold to enable early exit
        let maxDistance: CLLocationDistance = 50.0 // Only consider segments within 50 meters
        
        // Limit search to prevent excessive computation
        let maxSegmentsToCheck = 1000
        let segmentsToCheck = Array(segments.prefix(maxSegmentsToCheck))
        
        for segment in segmentsToCheck {
            // Check first coordinate to quickly filter far segments
            guard let firstCoord = segment.coordinates.first else { continue }
            let firstLocation = CLLocation(latitude: firstCoord.latitude, longitude: firstCoord.longitude)
            let firstDistance = pointLocation.distance(from: firstLocation)
            
            // Skip if first coordinate is already too far
            if firstDistance > maxDistance * 2 {
                continue
            }
            
            // Check all coordinates in segment
            for coord in segment.coordinates {
                let segmentLocation = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
                let distance = pointLocation.distance(from: segmentLocation)
                
                if distance < minDistance {
                    minDistance = distance
                    nearestSegment = segment
                    
                    // Early exit if we find a very close segment
                    if distance < 10 {
                        return nearestSegment
                    }
                }
            }
        }
        
        // Only return if within 50 meters
        return minDistance < maxDistance ? nearestSegment : nil
    }
}

// MARK: - Route Errors
enum RouteError: LocalizedError {
    case noRouteFound
    case locationUnavailable
    case routeCalculationFailed
    
    var errorDescription: String? {
        switch self {
        case .noRouteFound:
            return "No route found between the selected locations"
        case .locationUnavailable:
            return "Your location is not available"
        case .routeCalculationFailed:
            return "Failed to calculate route"
        }
    }
}

// MARK: - MKPolyline Extension
extension MKPolyline {
    var coordinates: [CLLocationCoordinate2D] {
        var coords: [CLLocationCoordinate2D] = []
        let pointCount = pointCount
        let points = self.points()
        
        for i in 0..<pointCount {
            let point = points[i]
            coords.append(point.coordinate)
        }
        
        return coords
    }
}
