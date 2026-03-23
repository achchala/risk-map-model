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
    /// "backend" when using Toronto graph routing; "mapkit" when fallback. Use to explain why routes may be identical.
    @Published var lastRouteSource: String? = nil
    /// True when origin/destination is outside the Toronto road network; show "out of bounds" message.
    @Published var isOutOfBounds: Bool = false

    private let riskService: RiskService
    private let weatherService: WeatherService
    private var activeRouteRequestID = UUID()

    init(riskService: RiskService, weatherService: WeatherService = WeatherService()) {
        self.riskService = riskService
        self.weatherService = weatherService
    }

    func calculateRoutes(
        from start: CLLocationCoordinate2D,
        to destination: CLLocationCoordinate2D,
        weather: WeatherData? = nil,
        timeOfDay: (hour: Int, isWeekend: Bool)? = nil
    ) async {
        let requestID = UUID()
        await MainActor.run {
            self.activeRouteRequestID = requestID
            isLoading = true
            errorMessage = nil
            lastRouteSource = nil
            isOutOfBounds = false
        }

        do {
            // Always use MapKit for the fastest route so displayed "fastest" time matches
            // Apple's ETA. Use backend only for the safer alternative when available.
            let optimalMKRoute = try await withTimeout(seconds: 30) {
                try await self.calculateOptimalRoute(from: start, to: destination)
            }

            var analyzedOptimal = try await withTimeout(seconds: 20) {
                try await self.analyzeRoute(optimalMKRoute, type: .optimal)
            }
            analyzedOptimal.estimatedTime = nil

            do {
                let safer = try await fetchBackendSaferRoute(from: start, to: destination, weather: weather, timeOfDay: timeOfDay)
                var analyzedSafer = safer
                analyzedSafer.estimatedTime = nil
                let finalOptimal = analyzedOptimal
                let finalSafer = analyzedSafer
                await MainActor.run {
                    self.optimalRoute = finalOptimal
                    self.saferRoute = finalSafer
                    self.lastRouteSource = "backend"
                    self.isLoading = false
                }
                refreshScrapedGoogleMapsETAs(requestID: requestID, start: start, destination: destination, optimalRoute: finalOptimal, saferRoute: finalSafer)
                return
            } catch APIError.outOfBounds {
                await MainActor.run { self.isOutOfBounds = true }
                // Fall through to MapKit fallback
            } catch {
                // Other errors (network, etc.) — also fall back to MapKit
            }

            // Fallback: MapKit for both (outside Toronto or backend unavailable)
            let saferMKRoute = try await withTimeout(seconds: 30) {
                try await self.calculateSaferRoute(from: start, to: destination, timePenaltyFactor: self.currentTimePenaltyFactor)
            }
            var analyzedSafer = try await withTimeout(seconds: 20) {
                try await self.analyzeRoute(saferMKRoute, type: .safer)
            }
            analyzedSafer.estimatedTime = nil

            let finalOptimal = analyzedOptimal
            let finalSafer = analyzedSafer
            await MainActor.run {
                self.optimalRoute = finalOptimal
                self.saferRoute = finalSafer
                self.lastRouteSource = "mapkit"
                self.isLoading = false
            }
            refreshScrapedGoogleMapsETAs(requestID: requestID, start: start, destination: destination, optimalRoute: finalOptimal, saferRoute: finalSafer)
        } catch {
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
            }
        }
    }

    private func refreshScrapedGoogleMapsETAs(
        requestID: UUID,
        start: CLLocationCoordinate2D,
        destination: CLLocationCoordinate2D,
        optimalRoute: Route,
        saferRoute: Route
    ) {
        Task { [weak self] in
            guard let self else { return }
            var namedURLs: [String: String] = [:]
            if let url = optimalRoute.googleMapsURL(origin: start, destination: destination) {
                namedURLs["fastest"] = url
            }
            if let url = saferRoute.googleMapsURL(origin: start, destination: destination) {
                namedURLs["safer"] = url
            }
            guard !namedURLs.isEmpty else { return }

            let response = await fetchScrapedGoogleMapsETAs(namedURLs: namedURLs)
            await MainActor.run {
                guard self.activeRouteRequestID == requestID else { return }
                if let eta = response.etasSeconds["fastest"],
                   var updated = self.optimalRoute {
                    updated.estimatedTime = eta
                    self.optimalRoute = updated
                }
                if let eta = response.etasSeconds["safer"],
                   var updated = self.saferRoute {
                    updated.estimatedTime = eta
                    self.saferRoute = updated
                }
            }
        }
    }

    private func fetchScrapedGoogleMapsETAs(namedURLs: [String: String]) async -> GoogleMapsETAResponse {
        do {
            let response = try await withTimeout(seconds: 35) {
                try await self.riskService.fetchGoogleMapsETAs(namedURLs: namedURLs)
            }
            print("[google-eta] applying ETAs=\(response.etasSeconds) sources=\(response.sources ?? [:]) failures=\(response.failures)")
            return response
        } catch {
            print("[google-eta] scrape request failed: \(error.localizedDescription)")
            return GoogleMapsETAResponse(etasSeconds: [:], failures: [:], sources: nil)
        }
    }

    private var safetySliderValue: Double {
        let key = "safetySpeedBalanceSlider"
        if UserDefaults.standard.object(forKey: key) == nil {
            return 0.5
        }
        return UserDefaults.standard.double(forKey: key)
    }

    private var currentBeta: Double {
        SafetySpeedBalance.betaFromSlider(safetySliderValue)
    }

    private var currentTimePenaltyFactor: Double {
        SafetySpeedBalance.timePenaltyFromSlider(safetySliderValue)
    }

    /// Fetch the backend safer route while keeping MapKit as the source of truth for the
    /// fastest route and its ETA.
    private func fetchBackendSaferRoute(
        from start: CLLocationCoordinate2D,
        to destination: CLLocationCoordinate2D,
        weather: WeatherData? = nil,
        timeOfDay: (hour: Int, isWeekend: Bool)? = nil
    ) async throws -> Route {
        let routeCenter = CLLocationCoordinate2D(
            latitude: (start.latitude + destination.latitude) / 2,
            longitude: (start.longitude + destination.longitude) / 2
        )
        let effectiveWeather: WeatherData
        if let w = weather {
            effectiveWeather = w
        } else {
            effectiveWeather = await weatherService.getWeatherData(for: routeCenter)
        }
        let beta = currentBeta

        let response = try await riskService.fetchSafetyAwareRoutes(
            origin: start,
            destination: destination,
            weather: effectiveWeather,
            timeOfDay: timeOfDay,
            beta: beta
        )

        let saferRoute = Route(
            routeOption: response.safer,
            routeType: .safer,
            estimatedTimeOverride: nil
        )

        return saferRoute
    }

    private func withTimeout<T>(seconds: TimeInterval, operation: @escaping () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw RouteError.routeCalculationFailed
            }
            defer { group.cancelAll() }
            return try await group.next()!
        }
    }

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

    private func calculateSaferRoute(from start: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D, timePenaltyFactor: Double = 1.0) async throws -> MKRoute {
        let request = MKDirections.Request()
        request.source = MKMapItem(placemark: MKPlacemark(coordinate: start))
        request.destination = MKMapItem(placemark: MKPlacemark(coordinate: destination))
        request.transportType = .automobile
        request.requestsAlternateRoutes = true

        let directions = MKDirections(request: request)
        let response = try await directions.calculate()

        let routes = response.routes
        guard !routes.isEmpty else { throw RouteError.noRouteFound }

        if routes.count == 1 {
            return routes[0]
        }

        let routeCenter = CLLocationCoordinate2D(
            latitude: (start.latitude + destination.latitude) / 2,
            longitude: (start.longitude + destination.longitude) / 2
        )
        let weather = await weatherService.getWeatherData(for: routeCenter)

        var allCoords: [CLLocationCoordinate2D] = []
        for route in routes.prefix(5) {
            allCoords.append(contentsOf: route.polyline.coordinates)
        }

        guard !allCoords.isEmpty else { return routes[0] }

        let minLat = allCoords.map { $0.latitude }.min()!
        let maxLat = allCoords.map { $0.latitude }.max()!
        let minLon = allCoords.map { $0.longitude }.min()!
        let maxLon = allCoords.map { $0.longitude }.max()!

        let combinedRegion = MKCoordinateRegion(
            center: CLLocationCoordinate2D(
                latitude: (minLat + maxLat) / 2,
                longitude: (minLon + maxLon) / 2
            ),
            span: MKCoordinateSpan(
                latitudeDelta: (maxLat - minLat) * 1.5,
                longitudeDelta: (maxLon - minLon) * 1.5
            )
        )

        let riskSegments: [RoadSegment]
        do {
            riskSegments = try await riskService.fetchRiskPredictions(for: combinedRegion, weather: weather)
        } catch {
            return routes.count > 1 ? routes[1] : routes[0]
        }

        let limitedRiskSegments = Array(riskSegments.prefix(5000))
        let minTime = routes.map { $0.expectedTravelTime }.min() ?? 1

        var routeScores: [(route: MKRoute, score: Double)] = []
        for route in routes.prefix(5) {
            let riskScore = calculateRouteRiskScore(route: route, riskSegments: limitedRiskSegments)
            let timeRatio = route.expectedTravelTime / minTime
            let timePenalty = timePenaltyFactor * max(0, timeRatio - 1.0)
            let adjustedScore = riskScore + timePenalty
            routeScores.append((route: route, score: adjustedScore))
        }

        routeScores.sort { $0.score < $1.score }
        return routeScores.first?.route ?? routes[0]
    }

    private func calculateRouteRiskScore(route: MKRoute, riskSegments: [RoadSegment]) -> Double {
        let routePoints = samplePointsAlongRoute(route)
        guard !routePoints.isEmpty else { return 0.0 }

        var totalRiskScore = 0.0
        var pointCount = 0

        for point in routePoints {
            if let nearestSegment = findNearestSegment(to: point, in: riskSegments) {
                let riskWeight: Double
                switch nearestSegment.riskLevel {
                case .high: riskWeight = 3.0
                case .medium: riskWeight = 2.0
                case .low: riskWeight = 1.0
                }
                totalRiskScore += riskWeight
                pointCount += 1
            }
        }

        return pointCount > 0 ? totalRiskScore / Double(pointCount) : 0.0
    }

    private func analyzeRoute(_ route: MKRoute, type: Route.RouteType) async throws -> Route {
        let routeRegion = createRegionForRoute(route)
        let routeCenter = routeRegion.center
        let weather = await weatherService.getWeatherData(for: routeCenter)

        let riskSegments: [RoadSegment]
        do {
            riskSegments = try await riskService.fetchRiskPredictions(for: routeRegion, weather: weather)
        } catch {
            return Route(
                mkRoute: route,
                riskScore: 0.0,
                highRiskSegments: 0,
                mediumRiskSegments: 0,
                lowRiskSegments: 0,
                routeType: type
            )
        }

        let routePoints = samplePointsAlongRoute(route)
        let sampledPoints = Array(routePoints.prefix(50))
        let limitedRiskSegments = Array(riskSegments.prefix(5000))

        var highRiskCount = 0
        var mediumRiskCount = 0
        var lowRiskCount = 0
        var totalRiskScore = 0.0
        var matchedPoints = 0

        for point in sampledPoints {
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
            mkRoute: route,
            riskScore: averageRiskScore,
            highRiskSegments: highRiskCount,
            mediumRiskSegments: mediumRiskCount,
            lowRiskSegments: lowRiskCount,
            routeType: type
        )
    }

    private func createRegionForRoute(_ route: MKRoute) -> MKCoordinateRegion {
        let coordinates = route.polyline.coordinates
        guard !coordinates.isEmpty else {
            return MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
                span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
            )
        }

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
        let latDelta = max((maxLat - minLat) * 1.5, 0.01)
        let lonDelta = max((maxLon - minLon) * 1.5, 0.01)

        guard center.latitude.isFinite && center.longitude.isFinite,
              latDelta.isFinite && lonDelta.isFinite else {
            return MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
                span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
            )
        }

        return MKCoordinateRegion(
            center: center,
            span: MKCoordinateSpan(latitudeDelta: latDelta, longitudeDelta: lonDelta)
        )
    }

    private func samplePointsAlongRoute(_ route: MKRoute) -> [CLLocationCoordinate2D] {
        let coordinates = route.polyline.coordinates
        guard !coordinates.isEmpty else { return [] }

        let maxCoordinates = 500
        if coordinates.count <= maxCoordinates {
            return samplePointsEvery(coordinates: coordinates, distance: 100, maxPoints: 100)
        } else {
            let step = max(1, coordinates.count / 100)
            return stride(from: 0, to: coordinates.count, by: step).map { coordinates[$0] }
        }
    }

    private func samplePointsEvery(coordinates: [CLLocationCoordinate2D], distance: CLLocationDistance, maxPoints: Int) -> [CLLocationCoordinate2D] {
        var sampledPoints: [CLLocationCoordinate2D] = []
        var accumulatedDistance: CLLocationDistance = 0

        for i in 0..<coordinates.count - 1 {
            let start = coordinates[i]
            let end = coordinates[i + 1]
            let segmentDistance = CLLocation(latitude: start.latitude, longitude: start.longitude)
                .distance(from: CLLocation(latitude: end.latitude, longitude: end.longitude))

            guard segmentDistance > 0.001 else {
                accumulatedDistance += segmentDistance
                continue
            }

            var segmentAccumulated: CLLocationDistance = 0
            while segmentAccumulated + distance <= segmentDistance && sampledPoints.count < maxPoints {
                let ratio = (segmentAccumulated + distance) / segmentDistance
                guard ratio.isFinite && ratio >= 0 && ratio <= 1 else { break }
                let lat = start.latitude + (end.latitude - start.latitude) * ratio
                let lon = start.longitude + (end.longitude - start.longitude) * ratio
                guard lat.isFinite && lon.isFinite else { break }
                sampledPoints.append(CLLocationCoordinate2D(latitude: lat, longitude: lon))
                segmentAccumulated += distance
                accumulatedDistance += distance
            }
            accumulatedDistance += segmentDistance
        }

        if let last = coordinates.last, !sampledPoints.contains(where: {
            abs($0.latitude - last.latitude) < 0.0001 && abs($0.longitude - last.longitude) < 0.0001
        }) {
            sampledPoints.append(last)
        }

        return sampledPoints.isEmpty ? Array(coordinates.prefix(maxPoints)) : sampledPoints
    }

    private func findNearestSegment(to point: CLLocationCoordinate2D, in segments: [RoadSegment]) -> RoadSegment? {
        let pointLocation = CLLocation(latitude: point.latitude, longitude: point.longitude)
        var nearestSegment: RoadSegment?
        var minDistance: CLLocationDistance = 50.0
        let maxDistance: CLLocationDistance = 50.0
        let segmentsToCheck = Array(segments.prefix(1000))

        for segment in segmentsToCheck {
            guard let firstCoord = segment.coordinates.first else { continue }
            let firstLocation = CLLocation(latitude: firstCoord.latitude, longitude: firstCoord.longitude)
            if pointLocation.distance(from: firstLocation) > maxDistance * 2 { continue }

            for coord in segment.coordinates {
                let segmentLocation = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
                let distance = pointLocation.distance(from: segmentLocation)
                if distance < minDistance {
                    minDistance = distance
                    nearestSegment = segment
                    if distance < 10 { return nearestSegment }
                }
            }
        }

        return minDistance < maxDistance ? nearestSegment : nil
    }
}

// MARK: - Route Errors
enum RouteError: Error, LocalizedError {
    case noRouteFound
    case locationUnavailable
    case routeCalculationFailed

    var errorDescription: String? {
        switch self {
        case .noRouteFound: return "No route found between the selected locations"
        case .locationUnavailable: return "Your location is not available"
        case .routeCalculationFailed: return "Failed to calculate route"
        }
    }
}

// MARK: - MKPolyline Extension
extension MKPolyline {
    var coordinates: [CLLocationCoordinate2D] {
        var coords: [CLLocationCoordinate2D] = []
        let pointCount = self.pointCount
        guard pointCount > 0 else { return coords }

        let points = self.points()
        for i in 0..<pointCount {
            let point = points[i]
            let coord = point.coordinate
            if coord.latitude.isFinite && coord.longitude.isFinite &&
               coord.latitude >= -90 && coord.latitude <= 90 &&
               coord.longitude >= -180 && coord.longitude <= 180 {
                coords.append(coord)
            }
        }
        return coords
    }
}
