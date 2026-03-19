//
//  NavigationView.swift
//  RiskMapApp
//
//  Navigation view with safer route planning and export to Apple/Google Maps
//

import SwiftUI
import MapKit
import CoreLocation
import UIKit

// Wrapper to inject environment objects into RouteNavigationView
struct RouteNavigationViewWrapper: View {
    @EnvironmentObject var riskService: RiskService
    @EnvironmentObject var weatherService: WeatherService

    var body: some View {
        RouteNavigationView(riskService: riskService, weatherService: weatherService)
    }
}

struct RouteNavigationView: View {
    let riskService: RiskService
    let weatherService: WeatherService
    @StateObject private var routeService: RouteService
    @StateObject private var locationManager = LocationManager()
    @AppStorage("mapStyle") private var mapStyleRaw = "standard"
    @AppStorage("defaultRoutePreference") private var defaultRoutePreference = "safest"

    @State private var startPoint: String = ""
    @State private var destination: String = ""
    @State private var startCoordinate: CLLocationCoordinate2D?
    @State private var destinationCoordinate: CLLocationCoordinate2D?
    @State private var selectedRoute: Route?
    @State private var showRouteComparison = false
    @State private var currentWeather: WeatherData?

    @State private var cameraPosition = MapCameraPosition.region(
        MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
            span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
        )
    )

    init(riskService: RiskService, weatherService: WeatherService) {
        self.riskService = riskService
        self.weatherService = weatherService
        _routeService = StateObject(wrappedValue: RouteService(riskService: riskService, weatherService: weatherService))
    }

    private var navMapStyle: MapStyle {
        switch mapStyleRaw {
        case "satellite": return .imagery
        case "hybrid": return .hybrid
        default: return .standard
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                searchBarView
                ZStack {
                    mapView
                    routeLegend
                    weatherInfoBadge
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                if showRouteComparison, routeService.saferRoute != nil, routeService.optimalRoute != nil {
                    RouteComparisonCard(
                        saferRoute: routeService.saferRoute!,
                        optimalRoute: routeService.optimalRoute!,
                        selectedRoute: $selectedRoute,
                        routeSource: routeService.lastRouteSource,
                        onExportToAppleMaps: { exportToAppleMaps(route: selectedRoute ?? routeService.saferRoute!) },
                        onExportToGoogleMaps: { exportToGoogleMaps(route: selectedRoute ?? routeService.saferRoute!) },
                        currentWeather: currentWeather
                    )
                    .transition(.move(edge: .bottom))
                }
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("Navigation")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                locationManager.requestLocation()
                loadWeatherInfo()
            }
        }
    }

    @ViewBuilder
    private var weatherInfoBadge: some View {
        if let weather = currentWeather {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    weatherIcon(for: weather.condition)
                    Text(weather.condition.displayName)
                        .font(.caption)
                        .fontWeight(.medium)
                    Text("\(Int(weather.temperature))°C")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color(.systemBackground).opacity(0.95))
            .cornerRadius(10)
            .shadow(radius: 5)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(.top, 12)
            .padding(.leading, 12)
        }
    }

    @ViewBuilder
    private func weatherIcon(for condition: WeatherData.WeatherCondition) -> some View {
        Group {
            switch condition {
            case .clear: Image(systemName: "sun.max.fill")
            case .cloudy: Image(systemName: "cloud.fill")
            case .rain, .heavyRain: Image(systemName: "cloud.rain.fill")
            case .snow, .heavySnow: Image(systemName: "cloud.snow.fill")
            case .fog, .mist: Image(systemName: "cloud.fog.fill")
            case .thunderstorm: Image(systemName: "cloud.bolt.fill")
            case .sleet: Image(systemName: "cloud.sleet.fill")
            }
        }
        .foregroundColor(.blue)
    }

    @ViewBuilder
    private var routeLegend: some View {
        if routeService.saferRoute != nil || routeService.optimalRoute != nil {
            VStack(alignment: .leading, spacing: 8) {
                // Routing algorithm indicator
                HStack(spacing: 6) {
                    Image(systemName: (routeService.lastRouteSource ?? "") == "backend" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundColor((routeService.lastRouteSource ?? "") == "backend" ? .green : .orange)
                        .font(.caption)
                    Text((routeService.lastRouteSource ?? "") == "backend"
                         ? "Risk-aware routing active"
                         : "MapKit fallback (backend unavailable)")
                        .font(.caption2)
                        .foregroundColor((routeService.lastRouteSource ?? "") == "backend" ? .green : .orange)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background((routeService.lastRouteSource == "backend" ? Color.green : Color.orange).opacity(0.15))
                .cornerRadius(6)

                if routeService.optimalRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle().fill(.orange).frame(width: 20, height: 4)
                        Text("Fastest Route").font(.caption)
                    }
                }
                if routeService.saferRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle().fill(.blue).frame(width: 20, height: 4)
                        Text("Safest Route").font(.caption)
                    }
                }
                if selectedRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle().fill(.purple).frame(width: 20, height: 4)
                        Text("Selected").font(.caption)
                    }
                }
            }
            .padding(12)
            .background(Color(.systemBackground).opacity(0.9))
            .cornerRadius(10)
            .shadow(radius: 5)
            .padding(.top, 12)
            .padding(.trailing, 12)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
        }
    }

    @ViewBuilder
    private var mapView: some View {
        Map(position: $cameraPosition, interactionModes: [.pan, .zoom]) {
            mapAnnotations
            mapRoutes
        }
        .mapStyle(navMapStyle)
        .mapControls {
            MapUserLocationButton()
            MapCompass()
        }
        .ignoresSafeArea(edges: .top)
    }

    @MapContentBuilder
    private var mapAnnotations: some MapContent {
        if let userLocation = locationManager.location {
            Annotation("My Location", coordinate: userLocation.coordinate) {
                Image(systemName: "location.circle.fill")
                    .foregroundColor(.blue)
                    .font(.title)
                    .background(Circle().fill(Color.white))
            }
        }
        if let start = startCoordinate {
            Annotation("Start", coordinate: start) {
                Image(systemName: "mappin.circle.fill")
                    .foregroundColor(.green)
                    .font(.title)
            }
        }
        if let dest = destinationCoordinate {
            Annotation("Destination", coordinate: dest) {
                Image(systemName: "flag.circle.fill")
                    .foregroundColor(.red)
                    .font(.title)
            }
        }
    }

    @MapContentBuilder
    private var mapRoutes: some MapContent {
        if let start = startCoordinate, let dest = destinationCoordinate,
           routeService.saferRoute == nil && routeService.optimalRoute == nil {
            MapPolyline(coordinates: [start, dest])
                .stroke(.gray.opacity(0.5), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round, dash: [5, 5]))
        }

        if let optimalRoute = routeService.optimalRoute {
            let coords = optimalRoute.detailedCoordinates.filter { $0.latitude.isFinite && $0.longitude.isFinite }
            if !coords.isEmpty {
                let isSelected = selectedRoute?.id == optimalRoute.id
                MapPolyline(coordinates: coords)
                    .stroke(.white, style: StrokeStyle(lineWidth: isSelected ? 14 : 12, lineCap: .round, lineJoin: .round))
                MapPolyline(coordinates: coords)
                    .stroke(isSelected ? .purple : .orange, style: StrokeStyle(lineWidth: isSelected ? 10 : 8, lineCap: .round, lineJoin: .round))
            }
        }

        if let saferRoute = routeService.saferRoute {
            let coords = saferRoute.detailedCoordinates.filter { $0.latitude.isFinite && $0.longitude.isFinite }
            if !coords.isEmpty {
                let optimalCoords = routeService.optimalRoute?.detailedCoordinates ?? []
                let isDifferent = optimalCoords.count != coords.count || !areRoutesSimilar(optimalCoords, coords)
                let isSelected = selectedRoute?.id == saferRoute.id
                if isDifferent {
                    MapPolyline(coordinates: coords)
                        .stroke(.white, style: StrokeStyle(lineWidth: isSelected ? 16 : 14, lineCap: .round, lineJoin: .round))
                    MapPolyline(coordinates: coords)
                        .stroke(isSelected ? .purple : .blue, style: StrokeStyle(lineWidth: isSelected ? 12 : 10, lineCap: .round, lineJoin: .round))
                } else {
                    MapPolyline(coordinates: coords)
                        .stroke(.white, style: StrokeStyle(lineWidth: isSelected ? 14 : 12, lineCap: .round, lineJoin: .round))
                    MapPolyline(coordinates: coords)
                        .stroke(isSelected ? .purple : .blue.opacity(0.8), style: StrokeStyle(lineWidth: isSelected ? 10 : 8, lineCap: .round, lineJoin: .round, dash: [15, 8]))
                }
            }
        }
    }

    private func areRoutesSimilar(_ coords1: [CLLocationCoordinate2D], _ coords2: [CLLocationCoordinate2D]) -> Bool {
        guard coords1.count >= 2, coords2.count >= 2 else { return coords1.count == coords2.count }
        let start1 = coords1[0], end1 = coords1[coords1.count - 1]
        let start2 = coords2[0], end2 = coords2[coords2.count - 1]
        let startDist = CLLocation(latitude: start1.latitude, longitude: start1.longitude)
            .distance(from: CLLocation(latitude: start2.latitude, longitude: start2.longitude))
        let endDist = CLLocation(latitude: end1.latitude, longitude: end1.longitude)
            .distance(from: CLLocation(latitude: end2.latitude, longitude: end2.longitude))
        return startDist < 50 && endDist < 50
    }

    @ViewBuilder
    private var searchBarView: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Button(action: useCurrentLocationAsStart) {
                    Image(systemName: "location.fill")
                        .foregroundColor(.blue)
                }
                AddressSearchField(
                    placeholder: "Start location",
                    iconName: "mappin.circle.fill",
                    iconColor: .green,
                    text: $startPoint
                ) { coord in
                    startCoordinate = coord
                    updateCamera()
                }
            }
            .padding(.horizontal)

            Divider().padding(.leading, 48)

            AddressSearchField(
                placeholder: "Destination",
                iconName: "flag.fill",
                iconColor: .red,
                text: $destination
            ) { coord in
                destinationCoordinate = coord
                updateCamera()
            }
            .padding(.horizontal)

            Button(action: calculateRoutesAction) {
                HStack {
                    if routeService.isLoading {
                        ProgressView().progressViewStyle(CircularProgressViewStyle(tint: .white))
                        Text("Calculating...")
                    } else {
                        Image(systemName: "arrow.triangle.2.circlepath")
                        Text("Find Safest Route")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background((startCoordinate != nil && destinationCoordinate != nil) ? Color.blue : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
            .disabled(routeService.isLoading || startCoordinate == nil || destinationCoordinate == nil)

            if let error = routeService.errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.horizontal)
                    .padding(.bottom, 4)
            }
        }
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 5)
        .padding()
    }

    private func loadWeatherInfo() {
        Task {
            let location = startCoordinate ?? CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
            let weather = await weatherService.getWeatherData(for: location)
            await MainActor.run { currentWeather = weather }
        }
    }

    private func useCurrentLocationAsStart() {
        if let location = locationManager.location {
            startCoordinate = location.coordinate
            startPoint = "Current Location"
            updateCamera()
        }
    }

    private func calculateRoutesAction() {
        guard let start = startCoordinate, let dest = destinationCoordinate else {
            routeService.errorMessage = "Please enter both start and destination"
            return
        }

        Task {
            let center = CLLocationCoordinate2D(
                latitude: (start.latitude + dest.latitude) / 2,
                longitude: (start.longitude + dest.longitude) / 2
            )
            let weather = await weatherService.getWeatherData(for: center)
            await MainActor.run { currentWeather = weather }
        }

        Task {
            await routeService.calculateRoutes(from: start, to: dest)
            await MainActor.run {
                if routeService.saferRoute != nil && routeService.optimalRoute != nil {
                    showRouteComparison = true
                    selectedRoute = defaultRoutePreference == "fastest" ? routeService.optimalRoute : routeService.saferRoute
                    updateCameraForRoutes()
                }
            }
        }
    }

    private func updateCamera() {
        guard let start = startCoordinate else {
            if let dest = destinationCoordinate {
                cameraPosition = .region(MKCoordinateRegion(center: dest, span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)))
            }
            return
        }
        if let dest = destinationCoordinate {
            let minLat = min(start.latitude, dest.latitude)
            let maxLat = max(start.latitude, dest.latitude)
            let minLon = min(start.longitude, dest.longitude)
            let maxLon = max(start.longitude, dest.longitude)
            let center = CLLocationCoordinate2D(latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2)
            cameraPosition = .region(MKCoordinateRegion(
                center: center,
                span: MKCoordinateSpan(
                    latitudeDelta: max((maxLat - minLat) * 1.5, 0.01),
                    longitudeDelta: max((maxLon - minLon) * 1.5, 0.01)
                )
            ))
        } else {
            cameraPosition = .region(MKCoordinateRegion(center: start, span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)))
        }
    }

    private func updateCameraForRoutes() {
        guard let safer = routeService.saferRoute, let optimal = routeService.optimalRoute else { return }
        let allCoords = safer.detailedCoordinates + optimal.detailedCoordinates
        let validCoords = allCoords.filter { $0.latitude.isFinite && $0.longitude.isFinite }
        guard !validCoords.isEmpty else { return }

        let lats = validCoords.map { $0.latitude }
        let lons = validCoords.map { $0.longitude }
        guard let minLat = lats.min(), let maxLat = lats.max(), let minLon = lons.min(), let maxLon = lons.max() else { return }

        let center = CLLocationCoordinate2D(latitude: (minLat + maxLat) / 2, longitude: (minLon + maxLon) / 2)
        let span = MKCoordinateSpan(
            latitudeDelta: max((maxLat - minLat) * 1.3, 0.01),
            longitudeDelta: max((maxLon - minLon) * 1.3, 0.01)
        )
        cameraPosition = .region(MKCoordinateRegion(center: center, span: span))
    }

    private func exportToAppleMaps(route: Route) {
        guard let start = startCoordinate, let dest = destinationCoordinate else { return }
        let startItem = MKMapItem(placemark: MKPlacemark(coordinate: start))
        startItem.name = startPoint.isEmpty ? "Start" : startPoint
        let destItem = MKMapItem(placemark: MKPlacemark(coordinate: dest))
        destItem.name = destination.isEmpty ? "Destination" : destination
        let waypointCoords = routeShapingWaypoints(for: route, maxWaypoints: 8)
        let waypointItems = waypointCoords.enumerated().map { index, coord in
            let item = MKMapItem(placemark: MKPlacemark(coordinate: coord))
            item.name = "Route waypoint \(index + 1)"
            return item
        }
        MKMapItem.openMaps(
            with: [startItem] + waypointItems + [destItem],
            launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving]
        )
    }

    private func exportToGoogleMaps(route: Route) {
        guard let start = startCoordinate, let dest = destinationCoordinate else { return }
        let maxWaypoints = 23
        let waypoints = routeShapingWaypoints(for: route, maxWaypoints: maxWaypoints)
            .map { "\($0.latitude),\($0.longitude)" }

        let limited = Array(waypoints.prefix(maxWaypoints))
        var url = "https://www.google.com/maps/dir/?api=1&origin=\(start.latitude),\(start.longitude)&destination=\(dest.latitude),\(dest.longitude)&travelmode=driving"
        if !limited.isEmpty {
            url += "&waypoints=\(limited.joined(separator: "|"))"
        }
        if let encoded = url.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed), let u = URL(string: encoded) {
            UIApplication.shared.open(u)
        }
    }

    private func routeShapingWaypoints(for route: Route, maxWaypoints: Int) -> [CLLocationCoordinate2D] {
        let coords = route.polyline.coordinates.filter { $0.latitude.isFinite && $0.longitude.isFinite }
        guard coords.count >= 3, maxWaypoints > 0 else { return [] }

        let minSpacingMeters: CLLocationDistance = 350
        var selected: [CLLocationCoordinate2D] = []
        var lastChosen = CLLocation(latitude: coords[0].latitude, longitude: coords[0].longitude)

        // Prefer meaningful bends/turns first so external navigation apps are nudged
        // toward the selected route shape rather than arbitrary evenly spaced points.
        for index in 1..<(coords.count - 1) {
            let prev = coords[index - 1]
            let curr = coords[index]
            let next = coords[index + 1]

            let turnAngle = abs(headingDelta(
                heading(from: prev, to: curr),
                heading(from: curr, to: next)
            ))
            let currLocation = CLLocation(latitude: curr.latitude, longitude: curr.longitude)
            let spacing = currLocation.distance(from: lastChosen)

            if turnAngle >= 25, spacing >= minSpacingMeters {
                selected.append(curr)
                lastChosen = currLocation
                if selected.count >= maxWaypoints { break }
            }
        }

        if selected.count < maxWaypoints {
            let evenlySpaced = evenlySpacedWaypoints(
                coordinates: coords,
                count: maxWaypoints - selected.count
            )
            for coord in evenlySpaced {
                if selected.count >= maxWaypoints { break }
                if !containsNearbyCoordinate(selected, coord: coord, thresholdMeters: 120) {
                    selected.append(coord)
                }
            }
        }

        return selected.prefix(maxWaypoints).sorted {
            routeDistanceAlong(coords, to: $0) < routeDistanceAlong(coords, to: $1)
        }
    }

    private func evenlySpacedWaypoints(coordinates: [CLLocationCoordinate2D], count: Int) -> [CLLocationCoordinate2D] {
        guard coordinates.count >= 3, count > 0 else { return [] }
        let interior = coordinates.count - 2
        guard interior > 0 else { return [] }

        let step = Double(coordinates.count - 1) / Double(count + 1)
        return (1...count).compactMap { i in
            let idx = Int(round(step * Double(i)))
            guard idx > 0 && idx < coordinates.count - 1 else { return nil }
            return coordinates[idx]
        }
    }

    private func containsNearbyCoordinate(_ coordinates: [CLLocationCoordinate2D], coord: CLLocationCoordinate2D, thresholdMeters: CLLocationDistance) -> Bool {
        let location = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
        return coordinates.contains {
            CLLocation(latitude: $0.latitude, longitude: $0.longitude).distance(from: location) < thresholdMeters
        }
    }

    private func routeDistanceAlong(_ route: [CLLocationCoordinate2D], to target: CLLocationCoordinate2D) -> CLLocationDistance {
        guard route.count >= 2 else { return 0 }
        var total: CLLocationDistance = 0
        var bestDistance = CLLocationDistance.greatestFiniteMagnitude
        var bestAlong: CLLocationDistance = 0

        for i in 0..<(route.count - 1) {
            let start = CLLocation(latitude: route[i].latitude, longitude: route[i].longitude)
            let end = CLLocation(latitude: route[i + 1].latitude, longitude: route[i + 1].longitude)
            let segmentDistance = start.distance(from: end)
            let targetDistance = start.distance(from: CLLocation(latitude: target.latitude, longitude: target.longitude))
            if targetDistance < bestDistance {
                bestDistance = targetDistance
                bestAlong = total
            }
            total += segmentDistance
        }
        return bestAlong
    }

    private func heading(from start: CLLocationCoordinate2D, to end: CLLocationCoordinate2D) -> Double {
        let dy = end.latitude - start.latitude
        let dx = end.longitude - start.longitude
        return atan2(dy, dx) * 180 / .pi
    }

    private func headingDelta(_ a: Double, _ b: Double) -> Double {
        var delta = b - a
        while delta > 180 { delta -= 360 }
        while delta < -180 { delta += 360 }
        return delta
    }
}

// MARK: - Route Comparison Card
struct RouteComparisonCard: View {
    let saferRoute: Route
    let optimalRoute: Route
    @Binding var selectedRoute: Route?
    let routeSource: String?
    let onExportToAppleMaps: () -> Void
    let onExportToGoogleMaps: () -> Void
    let currentWeather: WeatherData?
    @State private var isExpanded = true

    private var comparison: RouteComparison {
        RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)
    }

    private var routesAreSame: Bool {
        let coordDiff = abs(saferRoute.polyline.coordinates.count - optimalRoute.polyline.coordinates.count)
        let distanceDiff = abs(saferRoute.distance - optimalRoute.distance)
        return coordDiff < max(saferRoute.polyline.coordinates.count, optimalRoute.polyline.coordinates.count) / 10 && distanceDiff < 100
    }

    private var routesSameMessage: String {
        if routeSource == "mapkit" {
            return "Backend routing unavailable — using MapKit fallback. Start the backend (python app.py) and ensure the model is trained, then try routes in Toronto."
        }
        return "Only one route available — safest and fastest are the same for this path."
    }

    var body: some View {
        VStack(spacing: 0) {
            Button(action: { withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) { isExpanded.toggle() } }) {
                HStack {
                    Text("Route Options").font(.headline)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if isExpanded {
                ScrollView {
                    VStack(spacing: 16) {
                        // Routing algorithm status
                        HStack(spacing: 8) {
                            Image(systemName: (routeSource ?? "") == "backend" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundColor((routeSource ?? "") == "backend" ? .green : .orange)
                            Text((routeSource ?? "") == "backend"
                                 ? "Risk-aware routing: Dijkstra on time + crash risk (Toronto graph)"
                                 : "MapKit fallback: Backend unavailable — routes may be identical")
                                .font(.caption)
                                .foregroundColor((routeSource ?? "") == "backend" ? .secondary : .orange)
                            Spacer()
                        }
                        .padding()
                        .background((routeSource ?? "") == "backend" ? Color.green.opacity(0.08) : Color.orange.opacity(0.08))
                        .cornerRadius(8)

                        if routesAreSame {
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: "info.circle.fill").foregroundColor(.orange)
                                Text(routesSameMessage)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding()
                            .background(Color.orange.opacity(0.1))
                            .cornerRadius(8)
                        }

                        RouteOptionCard(
                            route: saferRoute,
                            title: "Safest Route",
                            subtitle: routesAreSame ? "Same as fastest route" : saferRoute.safetyExplanation(comparedTo: optimalRoute),
                            color: .blue,
                            isSelected: selectedRoute?.id == saferRoute.id
                        ) { selectedRoute = saferRoute }

                        RouteOptionCard(
                            route: optimalRoute,
                            title: "Fastest Route",
                            subtitle: routesAreSame ? "Same as safest route" : "Shortest travel time",
                            color: .orange,
                            isSelected: selectedRoute?.id == optimalRoute.id
                        ) { selectedRoute = optimalRoute }

                        if let weather = currentWeather {
                            HStack(spacing: 12) {
                                Image(systemName: "cloud.fill").foregroundColor(.blue)
                                Text("\(weather.condition.displayName) • \(timeOfDayLabel())")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                            }
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(10)
                        }

                        HStack(spacing: 8) {
                            Image(systemName: "clock")
                            Text("Time diff: \(formatTime(comparison.timeDifference))")
                                .foregroundColor(comparison.saferRouteSlower ? .orange : .green)
                            Image(systemName: "shield.checkered")
                            Text(comparison.safetyImprovement).foregroundColor(.blue)
                        }
                        .font(.subheadline)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)

                        HStack(spacing: 12) {
                            Button(action: onExportToAppleMaps) {
                                Label("Apple Maps", systemImage: "map.fill")
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            Button(action: onExportToGoogleMaps) {
                                Label("Google Maps", systemImage: "globe")
                                    .frame(maxWidth: .infinity)
                                    .padding()
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                    }
                    .padding(.bottom, 16)
                }
                .frame(maxHeight: 380)
            }
        }
        .background(Color(.systemBackground))
        .cornerRadius(16, corners: [.topLeft, .topRight])
        .shadow(radius: 10)
    }

    private func timeOfDayLabel() -> String {
        let calendar = Calendar.current
        let now = Date()
        let hour = calendar.component(.hour, from: now)
        let weekday = calendar.component(.weekday, from: now)
        let isWeekend = weekday == 1 || weekday == 7
        if hour >= 23 || hour < 5 { return "Late Night" }
        if hour >= 22 { return "Night" }
        if hour < 7 { return "Early Morning" }
        if hour >= 7 && hour < 10 { return isWeekend ? "Morning" : "Rush Hour" }
        if hour >= 10 && hour < 16 { return "Daytime" }
        if hour >= 16 && hour < 19 { return isWeekend ? "Evening" : "Rush Hour" }
        return "Evening"
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds / 60)
        return minutes < 60 ? "\(minutes) min" : "\(minutes / 60)h \(minutes % 60)m"
    }
}

// MARK: - Route Option Card
struct RouteOptionCard: View {
    let route: Route
    let title: String
    let subtitle: String
    let color: Color
    let isSelected: Bool
    let onTap: () -> Void
    @AppStorage("distanceUnits") private var distanceUnits = "km"

    var body: some View {
        Button(action: onTap) {
            HStack {
                RoundedRectangle(cornerRadius: 4).fill(color).frame(width: 4, height: 50)
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(title).font(.headline)
                        if isSelected { Image(systemName: "checkmark.circle.fill").foregroundColor(.green) }
                    }
                    Text(subtitle).font(.caption).foregroundColor(.secondary)
                    HStack(spacing: 16) {
                        Label(formatTime(route.estimatedTime), systemImage: "clock")
                        Label(formatDistance(route.distance, units: distanceUnits), systemImage: "ruler")
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                    HStack(spacing: 12) {
                        if route.highRiskSegments > 0 {
                            Label("\(route.highRiskSegments) high risk", systemImage: "exclamationmark.triangle.fill")
                                .font(.caption2).foregroundColor(.red)
                        }
                        if route.mediumRiskSegments > 0 {
                            Label("\(route.mediumRiskSegments) medium", systemImage: "exclamationmark.circle.fill")
                                .font(.caption2).foregroundColor(.orange)
                        }
                        if route.lowRiskSegments > 0 {
                            Label("\(route.lowRiskSegments) low risk", systemImage: "checkmark.circle.fill")
                                .font(.caption2).foregroundColor(.green)
                        }
                    }
                }
                Spacer()
            }
            .padding()
            .background(isSelected ? color.opacity(0.1) : Color(.systemGray6))
            .cornerRadius(10)
        }
        .buttonStyle(.plain)
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds / 60)
        return minutes < 60 ? "\(minutes) min" : "\(minutes / 60)h \(minutes % 60)m"
    }

    private func formatDistance(_ meters: CLLocationDistance, units: String = "km") -> String {
        if units == "mi" {
            let miles = meters / 1609.34
            return miles < 1 ? String(format: "%.0f ft", meters * 3.28084) : String(format: "%.1f mi", miles)
        }
        return meters < 1000 ? String(format: "%.0f m", meters) : String(format: "%.1f km", meters / 1000)
    }
}

// MARK: - View Extension
extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        Path(UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        ).cgPath)
    }
}
