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
                        onExportToAppleMaps: exportToAppleMaps,
                        onExportToGoogleMaps: exportToGoogleMaps,
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
        .foregroundColor(.brandTertiary)
    }

    @ViewBuilder
    private var routeLegend: some View {
        if routeService.saferRoute != nil || routeService.optimalRoute != nil {
            VStack(alignment: .leading, spacing: 8) {
                if routeService.optimalRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle().fill(Color.routeFastestYellow).frame(width: 20, height: 4)
                        Text("Fastest Route").font(.caption)
                    }
                }
                if routeService.saferRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle().fill(Color.routeSafestBlue).frame(width: 20, height: 4)
                        Text("Safest Route").font(.caption)
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
                    .foregroundColor(.brandTertiary)
                    .font(.title)
                    .background(Circle().fill(Color.white))
            }
        }
        if let start = startCoordinate {
            Annotation("Start", coordinate: start) {
                Image(systemName: "mappin.circle.fill")
                    .foregroundColor(.brandPrimary)
                    .font(.title)
            }
        }
        if let dest = destinationCoordinate {
            Annotation("Destination", coordinate: dest) {
                Image(systemName: "flag.circle.fill")
                    .foregroundColor(.brandSecondary)
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

        // Draw order: non-selected routes first (bottom), then selected route (on top), then dashed green overlay
        let optimalCoords = routeService.optimalRoute?.detailedCoordinates.filter { $0.latitude.isFinite && $0.longitude.isFinite } ?? []
        let saferCoords = routeService.saferRoute?.detailedCoordinates.filter { $0.latitude.isFinite && $0.longitude.isFinite } ?? []
        let routesAreSame = !optimalCoords.isEmpty && !saferCoords.isEmpty && areRoutesSimilar(optimalCoords, saferCoords)
        let optimalSelected = selectedRoute?.id == routeService.optimalRoute?.id
        let saferSelected = selectedRoute?.id == routeService.saferRoute?.id

        // 1. Non-selected route(s) drawn first (underneath) — faded colors
        if let optimalRoute = routeService.optimalRoute, !optimalCoords.isEmpty, !optimalSelected {
            MapPolyline(coordinates: optimalCoords)
                .stroke(.white, style: StrokeStyle(lineWidth: 12, lineCap: .round, lineJoin: .round))
            MapPolyline(coordinates: optimalCoords)
                .stroke(Color.routeFastestYellowFaded, style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round))
        }
        if let saferRoute = routeService.saferRoute, !saferCoords.isEmpty, !saferSelected {
            MapPolyline(coordinates: saferCoords)
                .stroke(.white, style: StrokeStyle(lineWidth: routesAreSame ? 12 : 14, lineCap: .round, lineJoin: .round))
            MapPolyline(coordinates: saferCoords)
                .stroke(Color.routeSafestBlueFaded, style: StrokeStyle(lineWidth: routesAreSame ? 8 : 10, lineCap: .round, lineJoin: .round, dash: routesAreSame ? [15, 8] : []))
        }

        // 2. Selected route drawn second (on top of non-selected) — full colors
        if let optimalRoute = routeService.optimalRoute, !optimalCoords.isEmpty, optimalSelected {
            MapPolyline(coordinates: optimalCoords)
                .stroke(.white, style: StrokeStyle(lineWidth: 14, lineCap: .round, lineJoin: .round))
            MapPolyline(coordinates: optimalCoords)
                .stroke(Color.routeFastestYellow, style: StrokeStyle(lineWidth: 10, lineCap: .round, lineJoin: .round))
        }
        if let saferRoute = routeService.saferRoute, !saferCoords.isEmpty, saferSelected {
            MapPolyline(coordinates: saferCoords)
                .stroke(.white, style: StrokeStyle(lineWidth: routesAreSame ? 14 : 16, lineCap: .round, lineJoin: .round))
            MapPolyline(coordinates: saferCoords)
                .stroke(Color.routeSafestBlue, style: StrokeStyle(lineWidth: routesAreSame ? 10 : 12, lineCap: .round, lineJoin: .round, dash: routesAreSame ? [15, 8] : []))
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
                        .foregroundColor(.brandTertiary)
                }
                AddressSearchField(
                    placeholder: "Start location",
                    iconName: "mappin.circle.fill",
                    iconColor: .brandPrimary,
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
                iconColor: .brandSecondary,
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
                        Image(systemName: "arrow.triangle.turn.up.right.diamond.fill")
                        Text("Find Route")
                    }
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background((startCoordinate != nil && destinationCoordinate != nil) ? Color.brandPrimary : Color.brandPrimary.opacity(0.45))
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
        let waypointCoords = route.shapingWaypoints(maxWaypoints: 8)
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
        if let encoded = route.googleMapsURL(origin: start, destination: dest), let u = URL(string: encoded) {
            UIApplication.shared.open(u)
        }
    }

}

// MARK: - Route Comparison Card
struct RouteComparisonCard: View {
    let saferRoute: Route
    let optimalRoute: Route
    @Binding var selectedRoute: Route?
    let routeSource: String?
    let onExportToAppleMaps: (Route) -> Void
    let onExportToGoogleMaps: (Route) -> Void
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
                            color: .routeSafestBlue,
                            isSelected: selectedRoute?.id == saferRoute.id
                        ) { selectedRoute = saferRoute }

                        RouteOptionCard(
                            route: optimalRoute,
                            title: "Fastest Route",
                            subtitle: routesAreSame ? "Same as safest route" : "Shortest travel time",
                            color: .routeFastestYellow,
                            isSelected: selectedRoute?.id == optimalRoute.id
                        ) { selectedRoute = optimalRoute }

                        if let weather = currentWeather {
                            HStack(spacing: 12) {
                                Image(systemName: "cloud.fill").foregroundColor(.brandTertiary)
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
                            if let timeDifference = comparison.timeDifference {
                                Image(systemName: "clock")
                                Text("Time diff: \(formatTime(timeDifference))")
                                    .foregroundColor((comparison.saferRouteSlower ?? false) ? .brandTertiary : .brandPrimary)
                            }
                            Image(systemName: "shield.checkered")
                            Text(comparison.safetyImprovement).foregroundColor(.brandPrimary)
                        }
                        .font(.subheadline)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)

                        HStack(spacing: 12) {
                            Button(action: { onExportToAppleMaps(selectedRoute ?? saferRoute) }) {
                                VStack(spacing: 6) {
                                    Image("applemapslogo")
                                        .resizable()
                                        .scaledToFit()
                                        .frame(height: 28)
                                    Text("Apple Maps")
                                        .font(.subheadline)
                                        .foregroundColor(.white)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.brandTertiary)
                                .cornerRadius(10)
                            }
                            Button(action: { onExportToGoogleMaps(selectedRoute ?? saferRoute) }) {
                                VStack(spacing: 6) {
                                    Image("googlemapslogo")
                                        .resizable()
                                        .scaledToFit()
                                        .frame(height: 28)
                                    Text("Google Maps")
                                        .font(.subheadline)
                                        .foregroundColor(.white)
                                }
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.brandPrimary)
                                .cornerRadius(10)
                            }
                        }
                        .padding(.horizontal, 16)
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
                        if isSelected { Image(systemName: "checkmark.circle.fill").foregroundColor(.brandSecondary) }
                    }
                    Text(subtitle).font(.caption).foregroundColor(.secondary)
                    HStack(spacing: 16) {
                        if let estimatedTime = route.estimatedTime {
                            Label(formatTime(estimatedTime), systemImage: "clock")
                        }
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
