//
//  NavigationView.swift
//  RiskMapApp
//
//  Navigation view with safer route planning and Google Maps export
//

import SwiftUI
import MapKit
import CoreLocation

struct RouteNavigationView: View {
    @EnvironmentObject var riskService: RiskService
    @StateObject private var routeService: RouteService
    @StateObject private var locationManager = LocationManager()
    @StateObject private var startSearchCompleter = LocationSearchCompleter()
    @StateObject private var destinationSearchCompleter = LocationSearchCompleter()
    
    @State private var startLocation: CLLocationCoordinate2D?
    @State private var destinationLocation: CLLocationCoordinate2D?
    @State private var startAddress: String = ""
    @State private var destinationAddress: String = ""
    @State private var selectedRoute: Route?
    @State private var showRouteComparison = false
    @State private var showStartSuggestions = false
    @State private var showDestinationSuggestions = false
    
    @State private var cameraPosition = MapCameraPosition.region(
        MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
            span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
        )
    )
    
    init(riskService: RiskService) {
        _routeService = StateObject(wrappedValue: RouteService(riskService: riskService))
    }
    
    var body: some View {
        ZStack {
            mapView
            searchAndControlsView
        }
        .onAppear {
            locationManager.requestLocation()
        }
        .onTapGesture {
            // Dismiss suggestions when tapping on map
            showStartSuggestions = false
            showDestinationSuggestions = false
        }
    }
    
    // MARK: - Map View
    @ViewBuilder
    private var mapView: some View {
        Map(position: $cameraPosition, interactionModes: [.pan, .zoom]) {
            mapAnnotations
            mapRoutes
        }
        .mapStyle(.standard)
        .mapControls {
            MapUserLocationButton()
            MapCompass()
        }
    }
    
    // MARK: - Map Annotations
    @MapContentBuilder
    private var mapAnnotations: some MapContent {
        // User location
        if let userLocation = locationManager.location {
            Annotation("My Location", coordinate: userLocation.coordinate) {
                Image(systemName: "location.circle.fill")
                    .foregroundColor(.blue)
                    .font(.title)
                    .background(Circle().fill(Color.white))
            }
        }
        
        // Start location
        if let start = startLocation {
            Annotation("Start", coordinate: start) {
                Image(systemName: "mappin.circle.fill")
                    .foregroundColor(.green)
                    .font(.title)
            }
        }
        
        // Destination location
        if let destination = destinationLocation {
            Annotation("Destination", coordinate: destination) {
                Image(systemName: "flag.circle.fill")
                    .foregroundColor(.red)
                    .font(.title)
            }
        }
    }
    
    // MARK: - Map Routes
    @MapContentBuilder
    private var mapRoutes: some MapContent {
        // Simple direct route line (for testing)
        if let start = startLocation, let destination = destinationLocation,
           routeService.saferRoute == nil && routeService.optimalRoute == nil {
            MapPolyline(coordinates: [start, destination])
                .stroke(.gray.opacity(0.5), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round, dash: [5, 5]))
        }
        
        // Safer route
        if let saferRoute = routeService.saferRoute {
            let saferCoords = saferRoute.polyline.coordinates
            if !saferCoords.isEmpty {
                MapPolyline(coordinates: saferCoords)
                    .stroke(.blue, style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round))
            }
        }
        
        // Optimal route
        if let optimalRoute = routeService.optimalRoute {
            let optimalCoords = optimalRoute.polyline.coordinates
            if !optimalCoords.isEmpty {
                MapPolyline(coordinates: optimalCoords)
                    .stroke(.orange, style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round))
            }
        }
        
        // Selected route (highlighted) - show selected route on top
        if let selected = selectedRoute {
            let selectedCoords = selected.polyline.coordinates
            if !selectedCoords.isEmpty {
                MapPolyline(coordinates: selectedCoords)
                    .stroke(.purple, style: StrokeStyle(lineWidth: 10, lineCap: .round, lineJoin: .round))
            }
        }
    }
    
    // MARK: - Search and Controls View
    @ViewBuilder
    private var searchAndControlsView: some View {
        VStack {
            searchBarView
            Spacer()
            routeComparisonCardView
        }
    }
    
    // MARK: - Search Bar View
    @ViewBuilder
    private var searchBarView: some View {
        VStack(spacing: 0) {
            startLocationSearchView
            destinationLocationSearchView
            calculateRoutesButton
            errorMessageView
        }
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 5)
        .padding()
    }
    
    // MARK: - Start Location Search
    @ViewBuilder
    private var startLocationSearchView: some View {
        VStack(spacing: 0) {
            HStack {
                Button(action: {
                    useCurrentLocationAsStart()
                }) {
                    Image(systemName: "location.fill")
                        .foregroundColor(.blue)
                }
                .padding(.leading)
                
                TextField("Start location", text: $startAddress)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: startAddress) { oldValue, newValue in
                        startSearchCompleter.searchQuery = newValue
                        showStartSuggestions = !newValue.isEmpty && startLocation == nil
                    }
                    .onTapGesture {
                        showStartSuggestions = !startAddress.isEmpty && startLocation == nil
                    }
                
                if startLocation != nil {
                    Button(action: {
                        startLocation = nil
                        startAddress = ""
                        showStartSuggestions = false
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding(.horizontal)
            .padding(.top, 8)
            
            if showStartSuggestions && !startSearchCompleter.completions.isEmpty {
                startSuggestionsList
            }
        }
    }
    
    // MARK: - Start Suggestions List
    @ViewBuilder
    private var startSuggestionsList: some View {
        SuggestionsListView(
            completions: startSearchCompleter.completions,
            iconName: "mappin.circle.fill",
            iconColor: .blue,
            onSelect: { completion in
                selectCompletion(completion, isStart: true)
            }
        )
    }
    
    // MARK: - Destination Location Search
    @ViewBuilder
    private var destinationLocationSearchView: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "mappin")
                    .foregroundColor(.gray)
                    .padding(.leading)
                
                TextField("Destination", text: $destinationAddress)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: destinationAddress) { oldValue, newValue in
                        destinationSearchCompleter.searchQuery = newValue
                        showDestinationSuggestions = !newValue.isEmpty && destinationLocation == nil
                    }
                    .onTapGesture {
                        showDestinationSuggestions = !destinationAddress.isEmpty && destinationLocation == nil
                    }
                
                if destinationLocation != nil {
                    Button(action: {
                        destinationLocation = nil
                        destinationAddress = ""
                        showDestinationSuggestions = false
                    }) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.gray)
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
            
            if showDestinationSuggestions && !destinationSearchCompleter.completions.isEmpty {
                destinationSuggestionsList
            }
        }
    }
    
    // MARK: - Destination Suggestions List
    @ViewBuilder
    private var destinationSuggestionsList: some View {
        SuggestionsListView(
            completions: destinationSearchCompleter.completions,
            iconName: "flag.circle.fill",
            iconColor: .red,
            onSelect: { completion in
                selectCompletion(completion, isStart: false)
            }
        )
    }
    
    // MARK: - Calculate Routes Button
    @ViewBuilder
    private var calculateRoutesButton: some View {
        Button(action: {
            if startLocation != nil && destinationLocation != nil {
                calculateRoutes()
            } else {
                routeService.errorMessage = "Please enter both start and destination locations"
            }
        }) {
            HStack {
                if routeService.isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    Text("Calculating...")
                } else {
                    Image(systemName: "arrow.triangle.2.circlepath")
                    Text("Find Safest Route")
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background((startLocation != nil && destinationLocation != nil) ? Color.blue : Color.gray)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .padding(.horizontal)
        .padding(.bottom, 8)
        .disabled(routeService.isLoading || startLocation == nil || destinationLocation == nil)
    }
    
    // MARK: - Error Message View
    @ViewBuilder
    private var errorMessageView: some View {
        if let error = routeService.errorMessage {
            Text(error)
                .font(.caption)
                .foregroundColor(.red)
                .padding(.horizontal)
                .padding(.bottom, 4)
        }
    }
    
    // MARK: - Route Comparison Card View
    @ViewBuilder
    private var routeComparisonCardView: some View {
        if showRouteComparison, let safer = routeService.saferRoute, let optimal = routeService.optimalRoute {
            RouteComparisonCard(
                saferRoute: safer,
                optimalRoute: optimal,
                selectedRoute: $selectedRoute,
                onExportToGoogleMaps: {
                    exportToGoogleMaps(route: selectedRoute ?? safer)
                }
            )
            .transition(.move(edge: .bottom))
        }
    }
    
    // MARK: - Helper Functions
    private func useCurrentLocationAsStart() {
        if let location = locationManager.location {
            startLocation = location.coordinate
            startAddress = "Current Location"
            showStartSuggestions = false
            updateCamera()
        }
    }
    
    private func selectCompletion(_ completion: MKLocalSearchCompletion, isStart: Bool) {
        // Use MKLocalSearch to get the actual location from the completion
        let searchRequest = MKLocalSearch.Request(completion: completion)
        searchRequest.region = MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832), // Toronto
            span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5)
        )
        
        let search = MKLocalSearch(request: searchRequest)
        
        // Capture values needed in closure
        let isStartValue = isStart
        let completionTitle = completion.title
        let completionSubtitle = completion.subtitle
        
        search.start { response, error in
            if let error = error {
                print("Search error: \(error.localizedDescription)")
                // Fallback: try to use the completion title directly
                Task { @MainActor in
                    if isStartValue {
                        self.startAddress = completionTitle
                        self.showStartSuggestions = false
                    } else {
                        self.destinationAddress = completionTitle
                        self.showDestinationSuggestions = false
                    }
                    // Try geocoding as fallback
                    self.geocodeAddress(completionTitle, isStart: isStartValue)
                }
                return
            }
            
            guard let response = response,
                  let item = response.mapItems.first else {
                print("No results from search")
                Task { @MainActor in
                    if isStartValue {
                        self.startAddress = completionTitle
                        self.showStartSuggestions = false
                    } else {
                        self.destinationAddress = completionTitle
                        self.showDestinationSuggestions = false
                    }
                }
                return
            }
            
            let coordinate = item.placemark.coordinate
            let name = item.name ?? completionTitle
            let address = completionSubtitle.isEmpty ? name : "\(name), \(completionSubtitle)"
            
            Task { @MainActor in
                if isStartValue {
                    self.startLocation = coordinate
                    self.startAddress = address
                    self.showStartSuggestions = false
                } else {
                    self.destinationLocation = coordinate
                    self.destinationAddress = address
                    self.showDestinationSuggestions = false
                }
                self.updateCamera()
            }
        }
    }
    
    private func geocodeAddress(_ address: String, isStart: Bool) {
        // Use MKLocalSearch instead of CLGeocoder for better results
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = address
        request.region = MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832), // Toronto
            span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5)
        )
        
        let search = MKLocalSearch(request: request)
        let isStartValue = isStart
        let addressValue = address
        
        search.start { response, error in
            if let error = error {
                print("Local search error: \(error.localizedDescription)")
                return
            }
            
            guard let response = response,
                  let item = response.mapItems.first else {
                print("No results found for: \(addressValue)")
                return
            }
            
            let coordinate = item.placemark.coordinate
            let name = item.name ?? addressValue
            
            Task { @MainActor in
                if isStartValue {
                    self.startLocation = coordinate
                    self.startAddress = name
                    self.showStartSuggestions = false
                } else {
                    self.destinationLocation = coordinate
                    self.destinationAddress = name
                    self.showDestinationSuggestions = false
                }
                self.updateCamera()
            }
        }
    }
    
    private func calculateRoutes() {
        guard let start = startLocation, let destination = destinationLocation else {
            print("Error: Missing start or destination")
            return
        }
        
        print("📍 Calculating routes from \(start.latitude), \(start.longitude) to \(destination.latitude), \(destination.longitude)")
        
        Task {
            // Calculate routes with safety analysis
            await routeService.calculateRoutes(from: start, to: destination)
            
            await MainActor.run {
                print("📊 Routes calculated. Safer: \(routeService.saferRoute != nil), Optimal: \(routeService.optimalRoute != nil)")
                
                if let safer = routeService.saferRoute {
                    let coords = safer.polyline.coordinates
                    print("✅ Safer route: \(coords.count) coordinates, \(String(format: "%.1f", safer.distance/1000))km, \(Int(safer.estimatedTime/60))min")
                }
                if let optimal = routeService.optimalRoute {
                    let coords = optimal.polyline.coordinates
                    print("✅ Optimal route: \(coords.count) coordinates, \(String(format: "%.1f", optimal.distance/1000))km, \(Int(optimal.estimatedTime/60))min")
                }
                
                if routeService.saferRoute != nil && routeService.optimalRoute != nil {
                    showRouteComparison = true
                    selectedRoute = routeService.saferRoute
                    updateCameraForRoutes()
                    print("✅ Routes displayed on map, camera updated")
                } else {
                    print("⚠️ Warning: Routes not available")
                    if let error = routeService.errorMessage {
                        print("❌ Error: \(error)")
                    }
                }
            }
        }
    }
    
    private func updateCamera() {
        if let start = startLocation, let destination = destinationLocation {
            let center = CLLocationCoordinate2D(
                latitude: (start.latitude + destination.latitude) / 2,
                longitude: (start.longitude + destination.longitude) / 2
            )
            
            let latDelta = abs(start.latitude - destination.latitude) * 1.5
            let lonDelta = abs(start.longitude - destination.longitude) * 1.5
            
            cameraPosition = .region(MKCoordinateRegion(
                center: center,
                span: MKCoordinateSpan(
                    latitudeDelta: max(latDelta, 0.01),
                    longitudeDelta: max(lonDelta, 0.01)
                )
            ))
        } else if let location = startLocation ?? destinationLocation {
            cameraPosition = .region(MKCoordinateRegion(
                center: location,
                span: MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
            ))
        }
    }
    
    private func updateCameraForRoutes() {
        guard let safer = routeService.saferRoute, let optimal = routeService.optimalRoute else {
            return
        }
        
        let saferCoords = safer.polyline.coordinates
        let optimalCoords = optimal.polyline.coordinates
        let allCoords = saferCoords + optimalCoords
        guard !allCoords.isEmpty else { return }
        
        let latitudes = allCoords.map { $0.latitude }
        let longitudes = allCoords.map { $0.longitude }
        
        guard let minLat = latitudes.min(),
              let maxLat = latitudes.max(),
              let minLon = longitudes.min(),
              let maxLon = longitudes.max() else {
            return
        }
        
        let center = CLLocationCoordinate2D(
            latitude: (minLat + maxLat) / 2.0,
            longitude: (minLon + maxLon) / 2.0
        )
        
        let span = MKCoordinateSpan(
            latitudeDelta: (maxLat - minLat) * 1.3,
            longitudeDelta: (maxLon - minLon) * 1.3
        )
        
        cameraPosition = .region(MKCoordinateRegion(center: center, span: span))
    }
    
    // MARK: - Google Maps Export
    private func exportToGoogleMaps(route: Route) {
        guard let start = startLocation, let destination = destinationLocation else {
            return
        }
        
        // Create Google Maps URL with directions
        // Format: https://www.google.com/maps/dir/?api=1&origin=lat,lng&destination=lat,lng&travelmode=driving
        let googleMapsURL = "https://www.google.com/maps/dir/?api=1&origin=\(start.latitude),\(start.longitude)&destination=\(destination.latitude),\(destination.longitude)&travelmode=driving"
        
        if let url = URL(string: googleMapsURL) {
            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url)
            } else {
                // Fallback to web version
                if let webURL = URL(string: googleMapsURL) {
                    UIApplication.shared.open(webURL)
                }
            }
        }
    }
}


// MARK: - Route Comparison Card
struct RouteComparisonCard: View {
    let saferRoute: Route
    let optimalRoute: Route
    @Binding var selectedRoute: Route?
    let onExportToGoogleMaps: () -> Void
    
    private var comparison: RouteComparison {
        RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Header
            HStack {
                Text("Route Options")
                    .font(.headline)
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top)
            
            // Safer Route Card
            RouteOptionCard(
                route: saferRoute,
                title: "Safest Route",
                subtitle: saferRoute.safetyExplanation(comparedTo: optimalRoute),
                color: .blue,
                isSelected: selectedRoute?.id == saferRoute.id
            ) {
                selectedRoute = saferRoute
            }
            
            // Optimal Route Card
            RouteOptionCard(
                route: optimalRoute,
                title: "Fastest Route",
                subtitle: "Shortest travel time",
                color: .orange,
                isSelected: selectedRoute?.id == optimalRoute.id
            ) {
                selectedRoute = optimalRoute
            }
            
            // Detailed Explanation
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(.blue)
                    Text("Why This Route?")
                        .font(.headline)
                }
                
                Text(comparison.detailedExplanation)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(10)
            .padding(.horizontal)
            
            // Comparison Info
            VStack(spacing: 8) {
                HStack {
                    Image(systemName: "clock")
                    Text("Time difference: \(formatTimeDifference(comparison.timeDifference))")
                        .font(.subheadline)
                        .foregroundColor(comparison.saferRouteSlower ? .orange : .green)
                }
                
                HStack {
                    Image(systemName: "shield.checkered")
                    Text("Safety improvement: \(comparison.safetyImprovement)")
                        .font(.subheadline)
                        .foregroundColor(.blue)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(Color(.systemGray6))
            .cornerRadius(10)
            .padding(.horizontal)
            
            // Export to Google Maps button
            if selectedRoute != nil {
                Button(action: onExportToGoogleMaps) {
                    HStack {
                        Image(systemName: "map.fill")
                        Text("Open in Google Maps")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .background(Color(.systemBackground))
        .cornerRadius(16, corners: [.topLeft, .topRight])
        .shadow(radius: 10)
    }
    
    private func formatTimeDifference(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds / 60)
        if minutes < 60 {
            return "\(minutes) min"
        } else {
            let hours = minutes / 60
            let mins = minutes % 60
            return "\(hours)h \(mins)m"
        }
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
    
    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    // Color indicator
                    RoundedRectangle(cornerRadius: 4)
                        .fill(color)
                        .frame(width: 4, height: 50)
                    
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(title)
                                .font(.headline)
                            if isSelected {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                            }
                        }
                        
                        // Safety explanation
                        Text(subtitle)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.leading)
                        
                        HStack(spacing: 16) {
                            Label(formatTime(route.estimatedTime), systemImage: "clock")
                            Label(formatDistance(route.distance), systemImage: "ruler")
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                        
                        // Risk info
                        HStack(spacing: 12) {
                            if route.highRiskSegments > 0 {
                                Label("\(route.highRiskSegments) high risk", systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption2)
                                    .foregroundColor(.red)
                            }
                            if route.mediumRiskSegments > 0 {
                                Label("\(route.mediumRiskSegments) medium", systemImage: "exclamationmark.circle.fill")
                                    .font(.caption2)
                                    .foregroundColor(.orange)
                            }
                            if route.lowRiskSegments > 0 {
                                Label("\(route.lowRiskSegments) low risk", systemImage: "checkmark.circle.fill")
                                    .font(.caption2)
                                    .foregroundColor(.green)
                            }
                        }
                    }
                    
                    Spacer()
                }
                .padding()
                .background(isSelected ? color.opacity(0.1) : Color(.systemGray6))
                .cornerRadius(10)
            }
        }
        .buttonStyle(.plain)
        .padding(.horizontal)
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        let minutes = Int(seconds / 60)
        if minutes < 60 {
            return "\(minutes) min"
        } else {
            let hours = minutes / 60
            let mins = minutes % 60
            return "\(hours)h \(mins)m"
        }
    }
    
    private func formatDistance(_ meters: CLLocationDistance) -> String {
        if meters < 1000 {
            return String(format: "%.0f m", meters)
        } else {
            return String(format: "%.1f km", meters / 1000)
        }
    }
}

// MARK: - Location Manager
class LocationManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    @Published var location: CLLocation?
    
    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
    }
    
    func requestLocation() {
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        location = locations.last
        manager.stopUpdatingLocation()
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error.localizedDescription)")
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
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}

// MARK: - Completion Item (Identifiable wrapper)
struct CompletionItem: Identifiable {
    let id: String
    let completion: MKLocalSearchCompletion
    let index: Int
    
    init(completion: MKLocalSearchCompletion, index: Int) {
        self.completion = completion
        self.index = index
        // Use a combination of title and subtitle as ID since identifier might not be stable
        self.id = "\(completion.title)-\(completion.subtitle)-\(index)"
    }
}

// MARK: - Suggestions List View
struct SuggestionsListView: View {
    let completions: [MKLocalSearchCompletion]
    let iconName: String
    let iconColor: Color
    let onSelect: (MKLocalSearchCompletion) -> Void
    
    private var completionItems: [CompletionItem] {
        completions.enumerated().map { index, completion in
            CompletionItem(completion: completion, index: index)
        }
    }
    
    var body: some View {
        ScrollView {
            suggestionsContent
        }
        .frame(maxHeight: 200)
        .background(Color(.systemBackground))
        .cornerRadius(8)
        .shadow(radius: 5)
        .padding(.horizontal)
    }
    
    @ViewBuilder
    private var suggestionsContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(completionItems) { item in
                suggestionRow(item: item)
            }
        }
    }
    
    @ViewBuilder
    private func suggestionRow(item: CompletionItem) -> some View {
        Button(action: {
            onSelect(item.completion)
        }) {
            suggestionRowContent(completion: item.completion)
        }
        .buttonStyle(.plain)
        
        if item.index < completions.count - 1 {
            Divider()
                .padding(.leading, 48)
        }
    }
    
    @ViewBuilder
    private func suggestionRowContent(completion: MKLocalSearchCompletion) -> some View {
        HStack {
            Image(systemName: iconName)
                .foregroundColor(iconColor)
            VStack(alignment: .leading, spacing: 4) {
                Text(completion.title)
                    .foregroundColor(.primary)
                    .font(.body)
                if !completion.subtitle.isEmpty {
                    Text(completion.subtitle)
                        .foregroundColor(.secondary)
                        .font(.caption)
                }
            }
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

// MARK: - Location Search Completer
class LocationSearchCompleter: NSObject, ObservableObject, MKLocalSearchCompleterDelegate {
    @Published var completions: [MKLocalSearchCompletion] = []
    @Published var searchQuery: String = "" {
        didSet {
            completer.queryFragment = searchQuery
        }
    }
    
    private let completer = MKLocalSearchCompleter()
    
    override init() {
        super.init()
        completer.delegate = self
        completer.resultTypes = [.address, .pointOfInterest]
        completer.region = MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832), // Toronto
            span: MKCoordinateSpan(latitudeDelta: 0.5, longitudeDelta: 0.5)
        )
    }
    
    func completerDidUpdateResults(_ completer: MKLocalSearchCompleter) {
        DispatchQueue.main.async {
            self.completions = Array(completer.results.prefix(5)) // Limit to 5 results
        }
    }
    
    func completer(_ completer: MKLocalSearchCompleter, didFailWithError error: Error) {
        print("Search completer error: \(error.localizedDescription)")
    }
}
