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
    @StateObject private var weatherService = WeatherService()
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
    @State private var currentWeather: WeatherData?
    
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
            routeLegend
            weatherInfoBadge
        }
        .onAppear {
            locationManager.requestLocation()
            loadWeatherInfo()
        }
        .onTapGesture {
            // Dismiss suggestions when tapping on map
            showStartSuggestions = false
            showDestinationSuggestions = false
        }
    }
    
    // MARK: - weather info badge
    @ViewBuilder
    private var weatherInfoBadge: some View {
        if let weather = currentWeather {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    weatherIcon(for: weather.condition)
                    Text(weather.condition.displayName)
                        .font(.caption)
                        .fontWeight(.medium)
                }
                
                HStack(spacing: 4) {
                    Image(systemName: "clock.fill")
                        .font(.system(size: 10))
                    Text(timeOfDayLabel())
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
                
                // shows if using API or fallback
                if weatherService.errorMessage != nil {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 8))
                            .foregroundColor(.orange)
                        Text("Fallback")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    }
                } else {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 8))
                            .foregroundColor(.green)
                        Text("Live")
                            .font(.caption2)
                            .foregroundColor(.green)
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color(.systemBackground).opacity(0.95))
            .cornerRadius(10)
            .shadow(radius: 5)
            .padding(.top, 100)
            .padding(.leading, 16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }
    
    @ViewBuilder
    private func weatherIcon(for condition: WeatherData.WeatherCondition) -> some View {
        switch condition {
        case .clear:
            Image(systemName: "sun.max.fill")
        case .cloudy:
            Image(systemName: "cloud.fill")
        case .rain, .heavyRain:
            Image(systemName: "cloud.rain.fill")
        case .snow, .heavySnow:
            Image(systemName: "cloud.snow.fill")
        case .fog, .mist:
            Image(systemName: "cloud.fog.fill")
        case .thunderstorm:
            Image(systemName: "cloud.bolt.fill")
        case .sleet:
            Image(systemName: "cloud.sleet.fill")
        }
    }
    
    private func timeOfDayLabel() -> String {
        let calendar = Calendar.current
        let now = Date()
        let hour = calendar.component(.hour, from: now)
        let weekday = calendar.component(.weekday, from: now)
        let isWeekend = weekday == 1 || weekday == 7
        
        // late night: 11 PM - 4:59 AM
        if hour >= 23 || hour < 5 {
            return "Late Night"
        }
        // night: 10 PM - 10:59 PM
        else if hour >= 22 {
            return "Night"
        }
        // early morning: 5 AM - 6:59 AM
        else if hour < 7 {
            return "Early Morning"
        }
        // morning rush: 7 AM - 9:59 AM
        else if hour >= 7 && hour < 10 {
            return isWeekend ? "Morning" : "Rush Hour"
        }
        // daytime: 10 AM - 3:59 PM
        else if hour >= 10 && hour < 16 {
            return "Daytime"
        }
        // evening rush: 4 PM - 6:59 PM
        else if hour >= 16 && hour < 19 {
            return isWeekend ? "Evening" : "Rush Hour"
        }
        // evening: 7 PM - 9:59 PM
        else {
            return "Evening"
        }
    }
    
    private func loadWeatherInfo() {
        Task {
            // load weather for toronto center (or use route center if available)
            let location = startLocation ?? CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
            let weather = await weatherService.getWeatherData(for: location)
            await MainActor.run {
                currentWeather = weather
            }
        }
    }
    
    // MARK: - Route Legend
    @ViewBuilder
    private var routeLegend: some View {
        if routeService.saferRoute != nil || routeService.optimalRoute != nil {
            VStack(alignment: .leading, spacing: 8) {
                if routeService.optimalRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle()
                            .fill(.orange)
                            .frame(width: 20, height: 4)
                        Text("Fastest Route")
                            .font(.caption)
                            .foregroundColor(.primary)
                    }
                }
                if routeService.saferRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle()
                            .fill(.blue)
                            .frame(width: 20, height: 4)
                        Text("Safest Route")
                            .font(.caption)
                            .foregroundColor(.primary)
                    }
                }
                if selectedRoute != nil {
                    HStack(spacing: 8) {
                        Rectangle()
                            .fill(.purple)
                            .frame(width: 20, height: 4)
                        Text("Selected")
                            .font(.caption)
                            .foregroundColor(.primary)
                    }
                }
            }
            .padding(12)
            .background(Color(.systemBackground).opacity(0.9))
            .cornerRadius(10)
            .shadow(radius: 5)
            .padding(.top, 100)
            .padding(.trailing, 16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
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
        
        // Optimal route (fastest) - show with orange color and white outline for visibility
        if let optimalRoute = routeService.optimalRoute {
            // Use detailed coordinates from route steps for better accuracy
            let optimalCoords = optimalRoute.detailedCoordinates.filter { coord in
                coord.latitude.isFinite && coord.longitude.isFinite
            }
            if !optimalCoords.isEmpty {
                // White outline for visibility
                MapPolyline(coordinates: optimalCoords)
                    .stroke(.white, style: StrokeStyle(lineWidth: 12, lineCap: .round, lineJoin: .round))
                // Orange route on top
                MapPolyline(coordinates: optimalCoords)
                    .stroke(.orange, style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round))
            }
        }
        
        // Safer route - show with blue color, make it more prominent if different
        if let saferRoute = routeService.saferRoute {
            // Use detailed coordinates from route steps for better accuracy
            let saferCoords = saferRoute.detailedCoordinates.filter { coord in
                coord.latitude.isFinite && coord.longitude.isFinite
            }
            if !saferCoords.isEmpty {
                // Check if safer route is different from optimal route
                let optimalCoords = routeService.optimalRoute?.detailedCoordinates ?? []
                let isDifferent = optimalCoords.count != saferCoords.count ||
                    !areRoutesSimilar(optimalCoords, saferCoords)
                
                if isDifferent {
                    // Different route - make it very visible with white outline
                    MapPolyline(coordinates: saferCoords)
                        .stroke(.white, style: StrokeStyle(lineWidth: 14, lineCap: .round, lineJoin: .round))
                    MapPolyline(coordinates: saferCoords)
                        .stroke(.blue, style: StrokeStyle(lineWidth: 10, lineCap: .round, lineJoin: .round))
                } else {
                    // Same route - still show it but with dashed line to indicate it's the only option
                    MapPolyline(coordinates: saferCoords)
                        .stroke(.white, style: StrokeStyle(lineWidth: 12, lineCap: .round, lineJoin: .round))
                    MapPolyline(coordinates: saferCoords)
                        .stroke(.blue.opacity(0.8), style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round, dash: [15, 8]))
                }
            }
        }
        
        // Selected route (highlighted) - show selected route on top with purple
        if let selected = selectedRoute {
            // Use detailed coordinates from route steps for better accuracy
            let selectedCoords = selected.detailedCoordinates.filter { coord in
                coord.latitude.isFinite && coord.longitude.isFinite
            }
            if !selectedCoords.isEmpty {
                // White outline
                MapPolyline(coordinates: selectedCoords)
                    .stroke(.white, style: StrokeStyle(lineWidth: 16, lineCap: .round, lineJoin: .round))
                // Purple highlight
                MapPolyline(coordinates: selectedCoords)
                    .stroke(.purple, style: StrokeStyle(lineWidth: 12, lineCap: .round, lineJoin: .round))
            }
        }
    }
    
    // Helper to check if two routes are similar (within tolerance)
    private func areRoutesSimilar(_ coords1: [CLLocationCoordinate2D], _ coords2: [CLLocationCoordinate2D]) -> Bool {
        guard coords1.count > 0 && coords2.count > 0 else { return false }
        
        // If coordinate counts are very different, they're different routes
        if abs(coords1.count - coords2.count) > coords1.count / 10 {
            return false
        }
        
        // Check if start and end points are similar
        let start1 = coords1[0]
        let end1 = coords1[coords1.count - 1]
        let start2 = coords2[0]
        let end2 = coords2[coords2.count - 1]
        
        let startDistance = CLLocation(latitude: start1.latitude, longitude: start1.longitude)
            .distance(from: CLLocation(latitude: start2.latitude, longitude: start2.longitude))
        let endDistance = CLLocation(latitude: end1.latitude, longitude: end1.longitude)
            .distance(from: CLLocation(latitude: end2.latitude, longitude: end2.longitude))
        
        // If start/end are very close and route lengths are similar, consider them the same
        return startDistance < 50 && endDistance < 50
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
                        // Debounce search to prevent timeout
                        Task {
                            try? await Task.sleep(nanoseconds: 300_000_000) // 300ms delay
                            if newValue == startAddress { // Only update if value hasn't changed
                                await MainActor.run {
                                    startSearchCompleter.searchQuery = newValue
                                    showStartSuggestions = !newValue.isEmpty && startLocation == nil
                                }
                            }
                        }
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
                        // Debounce search to prevent timeout
                        Task {
                            try? await Task.sleep(nanoseconds: 300_000_000) // 300ms delay
                            if newValue == destinationAddress { // Only update if value hasn't changed
                                await MainActor.run {
                                    destinationSearchCompleter.searchQuery = newValue
                                    showDestinationSuggestions = !newValue.isEmpty && destinationLocation == nil
                                }
                            }
                        }
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
                },
                currentWeather: currentWeather
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
        
        // Update weather info for route area
        Task {
            let routeCenter = CLLocationCoordinate2D(
                latitude: (start.latitude + destination.latitude) / 2,
                longitude: (start.longitude + destination.longitude) / 2
            )
            let weather = await weatherService.getWeatherData(for: routeCenter)
            await MainActor.run {
                currentWeather = weather
            }
        }
        
        Task {
            // Calculate routes with safety analysis
            await routeService.calculateRoutes(from: start, to: destination)
            
            await MainActor.run {
                print("📊 Routes calculated. Safer: \(routeService.saferRoute != nil), Optimal: \(routeService.optimalRoute != nil)")
                
                if let safer = routeService.saferRoute {
                    let coords = safer.detailedCoordinates
                    print("✅ Safer route: \(coords.count) detailed coordinates, \(String(format: "%.1f", safer.distance/1000))km, \(Int(safer.estimatedTime/60))min")
                }
                if let optimal = routeService.optimalRoute {
                    let coords = optimal.detailedCoordinates
                    print("✅ Optimal route: \(coords.count) detailed coordinates, \(String(format: "%.1f", optimal.distance/1000))km, \(Int(optimal.estimatedTime/60))min")
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
            // Validate coordinates are finite
            guard start.latitude.isFinite && start.longitude.isFinite &&
                  destination.latitude.isFinite && destination.longitude.isFinite else {
                return
            }
            
            let centerLat = (start.latitude + destination.latitude) / 2
            let centerLon = (start.longitude + destination.longitude) / 2
            
            guard centerLat.isFinite && centerLon.isFinite else {
                return
            }
            
            let center = CLLocationCoordinate2D(
                latitude: centerLat,
                longitude: centerLon
            )
            
            let latDelta = abs(start.latitude - destination.latitude) * 1.5
            let lonDelta = abs(start.longitude - destination.longitude) * 1.5
            
            guard latDelta.isFinite && lonDelta.isFinite else {
                return
            }
            
            cameraPosition = .region(MKCoordinateRegion(
                center: center,
                span: MKCoordinateSpan(
                    latitudeDelta: max(latDelta, 0.01),
                    longitudeDelta: max(lonDelta, 0.01)
                )
            ))
        } else if let location = startLocation ?? destinationLocation {
            // Validate location coordinates
            guard location.latitude.isFinite && location.longitude.isFinite else {
                return
            }
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
        
        // Use detailed coordinates for better camera positioning
        let saferCoords = safer.detailedCoordinates
        let optimalCoords = optimal.detailedCoordinates
        let allCoords = saferCoords + optimalCoords
        
        // Filter out invalid coordinates (NaN or infinite)
        let validCoords = allCoords.filter { coord in
            coord.latitude.isFinite && coord.longitude.isFinite &&
            coord.latitude >= -90 && coord.latitude <= 90 &&
            coord.longitude >= -180 && coord.longitude <= 180
        }
        
        guard !validCoords.isEmpty else { return }
        
        let latitudes = validCoords.map { $0.latitude }
        let longitudes = validCoords.map { $0.longitude }
        
        guard let minLat = latitudes.min(),
              let maxLat = latitudes.max(),
              let minLon = longitudes.min(),
              let maxLon = longitudes.max(),
              minLat.isFinite && maxLat.isFinite && minLon.isFinite && maxLon.isFinite else {
            return
        }
        
        let centerLat = (minLat + maxLat) / 2.0
        let centerLon = (minLon + maxLon) / 2.0
        
        guard centerLat.isFinite && centerLon.isFinite else {
            return
        }
        
        let center = CLLocationCoordinate2D(
            latitude: centerLat,
            longitude: centerLon
        )
        
        let latDelta = max((maxLat - minLat) * 1.3, 0.01)
        let lonDelta = max((maxLon - minLon) * 1.3, 0.01)
        
        guard latDelta.isFinite && lonDelta.isFinite else {
            return
        }
        
        let span = MKCoordinateSpan(
            latitudeDelta: latDelta,
            longitudeDelta: lonDelta
        )
        
        cameraPosition = .region(MKCoordinateRegion(center: center, span: span))
    }
    
    // MARK: - Google Maps Export
    private func exportToGoogleMaps(route: Route) {
        guard let start = startLocation, let destination = destinationLocation else {
            return
        }
        
        // Extract waypoints from route steps to preserve the selected route
        // Google Maps URL supports waypoints to guide the route (max ~23 waypoints)
        let steps = route.steps
        var waypoints: [String] = []
        
        // Use more waypoints (up to 23) to better preserve the route shape
        // Sample evenly along the route, prioritizing steps with turn instructions
        let maxWaypoints = 23
        let stepInterval = max(1, steps.count / maxWaypoints)
        
        // first pass: collect steps with instructions (turns, merges, etc.)
        var instructionSteps: [Int] = []
        for (index, step) in steps.enumerated() {
            if index > 0 && index < steps.count - 1 && !step.instructions.isEmpty {
                instructionSteps.append(index)
            }
        }
        
        // second pass: sample evenly along the route
        for (index, step) in steps.enumerated() {
            // skip first and last steps (they're the origin/destination)
            if index > 0 && index < steps.count - 1 {
                let shouldInclude = index % stepInterval == 0 || instructionSteps.contains(index)
                
                if shouldInclude {
                    // get the first coordinate from the step's polyline (start of the step)
                    let stepCoords = step.polyline.coordinates
                    if let firstCoord = stepCoords.first {
                        let waypointString = "\(firstCoord.latitude),\(firstCoord.longitude)"
                        // avoid duplicates
                        if !waypoints.contains(waypointString) {
                            waypoints.append(waypointString)
                        }
                    }
                }
            }
        }
        
        // limit to max waypoints to avoid URL length issues
        let limitedWaypoints = Array(waypoints.prefix(maxWaypoints))
        
        // build Google Maps URL with waypoints
        var googleMapsURL = "https://www.google.com/maps/dir/?api=1&origin=\(start.latitude),\(start.longitude)&destination=\(destination.latitude),\(destination.longitude)&travelmode=driving"
        
        // add waypoints if we have any
        if !limitedWaypoints.isEmpty {
            let waypointsString = limitedWaypoints.joined(separator: "|")
            googleMapsURL += "&waypoints=\(waypointsString)"
        }
        
        // URL encode the waypoints string to handle special characters
        if let encodedURL = googleMapsURL.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
           let url = URL(string: encodedURL) {
            if UIApplication.shared.canOpenURL(url) {
                UIApplication.shared.open(url)
            } else {
                // fallback to web version
                if let webURL = URL(string: encodedURL) {
                    UIApplication.shared.open(webURL)
                }
            }
        } else {
            // fallback: try without waypoints if encoding fails
            let fallbackURL = "https://www.google.com/maps/dir/?api=1&origin=\(start.latitude),\(start.longitude)&destination=\(destination.latitude),\(destination.longitude)&travelmode=driving"
            if let url = URL(string: fallbackURL) {
                UIApplication.shared.open(url)
            }
        }
    }
}


// MARK: - route comparison card
struct RouteComparisonCard: View {
    let saferRoute: Route
    let optimalRoute: Route
    @Binding var selectedRoute: Route?
    let onExportToGoogleMaps: () -> Void
    let currentWeather: WeatherData?
    @State private var isExpanded = true
    
    private var comparison: RouteComparison {
        RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)
    }
    
    private var routesAreSame: Bool {
        // Check if routes are essentially the same
        let saferCoords = saferRoute.polyline.coordinates
        let optimalCoords = optimalRoute.polyline.coordinates
        
        // If coordinate counts are very similar and distances are very close, they're likely the same
        let coordDiff = abs(saferCoords.count - optimalCoords.count)
        let distanceDiff = abs(saferRoute.distance - optimalRoute.distance)
        
        return coordDiff < max(saferCoords.count, optimalCoords.count) / 10 &&
               distanceDiff < 100 // Less than 100m difference
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Collapsible Header
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Text("Route Options")
                        .font(.headline)
                    Spacer()
                    
                    // Quick summary when collapsed
                    if !isExpanded {
                        HStack(spacing: 12) {
                            if selectedRoute?.id == saferRoute.id {
                                Label(formatTime(saferRoute.estimatedTime), systemImage: "shield.checkered")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                            } else if selectedRoute?.id == optimalRoute.id {
                                Label(formatTime(optimalRoute.estimatedTime), systemImage: "clock")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                        }
                    }
                    
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal)
                .padding(.vertical, 12)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            
            // Expandable content
            if isExpanded {
                VStack(spacing: 16) {
                    // Show warning if routes are the same
                    if routesAreSame {
                        HStack {
                            Image(systemName: "info.circle.fill")
                                .foregroundColor(.orange)
                            Text("Only one route available - safest and fastest routes are the same")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(8)
                        .padding(.horizontal)
                    }
                    
                    // Safer Route Card
                    RouteOptionCard(
                        route: saferRoute,
                        title: "Safest Route",
                        subtitle: routesAreSame ? "Same as fastest route (only one option available)" : saferRoute.safetyExplanation(comparedTo: optimalRoute),
                        color: .blue,
                        isSelected: selectedRoute?.id == saferRoute.id
                    ) {
                        selectedRoute = saferRoute
                    }
                    
                    // Optimal Route Card
                    RouteOptionCard(
                        route: optimalRoute,
                        title: "Fastest Route",
                        subtitle: routesAreSame ? "Same as safest route (only one option available)" : "Shortest travel time",
                        color: .orange,
                        isSelected: selectedRoute?.id == optimalRoute.id
                    ) {
                        selectedRoute = optimalRoute
                    }
                    
                    // weather & time context
                    if let weather = currentWeather {
                        HStack(spacing: 12) {
                            weatherIcon(for: weather.condition)
                                .foregroundColor(.blue)
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Current Conditions")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                Text("\(weather.condition.displayName) • \(timeOfDayLabel())")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                        .padding(.horizontal)
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
                                Image(systemName: "car.fill")
                                Text("Start Driving")
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
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .background(Color(.systemBackground))
        .cornerRadius(16, corners: [.topLeft, .topRight])
        .shadow(radius: 10)
    }
    
    @ViewBuilder
    private func weatherIcon(for condition: WeatherData.WeatherCondition) -> some View {
        switch condition {
        case .clear:
            Image(systemName: "sun.max.fill")
        case .cloudy:
            Image(systemName: "cloud.fill")
        case .rain, .heavyRain:
            Image(systemName: "cloud.rain.fill")
        case .snow, .heavySnow:
            Image(systemName: "cloud.snow.fill")
        case .fog, .mist:
            Image(systemName: "cloud.fog.fill")
        case .thunderstorm:
            Image(systemName: "cloud.bolt.fill")
        case .sleet:
            Image(systemName: "cloud.sleet.fill")
        }
    }
    
    private func timeOfDayLabel() -> String {
        let calendar = Calendar.current
        let now = Date()
        let hour = calendar.component(.hour, from: now)
        let weekday = calendar.component(.weekday, from: now)
        let isWeekend = weekday == 1 || weekday == 7
        
        // Late night: 11 PM - 4:59 AM
        if hour >= 23 || hour < 5 {
            return "Late Night"
        }
        // Night: 10 PM - 10:59 PM
        else if hour >= 22 {
            return "Night"
        }
        // Early morning: 5 AM - 6:59 AM
        else if hour < 7 {
            return "Early Morning"
        }
        // Morning rush: 7 AM - 9:59 AM
        else if hour >= 7 && hour < 10 {
            return isWeekend ? "Morning" : "Rush Hour"
        }
        // Daytime: 10 AM - 3:59 PM
        else if hour >= 10 && hour < 16 {
            return "Daytime"
        }
        // Evening rush: 4 PM - 6:59 PM
        else if hour >= 16 && hour < 19 {
            return isWeekend ? "Evening" : "Rush Hour"
        }
        // Evening: 7 PM - 9:59 PM
        else {
            return "Evening"
        }
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
        // MKErrorDomain error 5 is MKErrorLoadingThrottled - too many requests
        // This is expected and can be ignored - results will update when throttling ends
        if let mkError = error as? MKError {
            switch mkError.code {
            case .loadingThrottled:
                // Throttled - this is normal, just ignore
                break
            default:
                print("Search completer error: \(error.localizedDescription)")
            }
        } else {
            print("Search completer error: \(error.localizedDescription)")
        }
    }
}
