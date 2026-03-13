//
//  NavigationView.swift
//  RiskMapApp
//
//  Route planning with safety-aware backend, map, and avoided segments
//

import SwiftUI
import MapKit
import CoreLocation

struct RouteNavigationView: View {
    @EnvironmentObject var weatherService: WeatherService
    @StateObject private var routeService = RouteService()
    @State private var startPoint: String = ""
    @State private var destination: String = ""
    @State private var safetyPreference: Double = 0.5 // 0 = faster, 1 = safer
    @State private var originCoord: CLLocationCoordinate2D?
    @State private var destinationCoord: CLLocationCoordinate2D?
    @State private var selectedAvoidedSegment: AvoidedSegment?
    @State private var selectedRouteOption: RouteOptionChoice = .safer
    @State private var cameraPosition: MapCameraPosition = .region(.toronto())
    @State private var currentWeather: WeatherData?

    /// Higher maxBeta makes the safer route meaningfully avoid high-risk segments.
    /// beta = hours of "time equivalent" per expected crash on an edge.
    private let maxBeta = 15.0
    private var beta: Double { safetyPreference * maxBeta }

    var body: some View {
        NavigationView {
            ZStack(alignment: .top) {
                mapLayer
                weatherBadge
                VStack(spacing: 0) {
                    inputCard
                    if let r = routeService.safetyAwareResponse {
                        routeSummaryCard(r)
                        exportButtons(r)
                        avoidedSegmentsList(r.avoidedSegments)
                    }
                    Spacer(minLength: 0)
                }
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("Navigation")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(item: $selectedAvoidedSegment) { seg in
                AvoidedSegmentDetailView(segment: seg)
            }
            .onChange(of: selectedAvoidedSegment) { _, newValue in
                if let seg = newValue {
                    // Pan map to show the avoided segment when selected
                    withAnimation(.easeInOut(duration: 0.4)) {
                        cameraPosition = .region(MKCoordinateRegion(
                            center: seg.centerCoordinate,
                            span: MKCoordinateSpan(latitudeDelta: 0.008, longitudeDelta: 0.008)
                        ))
                    }
                }
            }
            .onAppear { loadWeather() }
        }
    }

    @ViewBuilder
    private var weatherBadge: some View {
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
            .background(Color(UIColor.systemBackground).opacity(0.95))
            .cornerRadius(10)
            .shadow(radius: 5)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(.top, 100)
            .padding(.leading, 16)
        }
    }

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

    private func loadWeather() {
        Task {
            let location = originCoord ?? CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
            let weather = await weatherService.getWeatherData(for: location)
            await MainActor.run { currentWeather = weather }
        }
    }

    private var mapLayer: some View {
        Map(position: $cameraPosition) {
            if let r = routeService.safetyAwareResponse {
                // Fastest route: purple when selected, orange otherwise
                let fastestCoords = r.fastest.fullRouteCoordinates
                if !fastestCoords.isEmpty {
                    let isSelected = selectedRouteOption == .fastest
                    MapPolyline(coordinates: fastestCoords)
                        .stroke(isSelected ? .purple : .orange, lineWidth: isSelected ? 7 : 5)
                }
                // Safer route: purple when selected, blue otherwise
                let saferCoords = r.safer.fullRouteCoordinates
                if !saferCoords.isEmpty {
                    let isSelected = selectedRouteOption == .safer
                    MapPolyline(coordinates: saferCoords)
                        .stroke(isSelected ? .purple : .blue, lineWidth: isSelected ? 7 : 5)
                }
                // Avoided segments: top 10 only, polylines only (no annotations)
                let topAvoided = Array(r.avoidedSegments.sorted { $0.lambdaPerHour > $1.lambdaPerHour }.prefix(10))
                ForEach(topAvoided) { seg in
                    if !seg.coordinates.isEmpty {
                        let coords = seg.coordinates.map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) }
                        MapPolyline(coordinates: coords)
                            .stroke(Color.red, style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round, dash: [12, 8]))
                    }
                }
            }
        }
        .mapStyle(.standard)
        .ignoresSafeArea(edges: .top)
    }

    private var inputCard: some View {
        VStack(spacing: 12) {
            VStack(spacing: 0) {
                AddressSearchField(
                    placeholder: "Starting point",
                    iconName: "mappin.circle.fill",
                    iconColor: .green,
                    text: $startPoint
                ) { coord in
                    originCoord = coord
                }
                Divider().padding(.leading, 44)
                AddressSearchField(
                    placeholder: "Destination",
                    iconName: "flag.fill",
                    iconColor: .red,
                    text: $destination
                ) { coord in
                    destinationCoord = coord
                }
            }
            .background(Color(UIColor.systemBackground))
            .cornerRadius(12)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Faster").font(.caption).foregroundColor(.secondary)
                    Spacer()
                    Text("Safer").font(.caption).foregroundColor(.secondary)
                }
                Slider(value: $safetyPreference, in: 0...1)
            }
            .padding(.horizontal, 4)

            if let msg = routeService.errorMessage {
                Text(msg).font(.caption).foregroundColor(.red).padding(.horizontal)
            }

            HStack(spacing: 12) {
                Button("Clear") { clearInputs() }
                    .frame(maxWidth: .infinity).padding().background(Color(UIColor.secondarySystemBackground)).cornerRadius(10)
                Button(action: planRoute) {
                    if routeService.isLoading {
                        ProgressView().progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Text("Plan Route")
                    }
                }
                .frame(maxWidth: .infinity).padding().background(canPlanRoute ? Color.blue : Color.gray).foregroundColor(.white).cornerRadius(10)
                .disabled(!canPlanRoute || routeService.isLoading)
            }
        }
        .padding()
        .background(Color(UIColor.systemBackground).opacity(0.95))
        .cornerRadius(16)
        .shadow(radius: 8)
        .padding()
    }

    private enum RouteOptionChoice {
        case fastest
        case safer
    }

    private func routeSummaryCard(_ r: SafetyAwareResponse) -> some View {
        let f = r.fastest.summary
        let s = r.safer.summary
        let fHigh = f.highRiskSegments ?? 0
        let fMed = f.mediumRiskSegments ?? 0
        let fLow = f.lowRiskSegments ?? 0
        let sHigh = s.highRiskSegments ?? 0
        let sMed = s.mediumRiskSegments ?? 0
        let sLow = s.lowRiskSegments ?? 0
        let highAvoided = fHigh - sHigh
        let lowGained = sLow - fLow

        return VStack(alignment: .leading, spacing: 8) {
            Text("Routes").font(.headline)
            HStack(spacing: 16) {
                Button(action: { selectedRouteOption = .fastest }) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Rectangle().fill(Color.orange).frame(width: 12, height: 4)
                            Text("Fastest").font(.subheadline).fontWeight(.medium)
                            if selectedRouteOption == .fastest {
                                Image(systemName: "checkmark.circle.fill").font(.caption).foregroundColor(.orange)
                            }
                        }
                        Text(formatHours(f.totalTravelTimeHours))
                            .font(.caption).foregroundColor(.secondary)
                        riskSegmentBadges(high: fHigh, medium: fMed, low: fLow)
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(selectedRouteOption == .fastest ? Color.orange.opacity(0.1) : Color.clear)
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)

                Button(action: { selectedRouteOption = .safer }) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack(spacing: 6) {
                            Rectangle().fill(Color.blue).frame(width: 12, height: 4)
                            Text("Safer").font(.subheadline).fontWeight(.medium)
                            if selectedRouteOption == .safer {
                                Image(systemName: "checkmark.circle.fill").font(.caption).foregroundColor(.blue)
                            }
                        }
                        Text(formatHours(s.totalTravelTimeHours))
                            .font(.caption).foregroundColor(.secondary)
                        riskSegmentBadges(high: sHigh, medium: sMed, low: sLow)
                    }
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(selectedRouteOption == .safer ? Color.blue.opacity(0.1) : Color.clear)
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
            if highAvoided > 0 || lowGained > 0 {
                HStack(spacing: 6) {
                    Image(systemName: "shield.checkered")
                        .font(.caption)
                        .foregroundColor(.blue)
                    Text(safetyImprovementText(highAvoided: highAvoided, lowGained: lowGained))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            if !r.avoidedSegments.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundColor(.red)
                    Text("Safer route avoids \(r.avoidedSegments.count) higher-risk segment(s). Tap a segment for details.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
        .padding(.horizontal)
    }

    private func riskSegmentBadges(high: Int, medium: Int, low: Int) -> some View {
        HStack(spacing: 6) {
            if high > 0 {
                Label("\(high)", systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2).foregroundColor(.red)
            }
            if medium > 0 {
                Label("\(medium)", systemImage: "exclamationmark.circle.fill")
                    .font(.caption2).foregroundColor(.orange)
            }
            if low > 0 {
                Label("\(low)", systemImage: "checkmark.circle.fill")
                    .font(.caption2).foregroundColor(.green)
            }
        }
    }

    private func safetyImprovementText(highAvoided: Int, lowGained: Int) -> String {
        var parts: [String] = []
        if highAvoided > 0 {
            parts.append("\(highAvoided) fewer high-risk segment\(highAvoided == 1 ? "" : "s")")
        }
        if lowGained > 0 {
            parts.append("\(lowGained) more low-risk segment\(lowGained == 1 ? "" : "s")")
        }
        return "Safer route: " + (parts.isEmpty ? "optimized for safety" : parts.joined(separator: ", "))
    }

    private func exportButtons(_ r: SafetyAwareResponse) -> some View {
        HStack(spacing: 12) {
            Button(action: { exportToAppleMaps(r) }) {
                Label("Apple Maps", systemImage: "map.fill")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            Button(action: { exportToGoogleMaps(r) }) {
                Label("Google Maps", systemImage: "globe")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding(.horizontal)
        .padding(.bottom, 4)
    }

    private func exportToAppleMaps(_ r: SafetyAwareResponse) {
        guard let origin = originCoord, let dest = destinationCoord else { return }
        let startItem = MKMapItem(placemark: MKPlacemark(coordinate: origin))
        startItem.name = startPoint.isEmpty ? "Start" : startPoint
        let destItem = MKMapItem(placemark: MKPlacemark(coordinate: dest))
        destItem.name = destination.isEmpty ? "Destination" : destination

        let coords = selectedRouteOption == .safer ? r.safer.fullRouteCoordinates : r.fastest.fullRouteCoordinates
        var mapItems: [MKMapItem] = [startItem]
        let maxWaypoints = 23
        if coords.count > 2 {
            let step = max(1, (coords.count - 2) / maxWaypoints)
            for i in stride(from: step, to: coords.count - 1, by: step) {
                mapItems.append(MKMapItem(placemark: MKPlacemark(coordinate: coords[i])))
            }
        }
        mapItems.append(destItem)

        MKMapItem.openMaps(
            with: Array(mapItems.prefix(25)),
            launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving]
        )
    }

    private func exportToGoogleMaps(_ r: SafetyAwareResponse) {
        guard let origin = originCoord, let dest = destinationCoord else { return }
        let coords = selectedRouteOption == .safer ? r.safer.fullRouteCoordinates : r.fastest.fullRouteCoordinates
        var waypoints: [String] = []
        let maxWaypoints = 23
        if coords.count > 2 {
            let step = max(1, (coords.count - 2) / maxWaypoints)
            for i in stride(from: step, to: coords.count - 1, by: step) {
                waypoints.append("\(coords[i].latitude),\(coords[i].longitude)")
            }
        }
        let limited = Array(waypoints.prefix(maxWaypoints))
        var url = "https://www.google.com/maps/dir/?api=1&origin=\(origin.latitude),\(origin.longitude)&destination=\(dest.latitude),\(dest.longitude)&travelmode=driving"
        if !limited.isEmpty {
            url += "&waypoints=\(limited.joined(separator: "|"))"
        }
        if let u = URL(string: url.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? url) {
            UIApplication.shared.open(u)
        }
    }

    private func avoidedSegmentsList(_ segments: [AvoidedSegment]) -> some View {
        Group {
            if segments.isEmpty { EmptyView() }
            else {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text("Segments you're avoiding").font(.headline)
                    }
                    Text("Tap a segment to see why it's risky")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(segments) { seg in
                                Button(action: { selectedAvoidedSegment = seg }) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(seg.LINEAR_NAME).font(.subheadline).fontWeight(.medium).lineLimit(1)
                                        if let loc = seg.segmentLocation, !loc.isEmpty {
                                            Text(loc).font(.caption2).foregroundColor(.secondary).lineLimit(1)
                                        }
                                        HStack(spacing: 4) {
                                            Text(seg.riskLevel.displayName).font(.caption2).foregroundColor(riskColor(seg.risk_label))
                                            Text("•").font(.caption2).foregroundColor(.secondary)
                                            Text(seg.ROAD_CLASS).font(.caption2).foregroundColor(.secondary)
                                        }
                                    }
                                    .padding(10)
                                    .frame(width: 150, alignment: .leading)
                                    .background(Color.red.opacity(0.08))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(Color.red.opacity(0.3), lineWidth: 1)
                                    )
                                    .cornerRadius(8)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.horizontal, 4)
                    }
                    .frame(height: 96)
                }
                .padding()
                .background(Color(UIColor.systemBackground))
                .cornerRadius(12)
                .padding(.horizontal)
            }
        }
    }

    private func riskColor(_ label: String) -> Color {
        switch label {
        case "high": return .red
        case "medium": return .orange
        default: return .green
        }
    }

    private func formatHours(_ h: Double) -> String {
        let m = Int(round(h * 60))
        return m < 60 ? "\(m) min" : "\(m / 60)h \(m % 60)m"
    }

    private var canPlanRoute: Bool {
        !startPoint.trimmingCharacters(in: .whitespaces).isEmpty &&
        !destination.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func swapPoints() {
        let t = startPoint
        startPoint = destination
        destination = t
    }

    private func clearInputs() {
        startPoint = ""
        destination = ""
        originCoord = nil
        destinationCoord = nil
        routeService.safetyAwareResponse = nil
        routeService.errorMessage = nil
    }

    private func planRoute() {
        Task {
            let origin: CLLocationCoordinate2D
            let dest: CLLocationCoordinate2D
            if let o = originCoord, let d = destinationCoord {
                origin = o
                dest = d
            } else {
                let geocoder = CLGeocoder()
                guard let startPlace = try? await geocoder.geocodeAddressString(startPoint).first,
                      let destPlace = try? await geocoder.geocodeAddressString(destination).first,
                      let o = startPlace.location?.coordinate,
                      let d = destPlace.location?.coordinate else {
                    await MainActor.run {
                        routeService.errorMessage = "Could not find location. Try a full address in Toronto."
                    }
                    return
                }
                origin = o
                dest = d
                await MainActor.run {
                    originCoord = o
                    destinationCoord = d
                }
            }
            await MainActor.run {
                originCoord = origin
                destinationCoord = dest
                let region = MKCoordinateRegion(
                    center: CLLocationCoordinate2D(
                        latitude: (origin.latitude + dest.latitude) / 2,
                        longitude: (origin.longitude + dest.longitude) / 2
                    ),
                    span: MKCoordinateSpan(latitudeDelta: 0.05, longitudeDelta: 0.05)
                )
                cameraPosition = .region(region)
            }
            let weather = await weatherService.getWeatherData(for: origin)
            await MainActor.run { currentWeather = weather }
            let weatherDict = weather.toBackendDict()
            let timeDict: [String: Any] = [
                "hour": Calendar.current.component(.hour, from: Date()),
                "is_weekend": [1, 7].contains(Calendar.current.component(.weekday, from: Date()))
            ]
            do {
                _ = try await routeService.fetchSafetyAwareRoutes(origin: origin, destination: dest, beta: beta, weather: weatherDict, timeOfDay: timeDict)
            } catch {}
        }
    }
}

// MARK: - Avoided segment detail (mirrors RiskDetailView for avoided segments)
struct AvoidedSegmentDetailView: View {
    let segment: AvoidedSegment
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text("You avoided this segment by taking the safer route.")
                        .font(.subheadline).foregroundColor(.secondary)
                        .padding(.bottom, 4)
                    VStack(alignment: .leading, spacing: 8) {
                        Text(segment.LINEAR_NAME).font(.title2).fontWeight(.bold)
                        if let loc = segment.segmentLocation, !loc.isEmpty {
                            Text(loc).font(.subheadline).foregroundColor(.secondary)
                        }
                        Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                            .foregroundColor(Color(hex: segment.riskLevel.color))
                        Text("Road class: \(segment.ROAD_CLASS)").font(.subheadline).foregroundColor(.secondary)
                        Text("Predicted crash rate (λ): \(String(format: "%.4f", segment.lambdaPerHour))/hr")
                            .font(.caption).foregroundColor(.secondary)
                        if let explanation = segment.riskExplanation, !explanation.isEmpty {
                            Text(explanation)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .lineSpacing(4)
                        }
                        if let drivers = segment.riskDrivers, !drivers.isEmpty {
                            let factors = Array(drivers.sorted(by: { abs($0.value) > abs($1.value) }).prefix(3))
                                .map { formatDriverLabel($0.key) + " " + formatDriverValue(key: $0.key, value: $0.value) }
                                .joined(separator: "; ")
                            Text("Contributing factors: \(factors)")
                                .font(.caption2).foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(hex: segment.riskLevel.color).opacity(0.1))
                    .cornerRadius(12)
                }
                .padding()
            }
            .navigationTitle("Avoided Segment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

extension AvoidedSegment {
    var centerCoordinate: CLLocationCoordinate2D {
        guard !coordinates.isEmpty else {
            return CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
        }
        let avgLat = coordinates.map { $0.latitude }.reduce(0, +) / Double(coordinates.count)
        let avgLon = coordinates.map { $0.longitude }.reduce(0, +) / Double(coordinates.count)
        return CLLocationCoordinate2D(latitude: avgLat, longitude: avgLon)
    }
}

private func formatDriverLabel(_ key: String) -> String {
    let labels: [String: String] = [
        "crashes_1d_ago": "Crashes (24h)", "crashes_7d_ago": "Crashes (7d)", "crashes_30d_ago": "Crashes (30d)",
        "rolling_mean_7d": "7d avg", "rolling_max_7d": "7d peak", "hist_crashes_per_year": "Hist/yr",
        "from_intersection_degree": "Intersection (from)", "to_intersection_degree": "Intersection (to)",
        "segment_length": "Length", "datetime_hour": "Hour", "day_of_week": "Day", "is_weekend": "Weekend", "month": "Month"
    ]
    return labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized
}

private func formatDriverValue(key: String, value: Double) -> String {
    if key.contains("ratio") && value <= 1 { return String(format: "%.0f%%", value * 100) }
    if key.contains("length") { return String(format: "%.0fm", value) }
    if value == floor(value) { return String(format: "%.0f", value) }
    return String(format: "%.2f", value)
}

struct RouteNavigationView_Previews: PreviewProvider {
    static var previews: some View {
        RouteNavigationView()
            .environmentObject(WeatherService())
    }
}
