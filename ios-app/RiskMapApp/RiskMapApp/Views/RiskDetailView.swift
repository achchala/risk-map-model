//
//  RiskDetailView.swift
//  RiskMapApp
//
//  detail view for a road segment showing risk information
//

import SwiftUI
import MapKit

struct RiskDetailView: View {
    let segment: RoadSegment
    @EnvironmentObject var riskService: RiskService
    @Environment(\.dismiss) var dismiss
    @State private var cameraPosition: MapCameraPosition

    init(segment: RoadSegment) {
        self.segment = segment
        let coords = segment.coordinates.map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) }
        let validCoords = coords.filter { $0.latitude.isFinite && $0.longitude.isFinite && $0.latitude >= -90 && $0.latitude <= 90 && $0.longitude >= -180 && $0.longitude <= 180 }
        if validCoords.isEmpty {
            _cameraPosition = State(initialValue: .region(MKCoordinateRegion(
                center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
                span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
            )))
        } else {
            let minLat = validCoords.map(\.latitude).min()!
            let maxLat = validCoords.map(\.latitude).max()!
            let minLon = validCoords.map(\.longitude).min()!
            let maxLon = validCoords.map(\.longitude).max()!
            let pad: CLLocationDegrees = 0.0015
            let region = MKCoordinateRegion(
                center: CLLocationCoordinate2D(
                    latitude: (minLat + maxLat) / 2,
                    longitude: (minLon + maxLon) / 2
                ),
                span: MKCoordinateSpan(
                    latitudeDelta: max(maxLat - minLat + pad * 2, 0.003),
                    longitudeDelta: max(maxLon - minLon + pad * 2, 0.003)
                )
            )
            _cameraPosition = State(initialValue: .region(region))
        }
    }

    private var segmentCoordinates: [CLLocationCoordinate2D] {
        segment.coordinates
            .map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) }
            .filter { $0.latitude.isFinite && $0.longitude.isFinite }
    }

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Map showing exact segment
                    if !segmentCoordinates.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Location")
                                .font(.headline)
                            Map(position: $cameraPosition, interactionModes: [.pan, .zoom]) {
                                MapPolyline(coordinates: segmentCoordinates)
                                    .stroke(Color(hex: segment.riskLevel.color), style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round))
                            }
                            .frame(height: 180)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                    }

                    // Header: road name + location
                    VStack(alignment: .leading, spacing: 8) {
                        Text(segment.linearName)
                            .font(.title)
                            .fontWeight(.bold)

                        if let loc = segment.segmentLocation, !loc.isEmpty, !isRawCoordinates(loc) {
                            Text(loc)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }

                        HStack {
                            Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                                .foregroundColor(Color(hex: segment.riskLevel.color))
                                .font(.headline)

                            Spacer()

                            Text(confidenceDescription(segment.confidence))
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(hex: segment.riskLevel.color).opacity(0.1))
                    .cornerRadius(12)

                    // Plain-language summary
                    VStack(alignment: .leading, spacing: 8) {
                        Text("What this means")
                            .font(.headline)

                        Text(plainLanguageSummary)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .lineSpacing(6)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Road information
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Road Information")
                            .font(.headline)

                        InfoRow(label: "Road type", value: formatRoadClass(segment.roadClass))
                        InfoRow(label: "Length", value: formatSegmentLength(segment.segmentLength))
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Crash history (historical record on this segment)
                    if segment.numTotalCrashes > 0 || segment.numKSICrashes > 0 || segment.fatalityCount > 0 {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Crash History")
                                .font(.headline)

                            Text("Recorded crashes on this road segment (Toronto Police data):")
                                .font(.subheadline)
                                .foregroundColor(.secondary)

                            if segment.numTotalCrashes > 0 {
                                InfoRow(label: "Total crashes", value: "\(segment.numTotalCrashes)")
                            }
                            if segment.numKSICrashes > 0 {
                                InfoRow(label: "Serious injuries (KSI)", value: "\(segment.numKSICrashes)")
                            }
                            if segment.fatalityCount > 0 {
                                InfoRow(label: "Fatalities", value: "\(segment.fatalityCount)")
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // Contributing factors (human-readable, ranked by model importance when available)
                    if let drivers = segment.riskDrivers, !drivers.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Why this rating?")
                                .font(.headline)

                            Text("Our model considers these factors when estimating risk:")
                                .font(.subheadline)
                                .foregroundColor(.secondary)

                            ForEach(Array(sortedRiskDrivers(drivers).prefix(5)), id: \.key) { item in
                                HStack(alignment: .top) {
                                    Text(formatDriverLabel(item.key))
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    Text(formatDriverValue(key: item.key, value: item.value))
                                        .fontWeight(.medium)
                                        .multilineTextAlignment(.trailing)
                                }
                                .font(.subheadline)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // Tips for the user
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Stay safe")
                            .font(.headline)

                        VStack(alignment: .leading, spacing: 8) {
                            TipRow(icon: "eye.fill", text: "Stay alert and scan for hazards")
                            TipRow(icon: "speedometer", text: "Drive at or below the speed limit")
                            TipRow(icon: "person.2.fill", text: "Watch for pedestrians and cyclists")
                            TipRow(icon: "iphone.slash", text: "Avoid distractions")
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
                .padding()
            }
            .navigationTitle("Road Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }

    private func isRawCoordinates(_ text: String) -> Bool {
        text.contains("(") && text.contains(")") && text.contains(",") && text.range(of: #"\d+\.\d+"#, options: .regularExpression) != nil
    }

    private func confidenceDescription(_ confidence: Double) -> String {
        let pct = Int(confidence * 100)
        switch pct {
        case 0..<40: return "\(pct)% confidence (low certainty)"
        case 40..<70: return "\(pct)% confidence (moderate certainty)"
        default: return "\(pct)% confidence (high certainty)"
        }
    }

    private var plainLanguageSummary: String {
        switch segment.riskLevel {
        case .high:
            return "This road has a higher predicted crash rate than most streets in the area. Historical data and road features (intersections, traffic patterns, time of day) suggest extra caution is warranted here."
        case .medium:
            return "This road has a moderate predicted crash rate. Some factors increase risk compared to safer streets. Drive with normal caution."
        case .low:
            return "This road has a lower predicted crash rate than most streets. It's still important to drive safely, but the model suggests relatively lower risk here."
        }
    }

    private func formatRoadClass(_ roadClass: String) -> String {
        let lower = roadClass.lowercased()
        switch lower {
        case "arterial": return "Major road (arterial)"
        case "collector": return "Collector road"
        case "local": return "Local street"
        case "minor_arterial": return "Minor arterial"
        default: return roadClass.capitalized
        }
    }

    private func formatSegmentLength(_ meters: Double) -> String {
        if meters < 100 { return "\(Int(meters))m (short)" }
        if meters < 300 { return "\(Int(meters))m (medium)" }
        return "\(Int(meters))m (long)"
    }

    /// Sort risk drivers by model feature importance when available; otherwise by magnitude.
    private func sortedRiskDrivers(_ drivers: [String: Double]) -> [(key: String, value: Double)] {
        let imp = riskService.featureImportance ?? [:]
        return drivers.sorted { a, b in
            let impA = imp[a.key] ?? 0
            let impB = imp[b.key] ?? 0
            if impA > 0 || impB > 0 {
                return impA > impB
            }
            return abs(a.value) > abs(b.value)
        }
    }
}

private struct TipRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 24, alignment: .center)
            Text(text)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

private func formatDriverLabel(_ key: String) -> String {
    if key.hasPrefix("road_class_") {
        let name = key.replacingOccurrences(of: "road_class_", with: "").replacingOccurrences(of: "_", with: " ")
        return "Road class: \(name)"
    }
    let labels: [String: String] = [
        "crashes_1d_ago": "Recent crashes (24h)",
        "crashes_7d_ago": "Recent crashes (7 days)",
        "crashes_30d_ago": "Recent crashes (30 days)",
        "rolling_mean_7d": "7-day crash average",
        "rolling_max_7d": "7-day crash peak",
        "hist_crashes_per_year": "Historical crashes per year",
        "from_intersection_degree": "Intersection complexity (start)",
        "to_intersection_degree": "Intersection complexity (end)",
        "segment_length": "Segment length",
        "datetime_hour": "Time of day",
        "day_of_week": "Day of week",
        "is_weekend": "Weekend vs weekday",
        "month": "Season",
        "avg_daily_vol": "Daily traffic volume",
        "avg_speed": "Average speed",
        "tmc_daily_ped_vol": "Pedestrian volume",
        "tmc_daily_cyclist_vol": "Cyclist volume",
        "is_school_zone": "School zone",
        "nearby_transit_frequency": "Transit frequency",
        "temperature": "Temperature",
        "precipitation": "Precipitation",
        "snow_depth_mm": "Snow depth",
        "wind_speed": "Wind speed",
        "is_freezing": "Freezing conditions",
        "is_precip": "Precipitation present",
    ]
    return labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized
}

private func formatDriverValue(key: String, value: Double) -> String {
    if key.contains("ratio") && value <= 1 { return String(format: "%.0f%%", value * 100) }
    if key.contains("length") { return String(format: "%.0fm", value) }
    if key.hasPrefix("road_class_") && value >= 0.99 { return "Yes" }

    // Human-readable values for common factors
    switch key {
    case "from_intersection_degree", "to_intersection_degree":
        let n = Int(value)
        switch n {
        case 2: return "2-way (simple)"
        case 3: return "3-way (T-junction)"
        case 4: return "4-way intersection"
        case 5...: return "\(n)-way (complex)"
        default: return "\(n)-way"
        }
    case "is_weekend":
        return value >= 0.5 ? "Weekend" : "Weekday"
    case "month":
        let months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        let idx = Int(value)
        guard idx >= 1, idx <= 12 else { return "Month \(idx)" }
        return months[idx]
    case "datetime_hour":
        let h = Int(value)
        if h >= 6 && h < 10 { return "Morning rush" }
        if h >= 16 && h < 19 { return "Evening rush" }
        if h >= 22 || h < 6 { return "Late night" }
        return "\(h):00"
    case "day_of_week":
        let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        let idx = Int(value) % 7
        return days.indices.contains(idx) ? days[idx] : "Day \(Int(value))"
    case "is_school_zone", "is_freezing", "is_precip":
        return value >= 0.5 ? "Yes" : "No"
    case "avg_daily_vol", "tmc_daily_ped_vol", "tmc_daily_cyclist_vol", "tmc_daily_vehicle_vol":
        return value >= 1000 ? String(format: "%.1fk", value / 1000) : String(format: "%.0f", value)
    case "avg_speed":
        return String(format: "%.0f km/h", value)
    case "temperature":
        return String(format: "%.0f°C", value)
    case "precipitation", "snow_depth_mm":
        return String(format: "%.1f mm", value)
    case "wind_speed":
        return String(format: "%.0f km/h", value)
    default:
        if value == floor(value) { return String(format: "%.0f", value) }
        return String(format: "%.2f", value)
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
    }
}




