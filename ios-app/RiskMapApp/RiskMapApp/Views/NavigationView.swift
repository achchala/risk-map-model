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
    @StateObject private var routeService = RouteService()
    @State private var startPoint: String = ""
    @State private var destination: String = ""
    @State private var safetyPreference: Double = 0.5 // 0 = faster, 1 = safer
    @State private var originCoord: CLLocationCoordinate2D?
    @State private var destinationCoord: CLLocationCoordinate2D?
    @State private var selectedAvoidedSegment: AvoidedSegment?
    @State private var cameraPosition: MapCameraPosition = .region(.toronto())

    private let maxBeta = 0.5
    private var beta: Double { safetyPreference * maxBeta }

    var body: some View {
        NavigationView {
            ZStack(alignment: .top) {
                mapLayer
                VStack(spacing: 0) {
                    inputCard
                    if let r = routeService.safetyAwareResponse {
                        routeSummaryCard(r)
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
        }
    }

    private var mapLayer: some View {
        Map(position: $cameraPosition) {
            if let r = routeService.safetyAwareResponse {
                // Fastest route (orange)
                ForEach(Array(r.fastest.segments.enumerated()), id: \.offset) { _, seg in
                    if !seg.coordinates.isEmpty {
                        MapPolyline(coordinates: seg.coordinates.map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) })
                            .stroke(.orange, lineWidth: 5)
                    }
                }
                // Safer route (blue)
                ForEach(Array(r.safer.segments.enumerated()), id: \.offset) { _, seg in
                    if !seg.coordinates.isEmpty {
                        MapPolyline(coordinates: seg.coordinates.map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) })
                            .stroke(.blue, lineWidth: 5)
                    }
                }
                // Avoided segments: red dashed line + tappable marker
                ForEach(r.avoidedSegments) { seg in
                    if !seg.coordinates.isEmpty {
                        let coords = seg.coordinates.map { CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude) }
                        MapPolyline(coordinates: coords)
                            .stroke(Color.red, style: StrokeStyle(lineWidth: 6, lineCap: .round, lineJoin: .round, dash: [12, 8]))
                        Annotation("", coordinate: seg.centerCoordinate) {
                            Button {
                                selectedAvoidedSegment = seg
                            } label: {
                                ZStack {
                                    Circle()
                                        .fill(Color.red.opacity(0.3))
                                        .frame(width: 28, height: 28)
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .font(.system(size: 14))
                                        .foregroundColor(.red)
                                }
                            }
                            .buttonStyle(.plain)
                        }
                        .annotationTitles(.hidden)
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

    private func routeSummaryCard(_ r: SafetyAwareResponse) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Routes").font(.headline)
            HStack(spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Rectangle().fill(Color.orange).frame(width: 12, height: 4)
                        Text("Fastest").font(.subheadline).fontWeight(.medium)
                    }
                    Text("\(formatHours(r.fastest.summary.totalTravelTimeHours)) · \(String(format: "%.3f", r.fastest.summary.expectedCrashes)) expected crashes")
                        .font(.caption).foregroundColor(.secondary)
                }
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Rectangle().fill(Color.blue).frame(width: 12, height: 4)
                        Text("Safer").font(.subheadline).fontWeight(.medium)
                    }
                    Text("\(formatHours(r.safer.summary.totalTravelTimeHours)) · \(String(format: "%.3f", r.safer.summary.expectedCrashes)) expected crashes")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
            if !r.avoidedSegments.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundColor(.red)
                    Text("Safer route avoids \(r.avoidedSegments.count) higher-risk segment(s). Tap a segment on the map or in the list below for details.")
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
            do {
                _ = try await routeService.fetchSafetyAwareRoutes(origin: origin, destination: dest, beta: beta)
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
    }
}
