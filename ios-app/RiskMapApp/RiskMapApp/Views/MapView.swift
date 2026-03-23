//
//  MapView.swift
//  RiskMapApp
//
//  map view showing road segments with risk levels
//

import SwiftUI
import MapKit

struct MapView: View {
    @EnvironmentObject var riskService: RiskService
    @State private var cameraPosition = MapCameraPosition.region(
        MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832), // Toronto
            span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
        )
    )
    @State private var currentRegion = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832),
        span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
    )
    @State private var selectedSegment: RoadSegment?
    @State private var showDetail = false
    
    var body: some View {
        ZStack {
            Map(position: $cameraPosition, interactionModes: [.pan, .zoom, .rotate, .pitch]) {
                ForEach(riskService.roadSegments) { segment in
                    if !segment.coordinates.isEmpty {
                        let coords = segment.coordinates.map { 
                            CLLocationCoordinate2D(latitude: $0.latitude, longitude: $0.longitude)
                        }
                        let segmentColor = Color(hex: segment.riskLevel.color)
                        MapPolyline(coordinates: coords)
                            .stroke(segmentColor, style: StrokeStyle(lineWidth: 4, lineCap: .round, lineJoin: .round))
                        Annotation("", coordinate: segment.centerCoordinate) {
                            Button {
                                selectedSegment = segment
                            } label: {
                                Circle()
                                    .fill(segmentColor.opacity(0.9))
                                    .frame(width: 12, height: 12)
                                    .overlay(Circle().stroke(Color.white, lineWidth: 1.5))
                            }
                            .buttonStyle(.plain)
                        }
                        .annotationTitles(.hidden)
                    }
                }
            }
            .mapStyle(.standard)
            .mapControls {
                MapCompass()
                MapScaleView()
            }
            .overlay(alignment: .topLeading) {
                RiskMapLegend()
                    .padding(.top, 12)
                    .padding(.leading, 12)
            }
            .overlay(alignment: .bottomTrailing) {
                Button {
                    cameraPosition = .region(MKCoordinateRegion(
                        center: CLLocationCoordinate2D(latitude: 43.6452, longitude: -79.3806), // Union Station Toronto
                        span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
                    ))
                } label: {
                    Image(systemName: "location.fill")
                        .foregroundColor(.brandPrimary)
                        .font(.title2)
                        .padding(10)
                        .background(.ultraThinMaterial)
                        .clipShape(Circle())
                }
                .padding(.trailing, 12)
                .padding(.bottom, 12)
            }
            .onAppear {
                loadRiskData()
                // Preload full segment list for Road Details tab (runs in background)
                Task {
                    guard riskService.allSegmentsForList.isEmpty else { return }
                    try? await riskService.fetchAllRiskPredictionsForList()
                }
            }
            .onMapCameraChange(frequency: .onEnd) { context in
                // load data when map region changes
                currentRegion = context.region
                loadRiskDataForRegion(context.region)
            }
            
            // loading indicator
            if riskService.isLoading {
                ProgressView("Loading risk data...")
                    .tint(.brandPrimary)
                    .padding()
                    .background(Color.white.opacity(0.88))
                    .cornerRadius(10)
            }
            
            // error message
            if let error = riskService.errorMessage {
                VStack {
                    Text("Error: \(error)")
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    Button("Retry") {
                        loadRiskData()
                    }
                    .padding()
                }
            }
            
            // tooltip when segment is selected (anchored at bottom)
            if let segment = selectedSegment {
                VStack {
                    Spacer()
                    SegmentTooltipView(segment: segment) {
                        selectedSegment = nil
                    } onSeeDetails: {
                        showDetail = true
                    }
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 24)
                }
            }
        }
        .sheet(isPresented: $showDetail) {
            if let segment = selectedSegment {
                RiskDetailView(segment: segment)
            }
        }
        .onChange(of: selectedSegment) { _, newValue in
            showDetail = false
        }
    }
    
    private func loadRiskData() {
        // use the current region
        loadRiskDataForRegion(currentRegion)
    }
    
    private func loadRiskDataForRegion(_ region: MKCoordinateRegion) {
        Task {
            do {
                _ = try await riskService.fetchRiskPredictions(for: region, limit: 200)
            } catch {
                let errorMessage = error.localizedDescription
                print("Error loading risk data: \(errorMessage)")
                await MainActor.run {
                    riskService.errorMessage = errorMessage
                }
            }
        }
    }
}

struct RiskMapLegend: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Risk Legend")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(.primary)

            ForEach(RiskLevel.allCases, id: \.self) { level in
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color(hex: level.color))
                        .frame(width: 10, height: 10)
                    Text(level.displayName)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            Text("Top 200 highest risk segments highlighted. Zoom in for more.")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(10)
        .background(Color(.systemBackground).opacity(0.92))
        .cornerRadius(10)
        .shadow(radius: 4)
    }
}

// MARK: - segment tooltip (shown when user taps a road segment)
struct SegmentTooltipView: View {
    let segment: RoadSegment
    let onDismiss: () -> Void
    let onSeeDetails: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(segment.linearName)
                        .font(.headline)
                    HStack(spacing: 6) {
                        Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(Color(hex: segment.riskLevel.color))
                    }
                }
                Spacer()
                Button {
                    onDismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
            }
            if let explanation = segment.riskExplanation, !explanation.isEmpty {
                Text(explanation)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(4)
            }
            Button("See details") {
                onSeeDetails()
            }
            .font(.subheadline)
            .fontWeight(.medium)
            .foregroundColor(.brandPrimary)
        }
        .padding()
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 8)
    }
}

// MARK: - risk annotation
struct RiskAnnotation: View {
    let segment: RoadSegment
    
    var body: some View {
        Image(systemName: segment.riskLevel.systemImage)
            .foregroundColor(Color(hex: segment.riskLevel.color))
            .font(.title2)
            .background(Circle().fill(Color.white))
            .shadow(radius: 3)
    }
}

// MARK: - road segment extension
extension RoadSegment {
    var centerCoordinate: CLLocationCoordinate2D {
        guard !coordinates.isEmpty else {
            return CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
        }
        
        let avgLat = coordinates.map { $0.latitude }.reduce(0, +) / Double(coordinates.count)
        let avgLon = coordinates.map { $0.longitude }.reduce(0, +) / Double(coordinates.count)
        
        return CLLocationCoordinate2D(latitude: avgLat, longitude: avgLon)
    }
}

// MARK: - color extension
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }

    static let brandPrimary = Color(hex: "2F3B69")
    static let brandSecondary = Color(hex: "00BF63")
    static let brandTertiary = Color(hex: "009FC1")
    /// Calming blue for safest route
    static let routeSafestBlue = Color(hex: "5B8FDB")
    /// Faded blue for unselected safest route
    static let routeSafestBlueFaded = Color(hex: "5B8FDB").opacity(0.45)
    /// Yellow for fastest route
    static let routeFastestYellow = Color(hex: "E6A800")
    /// Faded yellow for unselected fastest route
    static let routeFastestYellowFaded = Color(hex: "E6A800").opacity(0.45)
    static let brandPrimarySoft = Color.brandPrimary.opacity(0.14)
    static let brandSecondarySoft = Color.brandSecondary.opacity(0.14)
    static let brandTertiarySoft = Color.brandTertiary.opacity(0.14)
}

