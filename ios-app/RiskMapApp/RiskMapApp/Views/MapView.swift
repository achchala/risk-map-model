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
                    // draw road segments as colored polylines
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
            .overlay(alignment: .bottomTrailing) {
                Button {
                    cameraPosition = .region(MKCoordinateRegion(
                        center: CLLocationCoordinate2D(latitude: 43.6452, longitude: -79.3806), // Union Station Toronto
                        span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
                    ))
                } label: {
                    Image(systemName: "location.fill")
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
            }
            .onMapCameraChange(frequency: .onEnd) { context in
                // load data when map region changes
                currentRegion = context.region
                loadRiskDataForRegion(context.region)
            }
            
            // loading indicator
            if riskService.isLoading {
                ProgressView("Loading risk data...")
                    .padding()
                    .background(Color.white.opacity(0.8))
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
                _ = try await riskService.fetchRiskPredictions(for: region)
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
                        Text("•")
                            .foregroundColor(.secondary)
                        Text("\(Int(segment.confidence * 100))% confidence")
                            .font(.caption)
                            .foregroundColor(.secondary)
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
}

