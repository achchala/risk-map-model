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

    var body: some View {
        ZStack {
            MapReader { proxy in
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
                    }
                }
            }
            .mapStyle(.standard)
            .mapControls {
                MapUserLocationButton()
                MapCompass()
                MapScaleView()
            }
            .onAppear {
                loadRiskData()
            }
            .onMapCameraChange(frequency: .onEnd) { context in
                // load data when map region changes
                currentRegion = context.region
                loadRiskDataForRegion(context.region)
            }
            .onTapGesture { position in
                handleMapTap(at: position, proxy: proxy)
            }
        } // end MapReader
            
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
        }
        .overlay(alignment: .bottom) {
            if let segment = selectedSegment {
                // Semi-transparent background overlay
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                    .onTapGesture {
                        withAnimation(.spring(response: 0.3)) {
                            selectedSegment = nil
                        }
                    }

                // Bottom sheet popup
                SegmentDetailPopupView(segment: segment) {
                    withAnimation(.spring(response: 0.3)) {
                        selectedSegment = nil
                    }
                }
                .transition(.move(edge: .bottom).combined(with: .opacity))
                .padding(.bottom, 20)
            }
        }
        .animation(.spring(response: 0.3), value: selectedSegment)
    }

    // MARK: - Tap Handler

    private func handleMapTap(at position: CGPoint, proxy: MapProxy) {
        // Convert screen position to map coordinate
        guard let coordinate = proxy.convert(position, from: .local) else {
            return
        }

        // Find nearest road segment to tap location (within 50m threshold)
        if let nearest = findNearestSegment(
            to: coordinate,
            in: riskService.roadSegments
        ) {
            withAnimation(.spring(response: 0.3)) {
                selectedSegment = nearest
            }
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

