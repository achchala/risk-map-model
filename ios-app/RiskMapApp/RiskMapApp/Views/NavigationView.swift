//
//  NavigationView.swift
//  RiskMapApp
//
//  navigation view for route planning
//

import SwiftUI
import MapKit

struct RouteNavigationView: View {
    @State private var startPoint: String = ""
    @State private var destination: String = ""
    @State private var startCoordinate: CLLocationCoordinate2D?
    @State private var destinationCoordinate: CLLocationCoordinate2D?
    @State private var cameraPosition = MapCameraPosition.region(MKCoordinateRegion.toronto())
    
    private var hasMapContent: Bool {
        startCoordinate != nil || destinationCoordinate != nil
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text("Navigation")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Enter your starting point and destination")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal)
                .padding(.top)
                .padding(.bottom, 24)
                
                // Route Inputs
                VStack(spacing: 0) {
                    // Start Point with autocomplete
                    AddressAutocompleteField(
                        placeholder: "Starting point",
                        iconName: "mappin.circle.fill",
                        iconColor: .green,
                        text: $startPoint,
                        coordinate: $startCoordinate
                    )
                    
                    Divider()
                        .padding(.leading, 48)
                    
                    // Swap Button
                    Button(action: swapPoints) {
                        Image(systemName: "arrow.up.arrow.down")
                            .foregroundColor(.primary)
                            .padding(8)
                            .background(Color(UIColor.secondarySystemBackground))
                            .clipShape(Circle())
                    }
                    .padding(.vertical, 8)
                    .padding(.leading, 48)
                    
                    Divider()
                        .padding(.leading, 48)
                    
                    // Destination with autocomplete
                    AddressAutocompleteField(
                        placeholder: "Destination",
                        iconName: "flag.fill",
                        iconColor: .red,
                        text: $destination,
                        coordinate: $destinationCoordinate
                    )
                }
                .background(Color(UIColor.systemBackground))
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 2)
                .padding(.horizontal)
                
                // Map overlay showing selected locations
                if hasMapContent {
                    Map(position: $cameraPosition, interactionModes: [.pan, .zoom]) {
                        if let start = startCoordinate {
                            Marker("Start", systemImage: "mappin.circle.fill", coordinate: start)
                                .tint(.green)
                        }
                        if let dest = destinationCoordinate {
                            Marker("Destination", systemImage: "flag.fill", coordinate: dest)
                                .tint(.red)
                        }
                    }
                    .frame(height: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)
                    .padding(.top, 20)
                    .onAppear { updateCamera() }
                    .onChange(of: startPoint) { _, _ in updateCamera() }
                    .onChange(of: destination) { _, _ in updateCamera() }
                }
                
                Spacer()
                
                // Action Buttons
                VStack(spacing: 12) {
                    Button(action: clearInputs) {
                        Text("Clear")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color(UIColor.secondarySystemBackground))
                            .foregroundColor(.primary)
                            .cornerRadius(12)
                    }
                    
                    Button(action: planRoute) {
                        Text("Plan Route")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(canPlanRoute ? Color.blue : Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }
                    .disabled(!canPlanRoute)
                }
                .padding(.horizontal)
                .padding(.bottom, 32)
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("Navigation")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    private var canPlanRoute: Bool {
        startCoordinate != nil && destinationCoordinate != nil
    }
    
    private func swapPoints() {
        let tempPoint = startPoint
        let tempCoord = startCoordinate
        startPoint = destination
        startCoordinate = destinationCoordinate
        destination = tempPoint
        destinationCoordinate = tempCoord
    }
    
    private func clearInputs() {
        startPoint = ""
        destination = ""
        startCoordinate = nil
        destinationCoordinate = nil
    }
    
    private func planRoute() {
        guard let start = startCoordinate, let dest = destinationCoordinate else { return }
        let startItem = MKMapItem(placemark: MKPlacemark(coordinate: start))
        startItem.name = startPoint
        let destItem = MKMapItem(placemark: MKPlacemark(coordinate: dest))
        destItem.name = destination
        MKMapItem.openMaps(
            with: [startItem, destItem],
            launchOptions: [MKLaunchOptionsDirectionsModeKey: MKLaunchOptionsDirectionsModeDriving]
        )
    }
    
    private func updateCamera() {
        guard let start = startCoordinate else {
            if let dest = destinationCoordinate {
                withAnimation(.easeInOut(duration: 0.3)) {
                    cameraPosition = .region(MKCoordinateRegion(
                        center: dest,
                        span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
                    ))
                }
            }
            return
        }
        if let dest = destinationCoordinate {
            let padding = 1.4
            let minLat = min(start.latitude, dest.latitude)
            let maxLat = max(start.latitude, dest.latitude)
            let minLon = min(start.longitude, dest.longitude)
            let maxLon = max(start.longitude, dest.longitude)
            let center = CLLocationCoordinate2D(
                latitude: (minLat + maxLat) / 2,
                longitude: (minLon + maxLon) / 2
            )
            let span = MKCoordinateSpan(
                latitudeDelta: max((maxLat - minLat) * padding, 0.01),
                longitudeDelta: max((maxLon - minLon) * padding, 0.01)
            )
            withAnimation(.easeInOut(duration: 0.3)) {
                cameraPosition = .region(MKCoordinateRegion(center: center, span: span))
            }
        } else {
            withAnimation(.easeInOut(duration: 0.3)) {
                cameraPosition = .region(MKCoordinateRegion(
                    center: start,
                    span: MKCoordinateSpan(latitudeDelta: 0.02, longitudeDelta: 0.02)
                ))
            }
        }
    }
}

struct RouteNavigationView_Previews: PreviewProvider {
    static var previews: some View {
        RouteNavigationView()
    }
}
