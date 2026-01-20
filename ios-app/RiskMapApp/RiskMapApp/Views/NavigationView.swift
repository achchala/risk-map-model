//
//  NavigationView.swift
//  RiskMapApp
//
//  navigation view for route planning
//

import SwiftUI

struct RouteNavigationView: View {
    @State private var startPoint: String = ""
    @State private var destination: String = ""
    
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
                    // Start Point
                    HStack(spacing: 12) {
                        Image(systemName: "mappin.circle.fill")
                            .foregroundColor(.green)
                            .font(.title3)
                        
                        TextField("Starting point", text: $startPoint)
                            .textFieldStyle(.plain)
                    }
                    .padding()
                    .background(Color(UIColor.systemBackground))
                    
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
                    
                    // Destination
                    HStack(spacing: 12) {
                        Image(systemName: "flag.fill")
                            .foregroundColor(.red)
                            .font(.title3)
                        
                        TextField("Destination", text: $destination)
                            .textFieldStyle(.plain)
                    }
                    .padding()
                    .background(Color(UIColor.systemBackground))
                }
                .background(Color(UIColor.systemBackground))
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.1), radius: 8, x: 0, y: 2)
                .padding(.horizontal)
                
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
        !startPoint.trimmingCharacters(in: .whitespaces).isEmpty &&
        !destination.trimmingCharacters(in: .whitespaces).isEmpty
    }
    
    private func swapPoints() {
        let temp = startPoint
        startPoint = destination
        destination = temp
    }
    
    private func clearInputs() {
        startPoint = ""
        destination = ""
    }
    
    private func planRoute() {
        // Placeholder - route planning will be implemented later
        // Start point: startPoint
        // Destination: destination
        print("Planning route from: \(startPoint) to: \(destination)")
    }
}

struct RouteNavigationView_Previews: PreviewProvider {
    static var previews: some View {
        RouteNavigationView()
    }
}
