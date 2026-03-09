//
//  RiskListView.swift
//  RiskMapApp
//
//  list view showing high-risk roads
//

import SwiftUI

struct RiskListView: View {
    @EnvironmentObject var riskService: RiskService
    
    var highRiskRoads: [RoadSegment] {
        riskService.getHighRiskRoads()
    }
    
    var body: some View {
        NavigationView {
            List {
                if highRiskRoads.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "map")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)
                        Text("No high-risk roads found")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text("Load the map to see risk predictions")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                } else {
                    ForEach(highRiskRoads) { segment in
                        NavigationLink(destination: RiskDetailView(segment: segment)) {
                            RiskListRow(segment: segment)
                        }
                    }
                }
            }
            .navigationTitle("High Risk Roads")
            .refreshable {
                // refresh data if needed
            }
        }
    }
}

struct RiskListRow: View {
    let segment: RoadSegment
    
    var body: some View {
        HStack {
            // risk indicator
            Image(systemName: segment.riskLevel.systemImage)
                .foregroundColor(Color(hex: segment.riskLevel.color))
                .font(.title3)
            
            VStack(alignment: .leading, spacing: 4) {
                Text(segment.linearName)
                    .font(.headline)
                
                if segment.numTotalCrashes > 0 {
                    Label("\(segment.numTotalCrashes) crash(es) in recent period", systemImage: "car.fill")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    Text(segment.roadClass)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(segment.displayRiskScore)")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(Color(hex: segment.riskLevel.color))
                Text("risk score")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}




