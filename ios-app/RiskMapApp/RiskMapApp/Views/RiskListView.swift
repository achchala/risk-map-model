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
                if let loc = segment.segmentLocation, !loc.isEmpty {
                    Text(loc)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }
            
            Spacer()
            
            VStack(alignment: .trailing) {
                Text(segment.riskLevel.displayName)
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(Color(hex: segment.riskLevel.color))
                
                Text("\(Int(segment.confidence * 100))%")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}




