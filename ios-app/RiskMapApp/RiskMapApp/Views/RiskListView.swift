//
//  RiskListView.swift
//  RiskMapApp
//
//  list view showing high-risk roads
//

import SwiftUI

struct RiskListView: View {
    @EnvironmentObject var riskService: RiskService
    @State private var searchText = ""
    
    var highRiskRoads: [RoadSegment] {
        riskService.getHighRiskRoads()
    }
    
    var filteredRoads: [RoadSegment] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return highRiskRoads }
        return highRiskRoads.filter { segment in
            segment.linearName.lowercased().contains(query)
                || (segment.segmentLocation?.lowercased().contains(query) ?? false)
        }
    }
    
    var body: some View {
        NavigationView {
            List {
                if filteredRoads.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: searchText.isEmpty ? "map" : "magnifyingglass")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)
                        Text(searchText.isEmpty ? "No high-risk roads found" : "No matches for \"\(searchText)\"")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text(searchText.isEmpty ? "Load the map to see risk predictions" : "Try a different search term")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                } else {
                    ForEach(filteredRoads) { segment in
                        NavigationLink(destination: RiskDetailView(segment: segment)) {
                            RiskListRow(segment: segment)
                        }
                    }
                }
            }
            .navigationTitle("High Risk Roads")
            .searchable(text: $searchText, prompt: "Search road names")
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

            Text(segment.riskLevel.displayName)
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundColor(Color(hex: segment.riskLevel.color))
        }
        .padding(.vertical, 4)
    }
}




