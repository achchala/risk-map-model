//
//  RiskListView.swift
//  RiskMapApp
//
//  list view showing road segments with risk filtering
//

import SwiftUI

enum RiskFilter: String, CaseIterable {
    case all = "All"
    case low = "Low Risk"
    case medium = "Medium Risk"
    case high = "High Risk"
    
    var riskLevel: RiskLevel? {
        switch self {
        case .all: return nil
        case .low: return .low
        case .medium: return .medium
        case .high: return .high
        }
    }
}

struct RiskListView: View {
    @EnvironmentObject var riskService: RiskService
    @State private var searchText = ""
    @State private var riskFilter: RiskFilter = .all
    
    /// Segments for list: prefer full load (allSegmentsForList), fall back to map data.
    var allRoads: [RoadSegment] {
        !riskService.allSegmentsForList.isEmpty
            ? riskService.allSegmentsForList
            : riskService.roadSegments
    }
    
    var riskFilteredRoads: [RoadSegment] {
        let roads: [RoadSegment]
        if let level = riskFilter.riskLevel {
            roads = allRoads.filter { $0.riskLevel == level }
        } else {
            roads = allRoads
        }
        // Sort: high first, then medium, then low; within each group by name
        let priority: [RiskLevel: Int] = [.high: 0, .medium: 1, .low: 2]
        return roads.sorted { a, b in
            let pa = priority[a.riskLevel] ?? 3
            let pb = priority[b.riskLevel] ?? 3
            if pa != pb { return pa < pb }
            return a.linearName.localizedCaseInsensitiveCompare(b.linearName) == .orderedAscending
        }
    }
    
    var filteredRoads: [RoadSegment] {
        let roads = riskFilteredRoads
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return roads }
        return roads.filter { segment in
            segment.linearName.lowercased().contains(query)
                || (segment.segmentLocation?.lowercased().contains(query) ?? false)
        }
    }
    
    var body: some View {
        NavigationView {
            List {
                Section {
                    Picker("Filter by risk", selection: $riskFilter) {
                        ForEach(RiskFilter.allCases, id: \.self) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                    .pickerStyle(.menu)
                }
                if filteredRoads.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: searchText.isEmpty ? "map" : "magnifyingglass")
                            .font(.system(size: 50))
                            .foregroundColor(.secondary)
                        Text(emptyStateTitle)
                            .font(.headline)
                            .foregroundColor(.secondary)
                        Text(emptyStateSubtitle)
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
            .navigationTitle("Road Details")
            .searchable(text: $searchText, prompt: "Search road names")
            .refreshable {
                try? await riskService.fetchAllRiskPredictionsForList()
            }
            .onAppear {
                Task {
                    guard riskService.allSegmentsForList.isEmpty else { return }
                    try? await riskService.fetchAllRiskPredictionsForList()
                }
            }
        }
        .overlay {
            if riskService.isLoadingList {
                ProgressView("Loading all road segments...")
                    .tint(.primary)
                    .padding()
                    .background(.regularMaterial)
                    .cornerRadius(10)
            }
        }
    }
    
    private var emptyStateTitle: String {
        if !searchText.isEmpty { return "No matches for \"\(searchText)\"" }
        if riskFilter != .all { return "No \(riskFilter.rawValue) segments" }
        return "No road segments found"
    }
    
    private var emptyStateSubtitle: String {
        if !searchText.isEmpty { return "Try a different search term" }
        return "Load the map to see risk predictions"
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




