//
//  RiskDetailView.swift
//  RiskMapApp
//
//  detail view for a road segment showing risk information
//

import SwiftUI

struct RiskDetailView: View {
    let segment: RoadSegment
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(segment.linearName)
                            .font(.title)
                            .fontWeight(.bold)
                        
                        if let loc = segment.segmentLocation, !loc.isEmpty {
                            Text(loc)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        HStack {
                            Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                                .foregroundColor(Color(hex: segment.riskLevel.color))
                                .font(.headline)
                            
                            Spacer()
                            
                            Text("\(Int(segment.confidence * 100))% confidence")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(hex: segment.riskLevel.color).opacity(0.1))
                    .cornerRadius(12)
                    
                    // road information
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Road Information")
                            .font(.headline)
                        
                        InfoRow(label: "Road Class", value: segment.roadClass)
                        InfoRow(label: "Segment Length", value: "\(Int(segment.segmentLength))m")
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // risk assessment & contributing factors
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Why this risk level?")
                            .font(.headline)
                        
                        if let explanation = segment.riskExplanation, !explanation.isEmpty {
                            Text(explanation)
                                .font(.body)
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .lineSpacing(6)
                        } else {
                            Text("This road segment has been identified as \(segment.riskLevel.displayName.lowercased()) risk based on historical crash data and road characteristics.")
                                .font(.body)
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .lineSpacing(6)
                        }
                        
                        if let drivers = segment.riskDrivers, !drivers.isEmpty {
                            Text("Contributing factors")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .padding(.top, 4)
                            ForEach(Array(drivers.sorted(by: { abs($0.value) > abs($1.value) }).prefix(5)), id: \.key) { item in
                                HStack {
                                    Text(formatDriverLabel(item.key))
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    Text(formatDriverValue(key: item.key, value: item.value))
                                        .fontWeight(.medium)
                                }
                                .font(.caption)
                            }
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                }
                .padding()
            }
            .navigationTitle("Road Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

private func formatDriverLabel(_ key: String) -> String {
    let labels: [String: String] = [
        "crashes_1d_ago": "Crashes (24h)",
        "crashes_7d_ago": "Crashes (7d)",
        "crashes_30d_ago": "Crashes (30d)",
        "rolling_mean_7d": "7d rolling avg",
        "rolling_max_7d": "7d rolling peak",
        "hist_crashes_per_year": "Historical crashes/yr",
        "from_intersection_degree": "Intersection (from)",
        "to_intersection_degree": "Intersection (to)",
        "segment_length": "Segment length",
        "datetime_hour": "Hour of day",
        "day_of_week": "Day of week",
        "is_weekend": "Weekend",
        "month": "Month"
    ]
    return labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized
}

private func formatDriverValue(key: String, value: Double) -> String {
    if key.contains("ratio") && value <= 1 { return String(format: "%.0f%%", value * 100) }
    if key.contains("length") { return String(format: "%.0fm", value) }
    if value == floor(value) { return String(format: "%.0f", value) }
    return String(format: "%.2f", value)
}

struct InfoRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
    }
}




