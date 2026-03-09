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
                    // Header with name and risk level
                    VStack(alignment: .leading, spacing: 8) {
                        Text(segment.linearName)
                            .font(.title)
                            .fontWeight(.bold)
                        
                        Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                            .foregroundColor(Color(hex: segment.riskLevel.color))
                            .font(.headline)
                    }
                    .padding()
                    .background(Color(hex: segment.riskLevel.color).opacity(0.1))
                    .cornerRadius(12)
                    
                    // Explanation
                    if let explanation = segment.riskExplanation, !explanation.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Why This Risk Level?")
                                .font(.headline)
                            
                            Text(explanation)
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                    
                    // Road information
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Road Information")
                            .font(.headline)
                        
                        InfoRow(label: "Road Class", value: segment.roadClass)
                        InfoRow(label: "Segment Length", value: "\(Int(segment.segmentLength))m")
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Recent activity (crashes in latest model window)
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Recent Activity")
                            .font(.headline)
                        
                        InfoRow(label: "Crashes (latest period)", value: "\(segment.numTotalCrashes)")
                        if segment.numKSICrashes > 0 {
                            InfoRow(label: "KSI Crashes", value: "\(segment.numKSICrashes)")
                        }
                        if segment.fatalityCount > 0 {
                            InfoRow(label: "Fatalities", value: "\(segment.fatalityCount)")
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




