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
                    
                    // crash statistics
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Crash Statistics")
                            .font(.headline)
                        
                        InfoRow(label: "Total Crashes", value: "\(segment.numTotalCrashes)")
                        InfoRow(label: "KSI Crashes", value: "\(segment.numKSICrashes)")
                        InfoRow(label: "Fatalities", value: "\(segment.fatalityCount)")
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // risk assessment
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Risk Assessment")
                            .font(.headline)
                        
                        Text("This road segment has been identified as \(segment.riskLevel.displayName.lowercased()) risk based on historical crash data and road characteristics.")
                            .font(.body)
                            .foregroundColor(.secondary)
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




