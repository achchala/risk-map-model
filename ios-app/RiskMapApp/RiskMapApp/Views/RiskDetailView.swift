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
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Why This Risk Level?")
                            .font(.headline)

                        // Metrics row: confidence
                        if segment.confidence > 0 {
                            Label("\(Int(segment.confidence * 100))% confidence", systemImage: "checkmark.seal")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }

                        // Weather/time multipliers when they affect risk
                        if let wm = segment.weatherMult, let tm = segment.timeMult, (wm != 1.0 || tm != 1.0) {
                            HStack(spacing: 12) {
                                if wm != 1.0 {
                                    Text("Weather: \(wm > 1 ? "+" : "")\(Int(round((wm - 1) * 100)))%")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                if tm != 1.0 {
                                    Text("Time of day: \(tm > 1 ? "+" : "")\(Int(round((tm - 1) * 100)))%")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding(.vertical, 4)
                        }

                        // Probability breakdown
                        if let probs = segment.probabilities {
                            HStack(spacing: 12) {
                                Text("Probabilities:")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("Low \(Int(probs.low * 100))%")
                                    .font(.caption)
                                    .foregroundColor(.green)
                                Text("Medium \(Int(probs.medium * 100))%")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                                Text("High \(Int(probs.high * 100))%")
                                    .font(.caption)
                                    .foregroundColor(.red)
                            }
                            .padding(.vertical, 4)
                        }

                        if let explanation = segment.riskExplanation, !explanation.isEmpty {
                            Text(explanation)
                                .font(.body)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
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




