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
                    // Header: road name + location
                    VStack(alignment: .leading, spacing: 8) {
                        Text(segment.linearName)
                            .font(.title)
                            .fontWeight(.bold)

                        if let loc = segment.segmentLocation, !loc.isEmpty, !isRawCoordinates(loc) {
                            Text(loc)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }

                        HStack {
                            Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                                .foregroundColor(Color(hex: segment.riskLevel.color))
                                .font(.headline)

                            Spacer()

                            Text(confidenceDescription(segment.confidence))
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color(hex: segment.riskLevel.color).opacity(0.1))
                    .cornerRadius(12)

                    // Plain-language summary
                    VStack(alignment: .leading, spacing: 8) {
                        Text("What this means")
                            .font(.headline)

                        Text(plainLanguageSummary)
                            .font(.body)
                            .foregroundColor(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .lineSpacing(6)
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Road information
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Road Information")
                            .font(.headline)

                        InfoRow(label: "Road type", value: formatRoadClass(segment.roadClass))
                        InfoRow(label: "Length", value: formatSegmentLength(segment.segmentLength))
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)

                    // Contributing factors (human-readable)
                    if let drivers = segment.riskDrivers, !drivers.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Why this rating?")
                                .font(.headline)

                            Text("Our model considers these factors when estimating risk:")
                                .font(.subheadline)
                                .foregroundColor(.secondary)

                            ForEach(Array(drivers.sorted(by: { abs($0.value) > abs($1.value) }).prefix(5)), id: \.key) { item in
                                HStack(alignment: .top) {
                                    Text(formatDriverLabel(item.key))
                                        .foregroundColor(.secondary)
                                    Spacer()
                                    Text(formatDriverValue(key: item.key, value: item.value))
                                        .fontWeight(.medium)
                                        .multilineTextAlignment(.trailing)
                                }
                                .font(.subheadline)
                            }
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }

                    // Tips for the user
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Stay safe")
                            .font(.headline)

                        VStack(alignment: .leading, spacing: 8) {
                            TipRow(icon: "eye.fill", text: "Stay alert and scan for hazards")
                            TipRow(icon: "speedometer", text: "Drive at or below the speed limit")
                            TipRow(icon: "person.2.fill", text: "Watch for pedestrians and cyclists")
                            TipRow(icon: "iphone.slash", text: "Avoid distractions")
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

    private func isRawCoordinates(_ text: String) -> Bool {
        text.contains("(") && text.contains(")") && text.contains(",") && text.range(of: #"\d+\.\d+"#, options: .regularExpression) != nil
    }

    private func confidenceDescription(_ confidence: Double) -> String {
        let pct = Int(confidence * 100)
        switch pct {
        case 0..<40: return "\(pct)% confidence (low certainty)"
        case 40..<70: return "\(pct)% confidence (moderate certainty)"
        default: return "\(pct)% confidence (high certainty)"
        }
    }

    private var plainLanguageSummary: String {
        switch segment.riskLevel {
        case .high:
            return "This road has a higher predicted crash rate than most streets in the area. Historical data and road features (intersections, traffic patterns, time of day) suggest extra caution is warranted here."
        case .medium:
            return "This road has a moderate predicted crash rate. Some factors increase risk compared to safer streets. Drive with normal caution."
        case .low:
            return "This road has a lower predicted crash rate than most streets. It's still important to drive safely, but the model suggests relatively lower risk here."
        }
    }

    private func formatRoadClass(_ roadClass: String) -> String {
        let lower = roadClass.lowercased()
        switch lower {
        case "arterial": return "Major road (arterial)"
        case "collector": return "Collector road"
        case "local": return "Local street"
        case "minor_arterial": return "Minor arterial"
        default: return roadClass.capitalized
        }
    }

    private func formatSegmentLength(_ meters: Double) -> String {
        if meters < 100 { return "\(Int(meters))m (short)" }
        if meters < 300 { return "\(Int(meters))m (medium)" }
        return "\(Int(meters))m (long)"
    }
}

private struct TipRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 24, alignment: .center)
            Text(text)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

private func formatDriverLabel(_ key: String) -> String {
    let labels: [String: String] = [
        "crashes_1d_ago": "Recent crashes (24h)",
        "crashes_7d_ago": "Recent crashes (7 days)",
        "crashes_30d_ago": "Recent crashes (30 days)",
        "rolling_mean_7d": "7-day crash average",
        "rolling_max_7d": "7-day crash peak",
        "hist_crashes_per_year": "Historical crashes per year",
        "from_intersection_degree": "Intersection complexity (start)",
        "to_intersection_degree": "Intersection complexity (end)",
        "segment_length": "Segment length",
        "datetime_hour": "Time of day",
        "day_of_week": "Day of week",
        "is_weekend": "Weekend vs weekday",
        "month": "Season"
    ]
    return labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized
}

private func formatDriverValue(key: String, value: Double) -> String {
    if key.contains("ratio") && value <= 1 { return String(format: "%.0f%%", value * 100) }
    if key.contains("length") { return String(format: "%.0fm", value) }

    // Human-readable values for common factors
    switch key {
    case "from_intersection_degree", "to_intersection_degree":
        let n = Int(value)
        switch n {
        case 2: return "2-way (simple)"
        case 3: return "3-way (T-junction)"
        case 4: return "4-way intersection"
        case 5...: return "\(n)-way (complex)"
        default: return "\(n)-way"
        }
    case "is_weekend":
        return value >= 0.5 ? "Weekend" : "Weekday"
    case "month":
        let months = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        let idx = Int(value)
        guard idx >= 1, idx <= 12 else { return "Month \(idx)" }
        return months[idx]
    case "datetime_hour":
        let h = Int(value)
        if h >= 6 && h < 10 { return "Morning rush" }
        if h >= 16 && h < 19 { return "Evening rush" }
        if h >= 22 || h < 6 { return "Late night" }
        return "\(h):00"
    case "day_of_week":
        let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        let idx = Int(value) % 7
        return days.indices.contains(idx) ? days[idx] : "Day \(Int(value))"
    default:
        if value == floor(value) { return String(format: "%.0f", value) }
        return String(format: "%.2f", value)
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




