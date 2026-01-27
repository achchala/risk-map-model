//
//  SegmentDetailPopupView.swift
//  RiskMapApp
//
//  Bottom sheet popup showing quick road segment safety details
//

import SwiftUI
import MapKit

struct SegmentDetailPopupView: View {
    let segment: RoadSegment
    let onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Drag handle for swipe-down gesture discoverability
            Capsule()
                .fill(Color.secondary.opacity(0.3))
                .frame(width: 40, height: 5)
                .padding(.top, 8)

            // Header with title and close button
            HStack {
                Text("Road Safety Details")
                    .font(.headline)
                    .foregroundColor(.primary)

                Spacer()

                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .accessibilityLabel("Close")
            }
            .padding(.horizontal)
            .padding(.top, 12)
            .padding(.bottom, 8)

            Divider()
                .padding(.horizontal)

            // Scrollable content
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Road name with risk badge
                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(segment.linearName)
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.primary)

                            Text("\(Int(segment.segmentLength))m segment")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        // Risk level badge
                        Label(segment.riskLevel.displayName, systemImage: segment.riskLevel.systemImage)
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color(hex: segment.riskLevel.color))
                            .cornerRadius(8)
                    }

                    // Safety score with stars
                    HStack(spacing: 8) {
                        Text("Safety Score:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)

                        Text("\(segment.safetyScoreStars)")
                            .font(.body)

                        Text(String(format: "%.1f", segment.safetyScore))
                            .font(.headline)
                            .fontWeight(.bold)
                            .foregroundColor(.primary)

                        Text("/ 5")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityLabel("Safety score \(String(format: "%.1f", segment.safetyScore)) out of 5")

                    // Safety summary
                    Text(segment.safetySummary)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(Color(.systemGray6))
                        .cornerRadius(8)

                    Divider()

                    // Risk Factors section
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Risk Factors")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.primary)

                        // Show crash information if available
                        if segment.numTotalCrashes > 0 {
                            InfoRow(
                                label: "Total crashes",
                                value: "\(segment.numTotalCrashes)"
                            )
                        }

                        if segment.numKSICrashes > 0 {
                            InfoRow(
                                label: "Serious injuries (KSI)",
                                value: "\(segment.numKSICrashes)"
                            )
                        }

                        if segment.fatalityCount > 0 {
                            InfoRow(
                                label: "Fatalities",
                                value: "\(segment.fatalityCount)"
                            )
                        }

                        // Road characteristics
                        if !segment.roadClass.isEmpty {
                            InfoRow(
                                label: "Road class",
                                value: segment.roadClass
                            )
                        }

                        // Prediction confidence
                        InfoRow(
                            label: "Confidence",
                            value: "\(Int(segment.confidence * 100))%"
                        )
                    }

                    // Spacer to ensure content doesn't get cut off
                    Color.clear.frame(height: 8)
                }
                .padding(.horizontal)
                .padding(.top, 8)
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: UIScreen.main.bounds.height * 0.4) // 40% screen height
        .background(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.2), radius: 10, x: 0, y: -2)
        )
        .gesture(
            DragGesture()
                .onEnded { value in
                    // Dismiss if user swipes down more than 50 points
                    if value.translation.height > 50 {
                        onDismiss()
                    }
                }
        )
    }
}

// MARK: - Preview
#Preview {
    ZStack {
        Color.gray.opacity(0.3).ignoresSafeArea()

        VStack {
            Spacer()

            SegmentDetailPopupView(
                segment: RoadSegment(
                    id: "preview-1",
                    linearName: "Queen Street East",
                    roadClass: "Arterial",
                    segmentLength: 250,
                    riskLevel: .high,
                    confidence: 0.85,
                    numTotalCrashes: 12,
                    numKSICrashes: 3,
                    fatalityCount: 1,
                    coordinates: [
                        RoadSegment.Coordinate(latitude: 43.6532, longitude: -79.3832)
                    ]
                ),
                onDismiss: {}
            )
        }
    }
}
