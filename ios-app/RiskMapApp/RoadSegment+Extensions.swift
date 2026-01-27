//
//  RoadSegment+Extensions.swift
//  RiskMapApp
//
//  Extensions for RoadSegment to calculate safety scores and explanations
//

import Foundation

extension RoadSegment {
    /// Calculate safety score from 0-5 based on risk level, confidence, and crash data
    /// Higher score = safer road
    /// Formula: baseScore (inverted from risk) + confidence modifier - crash penalty
    var safetyScore: Double {
        // Base score (inverted: low risk = high safety score)
        let baseScore: Double = switch riskLevel {
        case .low: 4.0
        case .medium: 2.5
        case .high: 1.0
        }

        // Confidence modifier: ±0.25 range based on prediction confidence
        // Higher confidence = more certain of the score
        let confidenceModifier = (confidence - 0.5) * 0.5

        // Crash penalty: each KSI crash reduces score by 0.1, capped at -0.5
        // Prioritizes serious crashes (KSI) over total crashes
        let crashPenalty = min(Double(numKSICrashes) * 0.1, 0.5)

        // Calculate final score and clamp to [0, 5] range
        let finalScore = baseScore + confidenceModifier - crashPenalty
        return max(0, min(5, finalScore))
    }

    /// Star emoji representation of safety score (⭐ to ⭐⭐⭐⭐⭐)
    /// Visual representation for quick understanding
    var safetyScoreStars: String {
        let fullStars = Int(safetyScore.rounded())
        let starCount = max(1, fullStars) // Minimum 1 star
        return String(repeating: "⭐", count: starCount)
    }

    /// Human-readable explanation of risk factors
    /// Returns a summary of crash data and road characteristics
    var riskExplanation: String {
        var components: [String] = []

        // Crash information
        if numTotalCrashes > 0 {
            components.append("\(numTotalCrashes) crash\(numTotalCrashes > 1 ? "es" : "") recorded")
        }

        if numKSICrashes > 0 {
            components.append("\(numKSICrashes) serious \(numKSICrashes > 1 ? "injuries" : "injury") (KSI)")
        }

        if fatalityCount > 0 {
            components.append("\(fatalityCount) \(fatalityCount > 1 ? "fatalities" : "fatality")")
        }

        // Road characteristics
        if !roadClass.isEmpty && roadClass != "Unknown" {
            components.append("\(roadClass) road")
        }

        // Confidence level
        let confidencePercent = Int(confidence * 100)
        components.append("\(confidencePercent)% prediction confidence")

        // Join with commas and add period
        if components.isEmpty {
            return "No significant risk factors identified."
        }

        return components.joined(separator: ", ") + "."
    }

    /// Brief safety summary for display in popup
    /// Returns a concise 1-2 sentence explanation
    var safetySummary: String {
        switch riskLevel {
        case .high:
            if numKSICrashes > 2 {
                return "This road has a history of serious crashes. Exercise extra caution when traveling here."
            } else {
                return "This road segment is classified as high risk based on historical crash data."
            }

        case .medium:
            if numTotalCrashes > 5 {
                return "This road has experienced several crashes. Drive carefully and stay alert."
            } else {
                return "This road segment has moderate risk. Normal caution is advised."
            }

        case .low:
            if safetyScore > 4.0 {
                return "This is a very safe road segment with minimal crash history."
            } else {
                return "This road segment is relatively safe with low risk."
            }
        }
    }
}
