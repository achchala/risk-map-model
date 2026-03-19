//
//  SettingsView.swift
//  RiskMapApp
//
//  settings and information view
//

import SwiftUI

/// User preference for balancing safety vs speed when choosing the "safer" route.
/// Slider value 0 = speed, 1 = safety. Stored as Double in "safetySpeedBalanceSlider"
enum SafetySpeedBalance: String, CaseIterable {
    case speed = "speed"
    case balanced = "balanced"
    case safety = "safety"

    init(sliderValue: Double) {
        switch sliderValue {
        case ..<0.35: self = .speed
        case 0.35..<0.65: self = .balanced
        default: self = .safety
        }
    }

    var sliderValue: Double {
        switch self {
        case .speed: return 0.0
        case .balanced: return 0.5
        case .safety: return 1.0
        }
    }

    var displayName: String {
        switch self {
        case .speed: return "Prioritize speed"
        case .balanced: return "Balanced"
        case .safety: return "Prioritize safety"
        }
    }

    var subtitle: String {
        switch self {
        case .speed: return "Minimize travel time, accept some risk"
        case .balanced: return "Balance safety and speed"
        case .safety: return "Avoid high-risk roads, may add time"
        }
    }

    /// Beta for backend routing: higher = more weight on avoiding risk.
    /// Scaled so risk term meaningfully affects path choice (λ ~ 1e-5 to 6e-3).
    var beta: Double {
        switch self {
        case .speed: return 1.0
        case .balanced: return 5.0
        case .safety: return 25.0
        }
    }

    /// Time penalty factor for MapKit alternate selection: higher = penalize longer routes more
    var timePenaltyFactor: Double {
        switch self {
        case .speed: return 3.0
        case .balanced: return 1.0
        case .safety: return 0.2
        }
    }

    /// Interpolate beta from slider value (0...1). Range 1.0 (speed) to 25.0 (safety).
    static func betaFromSlider(_ value: Double) -> Double {
        let v = max(0, min(1, value))
        return 1.0 + v * 24.0
    }

    /// Interpolate time penalty from slider value (0...1)
    static func timePenaltyFromSlider(_ value: Double) -> Double {
        let v = max(0, min(1, value))
        return 3.0 - v * 2.8 // 3.0 (speed) to 0.2 (safety)
    }
}

struct SettingsView: View {
    @AppStorage("notificationsEnabled") private var notificationsEnabled = false
    @AppStorage("autoRefresh") private var autoRefresh = true
    @AppStorage("safetySpeedBalanceSlider") private var safetySpeedSlider = 0.5

    private var currentBalance: SafetySpeedBalance {
        SafetySpeedBalance(sliderValue: safetySpeedSlider)
    }

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Route Planning"), footer: Text("Controls how the \"safer\" route balances risk avoidance with travel time. Slide left for faster routes; slide right to avoid high-risk roads.")) {
                    HStack {
                        Text("Speed")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Slider(value: $safetySpeedSlider, in: 0...1)
                        Text("Safety")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(currentBalance.subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Preferences")) {
                    Toggle("Enable Notifications", isOn: $notificationsEnabled)
                    Toggle("Auto Refresh Map", isOn: $autoRefresh)
                }
                
                Section(header: Text("Data Sources")) {
                    Link("Traffic Collisions — Toronto Police Service", destination: URL(string: "https://data.torontopolice.on.ca/datasets/TorontoPS::traffic-collisions-open-data-asr-t-tbl-001/about")!)
                        .font(.subheadline)
                    Text("Motor vehicle collision records")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Link("KSI Data — Toronto Police Service", destination: URL(string: "https://data.torontopolice.on.ca/")!)
                        .font(.subheadline)
                    Text("Killed or Seriously Injured (KSI) collision records")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Link("Road Network — City of Toronto Open Data", destination: URL(string: "https://open.toronto.ca/")!)
                        .font(.subheadline)
                    Text("Toronto centreline geometry and segment attributes")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("Traffic Volumes — Toronto/MTO")
                        .font(.subheadline)
                    Text("Average daily traffic and speed (model dataset)")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("Historical Weather — NOAA GHCN")
                        .font(.subheadline)
                    Text("Historical conditions for model training")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Link("Live Weather — WeatherAPI.com", destination: URL(string: "https://www.weatherapi.com/")!)
                        .font(.subheadline)
                    Text("Current conditions for risk adjustment")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text("Maps & Routing — Apple MapKit")
                        .font(.subheadline)
                    Text("Map display and turn-by-turn directions")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Section(header: Text("Disclaimer")) {
                    Text("Risk predictions are based on historical data and should not be the sole factor in route planning. Always exercise caution while driving.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Settings")
        }
    }
}




