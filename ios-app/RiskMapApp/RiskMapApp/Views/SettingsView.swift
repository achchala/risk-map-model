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
    var beta: Double {
        switch self {
        case .speed: return 0.25
        case .balanced: return 1.5
        case .safety: return 8.0
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

    /// Interpolate beta from slider value (0...1). Range 0.25 (speed) to 8.0 (safety).
    static func betaFromSlider(_ value: Double) -> Double {
        let v = max(0, min(1, value))
        return 0.25 + v * 7.75
    }

    /// Interpolate time penalty from slider value (0...1)
    static func timePenaltyFromSlider(_ value: Double) -> Double {
        let v = max(0, min(1, value))
        return 3.0 - v * 2.8 // 3.0 (speed) to 0.2 (safety)
    }
}

struct SettingsView: View {
    @EnvironmentObject var riskService: RiskService
    @AppStorage("notificationsEnabled") private var notificationsEnabled = false
    @AppStorage("autoRefresh") private var autoRefresh = true
    @AppStorage("safetySpeedBalanceSlider") private var safetySpeedSlider = 0.5
    @AppStorage("backendAPIURL") private var backendAPIURL = ""
    @State private var connectionTestMessage: String?
    @State private var isTestingConnection = false

    private var currentBalance: SafetySpeedBalance {
        SafetySpeedBalance(sliderValue: safetySpeedSlider)
    }

    private static let riskAversionSteps: [Double] = [0, 0.5, 1.0]

    private var riskAversionBinding: Binding<Double> {
        Binding(
            get: { safetySpeedSlider },
            set: { newValue in
                let nearest = Self.riskAversionSteps.min(by: { abs($0 - newValue) < abs($1 - newValue) }) ?? 0.5
                safetySpeedSlider = nearest
            }
        )
    }

    private func runConnectionTest() async {
        connectionTestMessage = nil
        isTestingConnection = true
        let (_, msg) = await riskService.testConnection()
        await MainActor.run {
            isTestingConnection = false
            connectionTestMessage = msg
        }
    }

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Backend API"), footer: Text("When testing on a physical device, enter your Mac's IP (e.g. http://192.168.1.100:8000) or ngrok URL (e.g. https://xxxx.ngrok-free.app). For ngrok, use the exact URL from the ngrok terminal. Mac and iPhone must be on same Wi‑Fi unless using ngrok.")) {
                    TextField("e.g. http://192.168.1.100:8000", text: $backendAPIURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    Button(action: {
                        Task { await runConnectionTest() }
                    }) {
                        HStack {
                            if isTestingConnection {
                                ProgressView()
                                Text("Testing...")
                            } else {
                                Text("Test Connection")
                            }
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .disabled(isTestingConnection || backendAPIURL.trimmingCharacters(in: .whitespaces).isEmpty)
                    if let msg = connectionTestMessage {
                        Text(msg)
                            .font(.caption)
                            .foregroundColor(msg.contains("success") ? .green : .red)
                    }
                }

                Section(header: Text("How risk averse are you?"), footer: Text("Controls how the \"safer\" route balances risk avoidance with travel time. Slide left for faster routes; slide right to avoid high-risk roads.")) {
                    HStack {
                        Text("Less")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Slider(value: riskAversionBinding, in: 0...1)
                        Text("More")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(currentBalance.subtitle)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .onAppear {
                    let nearest = Self.riskAversionSteps.min(by: { abs($0 - safetySpeedSlider) < abs($1 - safetySpeedSlider) }) ?? 0.5
                    if abs(safetySpeedSlider - nearest) > 0.001 { safetySpeedSlider = nearest }
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




