//
//  SettingsView.swift
//  RiskMapApp
//
//  settings and information view
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("autoRefresh") private var autoRefresh = true
    @AppStorage("mapStyle") private var mapStyleRaw = "standard"
    @AppStorage("defaultRoutePreference") private var defaultRoutePreference = "safest"
    @AppStorage("distanceUnits") private var distanceUnits = "km"

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Map")) {
                    Toggle("Auto Refresh on Pan/Zoom", isOn: $autoRefresh)
                    Picker("Map Style", selection: $mapStyleRaw) {
                        Text("Standard").tag("standard")
                        Text("Satellite").tag("satellite")
                        Text("Hybrid").tag("hybrid")
                    }
                }

                Section(header: Text("Navigation")) {
                    Picker("Default Route", selection: $defaultRoutePreference) {
                        Text("Safest").tag("safest")
                        Text("Fastest").tag("fastest")
                    }
                    Picker("Distance Units", selection: $distanceUnits) {
                        Text("Kilometers").tag("km")
                        Text("Miles").tag("mi")
                    }
                }

                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                }

                Section(header: Text("Data Sources")) {
                    LabeledContent("Crash Data", value: "Toronto Police Open Data")
                    LabeledContent("KSI (Severe Crashes)", value: "Toronto Police Open Data")
                    LabeledContent("Road Network", value: "Toronto Open Data")
                    LabeledContent("Live Weather", value: "WeatherAPI.com")
                    LabeledContent("Historical Weather", value: "NOAA Climate Data")
                    LabeledContent("Routing & Maps", value: "Apple MapKit")
                }

                Section(header: Text("Disclaimer")) {
                    Text("Risk predictions use historical crash data and should not be the sole factor in route planning. Always drive safely.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Settings")
        }
    }
}




