//
//  SettingsView.swift
//  RiskMapApp
//
//  settings and information view
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("notificationsEnabled") private var notificationsEnabled = false
    @AppStorage("autoRefresh") private var autoRefresh = true
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Preferences")) {
                    Toggle("Enable Notifications", isOn: $notificationsEnabled)
                    Toggle("Auto Refresh Map", isOn: $autoRefresh)
                }
                
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    Link("Privacy Policy", destination: URL(string: "https://your-domain.com/privacy")!)
                    Link("Terms of Service", destination: URL(string: "https://your-domain.com/terms")!)
                }
                
                Section(header: Text("Data Source")) {
                    Text("Crash data provided by Toronto Police Open Data")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("Road network data from Toronto Open Data Portal")
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




