//
//  ContentView.swift
//  RiskMapApp
//
//  main content view with tab navigation
//

import SwiftUI

struct ContentView: View {
    @StateObject private var riskService = RiskService()
    @StateObject private var weatherService = WeatherService()

    var body: some View {
        TabView {
            MapView()
                .tabItem {
                    Label("Map", systemImage: "map")
                }
            
            RiskListView()
                .tabItem {
                    Label("High Risk", systemImage: "exclamationmark.triangle")
                }
            
            RouteNavigationViewWrapper()
                .tabItem {
                    Label("Navigation", systemImage: "location.north.circle")
                }
            
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
        .environmentObject(riskService)
        .environmentObject(weatherService)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}






