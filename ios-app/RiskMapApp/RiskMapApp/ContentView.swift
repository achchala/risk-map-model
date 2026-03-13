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
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            HomeView(selectedTab: $selectedTab)
                .tabItem {
                    Label("Home", systemImage: "house")
                }
                .tag(0)

            MapView()
                .tabItem {
                    Label("Map", systemImage: "map")
                }
                .tag(1)

            RiskListView()
                .tabItem {
                    Label("High Risk", systemImage: "exclamationmark.triangle")
                }
                .tag(2)

            RouteNavigationView()
                .tabItem {
                    Label("Navigation", systemImage: "location.north.circle")
                }
                .tag(3)

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(4)
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






