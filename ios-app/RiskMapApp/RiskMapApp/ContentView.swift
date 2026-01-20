//
//  ContentView.swift
//  RiskMapApp
//
//  main content view with tab navigation
//

import SwiftUI

struct ContentView: View {
    @StateObject private var riskService = RiskService()
    
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
            
            RouteNavigationView(riskService: riskService)
                .tabItem {
                    Label("Navigation", systemImage: "location.north.circle")
                }
            
            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
        .environmentObject(riskService)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}






