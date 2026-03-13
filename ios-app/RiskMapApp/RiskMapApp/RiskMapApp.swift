//
//  RiskMapApp.swift
//  RiskMapApp
//
//  StreetSmart — safer routes in Toronto
//

import SwiftUI

@main
struct RiskMapApp: App {
    @State private var showLanding = true

    var body: some Scene {
        WindowGroup {
            if showLanding {
                LandingView(onContinue: { showLanding = false })
            } else {
                ContentView()
            }
        }
    }
}




