//
//  RiskMapApp.swift
//  RiskMapApp
//
//  toronto road risk prediction!
//

import SwiftUI

@main
struct RiskMapApp: App {
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    
    var body: some Scene {
        WindowGroup {
            if hasCompletedOnboarding {
                ContentView()
            } else {
                OnboardingView(hasCompletedOnboarding: $hasCompletedOnboarding)
            }
        }
    }
}




