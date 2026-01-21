//
//  OnboardingView.swift
//  RiskMapApp
//
//  welcome and onboarding experience for first-time users
//

import SwiftUI

struct OnboardingView: View {
    @Binding var hasCompletedOnboarding: Bool
    @State private var currentPage = 0
    
    var body: some View {
        TabView(selection: $currentPage) {
            OnboardingPage(
                title: "welcome to streetsmart",
                description: "plan safer routes by avoiding high-risk road segments based on historical crash data!",
                image: "map.fill",
                color: .blue
            )
            .tag(0)
            
            // Page 2: Map Features
            OnboardingPage(
                title: "visualize risk",
                description: "see road segments color-coded by risk level. green for low risk, yellow for medium, and red for high risk.",
                image: "map.circle.fill",
                color: .green
            )
            .tag(1)
            
            // Page 3: Route Planning
            OnboardingPage(
                title: "plan safer routes",
                description: "compare fastest and safest routes. choose the route that balances safety with travel time.",
                image: "location.north.circle.fill",
                color: .orange
            )
            .tag(2)
            
            // Page 4: Get Started
            OnboardingPage(
                title: "ready to get started?",
                description: "start planning safer routes and make your commute safer. tap get started to begin.",
                image: "checkmark.circle.fill",
                color: .green
            )
            .tag(3)
        }
        .tabViewStyle(.page)
        .indexViewStyle(.page(backgroundDisplayMode: .always))
        .overlay(
            VStack {
                Spacer()
                if currentPage == 3 {
                    Button(action: {
                        hasCompletedOnboarding = true
                    }) {
                        Text("Get Started")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                    .padding()
                } else {
                    Button(action: {
                        withAnimation {
                            currentPage += 1
                        }
                    }) {
                        Text("Next")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                    }
                    .padding()
                }
            }
        )
    }
}

struct OnboardingPage: View {
    let title: String
    let description: String
    let image: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 32) {
            Spacer()
            
            Image(systemName: image)
                .font(.system(size: 80))
                .foregroundColor(color)
                .padding()
            
            VStack(spacing: 16) {
                Text(title)
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                Text(description)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
            
            Spacer()
        }
    }
}
