//
//  HomeView.swift
//  RiskMapApp
//
//  Home tab: app intro, "What is risk?" explanation, quick actions
//

import SwiftUI

struct HomeView: View {
    @StateObject private var routeService = RouteService()
    @Binding var selectedTab: Int
    @State private var riskDefinition: RiskDefinitionResponse?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    headerSection
                    whatIsRiskSection
                    quickActionsSection
                }
                .padding()
            }
            .background(Color(UIColor.systemGroupedBackground))
            .navigationTitle("Home")
            .navigationBarTitleDisplayMode(.large)
            .onAppear { loadRiskDefinition() }
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("StreetSmart")
                .font(.title)
                .fontWeight(.bold)
            Text("See Risk Before It Happens")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.secondary)
            Text("AI-Powered Risk-Aware Navigation")
                .font(.caption)
                .foregroundColor(.secondary.opacity(0.9))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
    }

    private var whatIsRiskSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What is a risky road?")
                .font(.headline)
            if let def = riskDefinition {
                Text(def.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                VStack(alignment: .leading, spacing: 6) {
                    Label(def.low, systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                    Label(def.medium, systemImage: "exclamationmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.orange)
                    Label(def.high, systemImage: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.red)
                }
                .padding(.top, 4)
            } else {
                Text("Risk is based on predicted crash rate (λ) per segment. Low = bottom 70% of segments, Medium = 70th–90th percentile, High = top 10%. Thresholds come from the current model.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Text("Tap to load latest thresholds from server.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
    }

    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick actions")
                .font(.headline)
            HStack(spacing: 12) {
                Button(action: { selectedTab = 1 }) {
                    Label("View map", systemImage: "map")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue.opacity(0.15))
                        .foregroundColor(.blue)
                        .cornerRadius(10)
                }
                .buttonStyle(.plain)
                Button(action: { selectedTab = 3 }) {
                    Label("Plan route", systemImage: "location.north.circle")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green.opacity(0.15))
                        .foregroundColor(.green)
                        .cornerRadius(10)
                }
                .buttonStyle(.plain)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(UIColor.systemBackground))
        .cornerRadius(12)
    }

    private func loadRiskDefinition() {
        Task {
            if let def = try? await routeService.fetchRiskDefinition() {
                await MainActor.run { riskDefinition = def }
            }
        }
    }
}
