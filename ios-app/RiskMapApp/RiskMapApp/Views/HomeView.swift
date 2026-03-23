//
//  HomeView.swift
//  RiskMapApp
//
//  Home tab: app intro, "What is risk?" explanation, quick actions
//

import SwiftUI

struct HomeView: View {
    @EnvironmentObject var riskService: RiskService
    @Binding var selectedTab: Int
    @State private var riskDefinition: RiskDefinitionResponse?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 28) {
                    headerSection
                    whatIsRiskSection
                    quickActionsSection
                }
                .padding(20)
                .padding(.bottom, 32)
            }
            .background(
                ZStack {
                    Image("mapoverlay")
                        .resizable()
                        .scaledToFill()
                        .frame(width: 1000)
                        .rotationEffect(.degrees(90))
                        .opacity(0.15)
                        .ignoresSafeArea()
                    LinearGradient(
                        colors: [
                            Color(red: 0.92, green: 0.95, blue: 1.0).opacity(0.85),
                            Color(UIColor.systemGroupedBackground)
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                }
            )
            .navigationTitle("Home")
            .navigationBarTitleDisplayMode(.large)
            .onAppear { loadRiskDefinition() }
        }
    }

    private var headerSection: some View {
        VStack(spacing: 0) {
            ZStack {
                RoundedRectangle(cornerRadius: 20)
                    .fill(
                        LinearGradient(
                            colors: [Color.brandPrimary, Color.brandPrimary.opacity(0.85)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(height: 140)
                    .shadow(color: Color.brandPrimary.opacity(0.3), radius: 12, y: 6)
                VStack(spacing: 10) {
                    Image("streetsmart_white")
                        .resizable()
                        .scaledToFit()
                        .frame(height: 48)
                    Text("StreetSmart")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                    Text("See risk before it happens")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white.opacity(0.95))
                    Text("Toronto")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.8))
                }
            }
            .cornerRadius(20)
        }
    }

    private var whatIsRiskSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Image(systemName: "shield.checkered")
                    .font(.title3)
                    .foregroundColor(.brandPrimary)
                Text("What is a risky road?")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
            }
            if let def = riskDefinition {
                Text("Risk is based on predicted crash rate (λ) per road segment. Each segment is ranked by its predicted crash likelihood.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineSpacing(4)
                VStack(alignment: .leading, spacing: 10) {
                    riskLevelRow(icon: "checkmark.circle.fill", color: Color.brandSecondary, text: def.low)
                    riskLevelRow(icon: "exclamationmark.circle.fill", color: Color.orange, text: def.medium)
                    riskLevelRow(icon: "xmark.circle.fill", color: Color.red, text: def.high)
                }
                .padding(.top, 4)
            } else {
                Text("Risk is based on predicted crash rate (λ) per road segment. Each segment is ranked by its predicted crash likelihood.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineSpacing(4)
                Text("Loading latest thresholds…")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(UIColor.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.06), radius: 10, y: 4)
    }

    private func riskLevelRow(icon: String, color: Color, text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.body)
                .foregroundColor(color)
            Text(text)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var quickActionsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Image(systemName: "bolt.fill")
                    .font(.title3)
                    .foregroundColor(.brandTertiary)
                Text("Quick actions")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.primary)
            }
            HStack(spacing: 14) {
                quickActionButton(
                    title: "View map",
                    icon: "map.fill",
                    color: .brandPrimary
                ) { selectedTab = 1 }
                quickActionButton(
                    title: "Plan route",
                    icon: "arrow.triangle.turn.up.right.diamond.fill",
                    color: .brandSecondary
                ) { selectedTab = 3 }
            }
            tipSection
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(UIColor.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.06), radius: 10, y: 4)
    }

    private var tipSection: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "slider.horizontal.3")
                .font(.caption)
                .foregroundColor(.brandTertiary)
            Text("To customize your experience based on your personal risk tolerance, go to \"How Risk Averse Are You?\" on the Settings page.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.brandTertiary.opacity(0.08))
        .cornerRadius(10)
        .padding(.top, 4)
    }

    private func quickActionButton(title: String, icon: String, color: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 26))
                    .foregroundColor(color)
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(color)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 20)
            .background(color.opacity(0.12))
            .cornerRadius(14)
        }
        .buttonStyle(.plain)
    }

    private func loadRiskDefinition() {
        Task {
            if let def = try? await riskService.fetchRiskDefinition() {
                await MainActor.run { riskDefinition = def }
            }
        }
    }
}
