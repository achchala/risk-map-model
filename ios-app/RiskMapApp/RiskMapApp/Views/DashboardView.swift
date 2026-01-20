//
//  DashboardView.swift
//  RiskMapApp
//
//  Dashboard view showing risk analysis statistics and charts
//

import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var riskService: RiskService
    @State private var dashboardStats: DashboardStats?
    @State private var isLoading = false
    @State private var errorMessage: String?
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    if isLoading {
                        ProgressView("Loading dashboard...")
                            .padding()
                    } else if let errorMessage = errorMessage {
                        VStack(spacing: 12) {
                            Image(systemName: "exclamationmark.triangle")
                                .font(.system(size: 48))
                                .foregroundColor(.orange)
                            Text(errorMessage)
                                .foregroundColor(.secondary)
                            Button("Retry") {
                                Task {
                                    await loadDashboardStats()
                                }
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding()
                    } else if let stats = dashboardStats {
                        // Header
                        VStack(spacing: 8) {
                            Text("Toronto Road Risk Analysis Dashboard")
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("Generated on \(Date().formatted(date: .abbreviated, time: .shortened))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top)
                        
                        // Statistics Cards
                        LazyVGrid(columns: [
                            GridItem(.flexible()),
                            GridItem(.flexible())
                        ], spacing: 16) {
                            StatCard(
                                title: "Low Risk Segments",
                                count: stats.riskDistribution.low,
                                percentage: stats.riskPercentages.low,
                                color: .green
                            )
                            
                            StatCard(
                                title: "Medium Risk Segments",
                                count: stats.riskDistribution.medium,
                                percentage: stats.riskPercentages.medium,
                                color: .orange
                            )
                            
                            StatCard(
                                title: "High Risk Segments",
                                count: stats.riskDistribution.high,
                                percentage: stats.riskPercentages.high,
                                color: .red
                            )
                            
                            StatCard(
                                title: "Total Segments",
                                count: stats.totalSegments,
                                percentage: nil,
                                color: .blue
                            )
                        }
                        .padding(.horizontal)
                        
                        // Pie Chart
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Risk Distribution")
                                .font(.headline)
                                .padding(.horizontal)
                            
                            Chart {
                                SectorMark(
                                    angle: .value("Low", stats.riskDistribution.low),
                                    innerRadius: .ratio(0.5),
                                    angularInset: 2
                                )
                                .foregroundStyle(Color.green)
                                .annotation(position: .overlay) {
                                    if stats.riskPercentages.low > 5 {
                                        Text("\(stats.riskPercentages.low, specifier: "%.1f")%")
                                            .font(.caption2)
                                            .fontWeight(.bold)
                                    }
                                }
                                
                                SectorMark(
                                    angle: .value("Medium", stats.riskDistribution.medium),
                                    innerRadius: .ratio(0.5),
                                    angularInset: 2
                                )
                                .foregroundStyle(Color.orange)
                                .annotation(position: .overlay) {
                                    if stats.riskPercentages.medium > 5 {
                                        Text("\(stats.riskPercentages.medium, specifier: "%.1f")%")
                                            .font(.caption2)
                                            .fontWeight(.bold)
                                    }
                                }
                                
                                SectorMark(
                                    angle: .value("High", stats.riskDistribution.high),
                                    innerRadius: .ratio(0.5),
                                    angularInset: 2
                                )
                                .foregroundStyle(Color.red)
                                .annotation(position: .overlay) {
                                    if stats.riskPercentages.high > 5 {
                                        Text("\(stats.riskPercentages.high, specifier: "%.1f")%")
                                            .font(.caption2)
                                            .fontWeight(.bold)
                                    }
                                }
                            }
                            .frame(height: 300)
                            .padding()
                            
                            // Legend
                            HStack(spacing: 20) {
                                LegendItem(color: .green, label: "Low Risk")
                                LegendItem(color: .orange, label: "Medium Risk")
                                LegendItem(color: .red, label: "High Risk")
                            }
                            .padding(.horizontal)
                        }
                        .padding(.vertical)
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                        .padding(.horizontal)
                        
                        // Bar Chart
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Risk Percentages")
                                .font(.headline)
                                .padding(.horizontal)
                            
                            Chart {
                                BarMark(
                                    x: .value("Risk", "Low Risk"),
                                    y: .value("Percentage", stats.riskPercentages.low)
                                )
                                .foregroundStyle(Color.green)
                                .annotation(position: .top) {
                                    Text("\(stats.riskPercentages.low, specifier: "%.1f")%")
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                }
                                
                                BarMark(
                                    x: .value("Risk", "Medium Risk"),
                                    y: .value("Percentage", stats.riskPercentages.medium)
                                )
                                .foregroundStyle(Color.orange)
                                .annotation(position: .top) {
                                    Text("\(stats.riskPercentages.medium, specifier: "%.1f")%")
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                }
                                
                                BarMark(
                                    x: .value("Risk", "High Risk"),
                                    y: .value("Percentage", stats.riskPercentages.high)
                                )
                                .foregroundStyle(Color.red)
                                .annotation(position: .top) {
                                    Text("\(stats.riskPercentages.high, specifier: "%.1f")%")
                                        .font(.caption)
                                        .fontWeight(.semibold)
                                }
                            }
                            .frame(height: 300)
                            .chartYAxis {
                                AxisMarks(position: .leading, values: .automatic(desiredCount: 5))
                            }
                            .chartYScale(domain: 0...100)
                            .padding()
                        }
                        .padding(.vertical)
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                        .padding(.horizontal)
                        .padding(.bottom)
                    }
                }
            }
            .navigationTitle("Dashboard")
            .refreshable {
                await loadDashboardStats()
            }
            .task {
                if dashboardStats == nil {
                    await loadDashboardStats()
                }
            }
        }
    }
    
    private func loadDashboardStats() async {
        await MainActor.run {
            isLoading = true
            errorMessage = nil
        }
        
        do {
            let stats = try await riskService.fetchDashboardStats()
            await MainActor.run {
                self.dashboardStats = stats
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.errorMessage = error.localizedDescription
                self.isLoading = false
            }
        }
    }
}

// MARK: - Stat Card
struct StatCard: View {
    let title: String
    let count: Int
    let percentage: Double?
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Text("\(count, format: .number.grouping(.automatic))")
                .font(.system(size: 32, weight: .bold))
                .foregroundColor(color)
            
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            if let percentage = percentage {
                Text("\(percentage, specifier: "%.1f")%")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(color)
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Legend Item
struct LegendItem: View {
    let color: Color
    let label: String
    
    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
            Text(label)
                .font(.caption)
        }
    }
}

struct DashboardView_Previews: PreviewProvider {
    static var previews: some View {
        DashboardView()
            .environmentObject(RiskService())
    }
}
