//
//  MockRiskService.swift
//  RiskMapAppTests
//
//  Mock risk service for testing
//

import Foundation
import MapKit
import Combine
@testable import RiskMapApp

class MockRiskService: RiskService {
    // Track method calls
    var fetchRiskPredictionsCallCount = 0
    var getRiskPredictionCallCount = 0
    var lastRegion: MKCoordinateRegion?
    var lastLocation: CLLocationCoordinate2D?

    // Configure mock responses
    var mockSegments: [RoadSegment] = []
    var mockPrediction: RiskPredictionResponse?
    var shouldThrowError = false
    var errorToThrow: Error = APIError.networkError(NSError(domain: "test", code: -1))

    override func fetchRiskPredictions(for region: MKCoordinateRegion) async throws -> [RoadSegment] {
        fetchRiskPredictionsCallCount += 1
        lastRegion = region

        await MainActor.run {
            self.isLoading = true
        }

        // Simulate network delay
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds

        if shouldThrowError {
            await MainActor.run {
                self.isLoading = false
                self.errorMessage = errorToThrow.localizedDescription
            }
            throw errorToThrow
        }

        await MainActor.run {
            self.roadSegments = mockSegments
            self.isLoading = false
            self.errorMessage = nil
        }

        return mockSegments
    }

    override func getRiskPrediction(for location: CLLocationCoordinate2D) async throws -> RiskPredictionResponse {
        getRiskPredictionCallCount += 1
        lastLocation = location

        if shouldThrowError {
            throw errorToThrow
        }

        guard let prediction = mockPrediction else {
            throw APIError.noData
        }

        return prediction
    }

    override func getHighRiskRoads() -> [RoadSegment] {
        return mockSegments.filter { $0.riskLevel == .high }
            .sorted { $0.numTotalCrashes > $1.numTotalCrashes }
    }

    // MARK: - Helper Methods

    func reset() {
        fetchRiskPredictionsCallCount = 0
        getRiskPredictionCallCount = 0
        lastRegion = nil
        lastLocation = nil
        mockSegments = []
        mockPrediction = nil
        shouldThrowError = false
        roadSegments = []
        isLoading = false
        errorMessage = nil
    }

    func setMockSegments(_ segments: [RoadSegment]) {
        mockSegments = segments
    }

    func setMockPrediction(_ prediction: RiskPredictionResponse) {
        mockPrediction = prediction
    }

    func simulateError(_ error: Error) {
        shouldThrowError = true
        errorToThrow = error
    }
}
