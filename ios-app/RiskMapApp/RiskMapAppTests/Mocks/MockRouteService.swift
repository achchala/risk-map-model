//
//  MockRouteService.swift
//  RiskMapAppTests
//
//  Mock route service for testing
//

import Foundation
import MapKit
import Combine
@testable import RiskMapApp

class MockRouteService: RouteService {
    // Track method calls
    var calculateRoutesCallCount = 0
    var lastStartLocation: CLLocationCoordinate2D?
    var lastDestinationLocation: CLLocationCoordinate2D?

    // Configure mock responses
    var mockSaferRoute: Route?
    var mockOptimalRoute: Route?
    var shouldThrowError = false
    var errorToThrow: Error = RouteError.routeCalculationFailed

    override func calculateRoutes(from start: CLLocationCoordinate2D, to destination: CLLocationCoordinate2D) async {
        calculateRoutesCallCount += 1
        lastStartLocation = start
        lastDestinationLocation = destination

        await MainActor.run {
            self.isLoading = true
            self.errorMessage = nil
        }

        // Simulate network delay
        do {
            try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        } catch {
            // Ignore cancellation
        }

        if shouldThrowError {
            await MainActor.run {
                self.errorMessage = errorToThrow.localizedDescription
                self.isLoading = false
            }
            return
        }

        await MainActor.run {
            self.saferRoute = mockSaferRoute
            self.optimalRoute = mockOptimalRoute
            self.isLoading = false
            self.errorMessage = nil
        }
    }

    // MARK: - Helper Methods

    func reset() {
        calculateRoutesCallCount = 0
        lastStartLocation = nil
        lastDestinationLocation = nil
        mockSaferRoute = nil
        mockOptimalRoute = nil
        shouldThrowError = false
        saferRoute = nil
        optimalRoute = nil
        isLoading = false
        errorMessage = nil
    }

    func setMockRoutes(safer: Route?, optimal: Route?) {
        mockSaferRoute = safer
        mockOptimalRoute = optimal
    }

    func simulateError(_ error: Error) {
        shouldThrowError = true
        errorToThrow = error
    }
}
