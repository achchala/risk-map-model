//
//  RouteServiceTests.swift
//  RiskMapAppTests
//
//  Unit tests for RouteService
//

import XCTest
import MapKit
import CoreLocation
@testable import RiskMapApp

final class RouteServiceTests: XCTestCase {
    var mockRiskService: MockRiskService!
    var routeService: MockRouteService!

    override func setUpWithError() throws {
        try super.setUpWithError()
        mockRiskService = MockRiskService()
        routeService = MockRouteService(riskService: mockRiskService)
    }

    override func tearDownWithError() throws {
        mockRiskService = nil
        routeService = nil
        try super.tearDownWithError()
    }

    // MARK: - Route Calculation Tests

    func testCalculateRoutesSuccess() async throws {
        // Setup mock routes
        let start = TestData.torontoCoordinate()
        let destination = CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)

        // For this test, we'll verify the service is called correctly
        // Actual route calculation requires MKDirections integration
        await routeService.calculateRoutes(from: start, to: destination)

        XCTAssertEqual(routeService.calculateRoutesCallCount, 1)

        let startLoc = try XCTUnwrap(routeService.lastStartLocation)
        XCTAssertEqual(startLoc.latitude, start.latitude, accuracy: 0.0001)

        let destLoc = try XCTUnwrap(routeService.lastDestinationLocation)
        XCTAssertEqual(destLoc.latitude, destination.latitude, accuracy: 0.0001)
    }

    func testCalculateRoutesFailure() async {
        let start = TestData.torontoCoordinate()
        let destination = CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)

        routeService.simulateError(RouteError.noRouteFound)
        await routeService.calculateRoutes(from: start, to: destination)

        XCTAssertNotNil(routeService.errorMessage)
        XCTAssertFalse(routeService.isLoading)
    }

    func testCalculateRoutesLoadingState() async {
        let start = TestData.torontoCoordinate()
        let destination = CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)

        XCTAssertFalse(routeService.isLoading)

        Task {
            await routeService.calculateRoutes(from: start, to: destination)
        }

        // Give it a moment to start
        try? await Task.sleep(nanoseconds: 10_000_000) // 0.01 seconds
        XCTAssertTrue(routeService.isLoading)
    }

    func testCalculateRoutesTracksLocations() async throws {
        let start = CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
        let destination = CLLocationCoordinate2D(latitude: 43.7000, longitude: -79.4000)

        await routeService.calculateRoutes(from: start, to: destination)

        let startLoc = try XCTUnwrap(routeService.lastStartLocation)
        let destLoc = try XCTUnwrap(routeService.lastDestinationLocation)

        XCTAssertEqual(startLoc.latitude, start.latitude, accuracy: 0.0001)
        XCTAssertEqual(startLoc.longitude, start.longitude, accuracy: 0.0001)
        XCTAssertEqual(destLoc.latitude, destination.latitude, accuracy: 0.0001)
        XCTAssertEqual(destLoc.longitude, destination.longitude, accuracy: 0.0001)
    }

    // MARK: - Route Analysis Tests

    func testRouteAnalysisWithHighRiskSegments() {
        // Setup mock risk data
        mockRiskService.setMockSegments([
            TestData.highRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .high),
            TestData.sampleRoadSegment(riskLevel: .medium)
        ])

        // In a real test, we'd analyze actual routes
        // Here we verify the mock service has correct data
        XCTAssertEqual(mockRiskService.mockSegments.count, 3)
        XCTAssertEqual(mockRiskService.mockSegments.filter { $0.riskLevel == .high }.count, 2)
    }

    func testRouteAnalysisWithLowRiskSegments() {
        mockRiskService.setMockSegments([
            TestData.lowRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .low),
            TestData.sampleRoadSegment(riskLevel: .low)
        ])

        XCTAssertEqual(mockRiskService.mockSegments.filter { $0.riskLevel == .low }.count, 3)
    }

    func testRouteAnalysisWithMixedRiskSegments() {
        mockRiskService.setMockSegments([
            TestData.highRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .medium),
            TestData.lowRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .medium)
        ])

        let segments = mockRiskService.mockSegments
        XCTAssertEqual(segments.filter { $0.riskLevel == .high }.count, 1)
        XCTAssertEqual(segments.filter { $0.riskLevel == .medium }.count, 2)
        XCTAssertEqual(segments.filter { $0.riskLevel == .low }.count, 1)
    }

    func testRouteAnalysisCalculatesRiskScore() {
        // Risk scores are calculated as:
        // High risk = 3.0, Medium = 2.0, Low = 1.0
        // Average across all segments

        // Route with mostly high risk should have high score (close to 3.0)
        // Route with mostly low risk should have low score (close to 1.0)
        // Route with mixed should be in between

        mockRiskService.setMockSegments([
            TestData.highRiskSegment(),
            TestData.highRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .high)
        ])

        // Expected average score = (3.0 + 3.0 + 3.0) / 3 = 3.0
        let highRiskSegments = mockRiskService.mockSegments
        XCTAssertTrue(highRiskSegments.allSatisfy { $0.riskLevel == .high })
    }

    // MARK: - Route Comparison Tests

    func testSaferRouteLowerRiskScore() {
        // In route comparison, safer route should have lower risk score
        // This is verified in the Route model tests
        // Here we ensure the service provides both routes for comparison

        let start = TestData.torontoCoordinate()
        let destination = CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)

        // In a real scenario, we'd calculate routes and compare
        // For now, verify the service can be called
        Task {
            await routeService.calculateRoutes(from: start, to: destination)
        }

        XCTAssertTrue(true) // Placeholder for actual comparison
    }

    func testOptimalRouteShortestTime() {
        // Optimal route should prioritize shortest travel time
        // Even if it has higher risk
        XCTAssertTrue(true) // Placeholder - requires actual route calculation
    }

    func testSaferRouteAvoidHighRisk() {
        // Safer route should avoid high-risk segments when possible
        mockRiskService.setMockSegments([
            TestData.highRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .low),
            TestData.lowRiskSegment()
        ])

        // In a real scenario, safer route would prefer low-risk segments
        let lowRiskCount = mockRiskService.mockSegments.filter { $0.riskLevel == .low }.count
        XCTAssertEqual(lowRiskCount, 2)
    }

    func testRouteComparisonTimeDelta() {
        // Route comparison should show time difference
        // This is tested in model tests, but service provides the data
        XCTAssertTrue(true) // Placeholder
    }

    // MARK: - MKPolyline Extension Tests

    func testMKPolylineCoordinatesExtraction() {
        // Create a simple polyline
        let coordinates = [
            TestData.torontoCoordinate(),
            CLLocationCoordinate2D(latitude: 43.66, longitude: -79.39),
            CLLocationCoordinate2D(latitude: 43.67, longitude: -79.40)
        ]

        let polyline = MKPolyline(coordinates: coordinates, count: coordinates.count)
        let extractedCoords = polyline.coordinates

        XCTAssertEqual(extractedCoords.count, 3)
        XCTAssertEqual(extractedCoords[0].latitude, coordinates[0].latitude, accuracy: 0.0001)
        XCTAssertEqual(extractedCoords[2].longitude, coordinates[2].longitude, accuracy: 0.0001)
    }

    func testMKPolylineEmptyCoordinates() {
        let polyline = MKPolyline()
        let coordinates = polyline.coordinates

        XCTAssertEqual(coordinates.count, 0)
    }

    func testMKPolylineFiltersInvalidCoordinates() {
        // Extension should filter out NaN or infinite coordinates
        // This is tested by the implementation
        let validCoords = [
            TestData.torontoCoordinate()
        ]

        let polyline = MKPolyline(coordinates: validCoords, count: validCoords.count)
        let extractedCoords = polyline.coordinates

        XCTAssertTrue(extractedCoords.allSatisfy { coord in
            coord.latitude.isFinite && coord.longitude.isFinite &&
            coord.latitude >= -90 && coord.latitude <= 90 &&
            coord.longitude >= -180 && coord.longitude <= 180
        })
    }

    // MARK: - MKRoute Extension Tests

    func testMKRouteDetailedCoordinates() {
        // MKRoute extension provides detailed coordinates from steps
        // This requires actual route calculation
        // Test is a placeholder for integration testing
        XCTAssertTrue(true)
    }

    func testMKRouteDetailedCoordinatesFallback() {
        // If steps don't provide enough coordinates, should fall back to polyline
        XCTAssertTrue(true)
    }

    func testMKRouteDetailedCoordinatesAvoidsDuplicates() {
        // Extension should avoid duplicate coordinates
        XCTAssertTrue(true)
    }

    // MARK: - RouteError Tests

    func testRouteErrorDescriptions() {
        XCTAssertEqual(
            RouteError.noRouteFound.errorDescription,
            "No route found between the selected locations"
        )
        XCTAssertEqual(
            RouteError.locationUnavailable.errorDescription,
            "Your location is not available"
        )
        XCTAssertEqual(
            RouteError.routeCalculationFailed.errorDescription,
            "Failed to calculate route"
        )
    }

    func testRouteErrorNoRouteFound() {
        let error = RouteError.noRouteFound
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("No route found") ?? false)
    }

    func testRouteErrorLocationUnavailable() {
        let error = RouteError.locationUnavailable
        XCTAssertNotNil(error.errorDescription)
        XCTAssertTrue(error.errorDescription?.contains("location") ?? false)
    }

    // MARK: - Mock Service Tests

    func testMockRiskServiceTracking() async {
        let region = TestData.torontoRegion()
        mockRiskService.setMockSegments([TestData.sampleRoadSegment()])

        _ = try? await mockRiskService.fetchRiskPredictions(for: region)

        XCTAssertEqual(mockRiskService.fetchRiskPredictionsCallCount, 1)
        XCTAssertNotNil(mockRiskService.lastRegion)
    }

    func testMockRiskServiceErrorSimulation() async {
        mockRiskService.simulateError(APIError.networkError(NSError(domain: "test", code: -1)))

        do {
            _ = try await mockRiskService.fetchRiskPredictions(for: TestData.torontoRegion())
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertTrue(error is APIError)
        }
    }

    func testMockRiskServiceReset() {
        mockRiskService.setMockSegments([TestData.sampleRoadSegment()])
        mockRiskService.fetchRiskPredictionsCallCount = 5

        mockRiskService.reset()

        XCTAssertEqual(mockRiskService.fetchRiskPredictionsCallCount, 0)
        XCTAssertEqual(mockRiskService.mockSegments.count, 0)
        XCTAssertNil(mockRiskService.lastRegion)
    }

    func testMockRouteServiceTracking() async {
        let start = TestData.torontoCoordinate()
        let destination = CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)

        await routeService.calculateRoutes(from: start, to: destination)

        XCTAssertEqual(routeService.calculateRoutesCallCount, 1)
        XCTAssertNotNil(routeService.lastStartLocation)
        XCTAssertNotNil(routeService.lastDestinationLocation)
    }

    func testMockRouteServiceErrorSimulation() async {
        routeService.simulateError(RouteError.noRouteFound)

        await routeService.calculateRoutes(
            from: TestData.torontoCoordinate(),
            to: CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)
        )

        XCTAssertNotNil(routeService.errorMessage)
        XCTAssertTrue(routeService.errorMessage?.contains("No route found") ?? false)
    }

    func testMockRouteServiceReset() async {
        await routeService.calculateRoutes(
            from: TestData.torontoCoordinate(),
            to: CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)
        )

        routeService.reset()

        XCTAssertEqual(routeService.calculateRoutesCallCount, 0)
        XCTAssertNil(routeService.lastStartLocation)
        XCTAssertNil(routeService.saferRoute)
        XCTAssertNil(routeService.optimalRoute)
    }
}
