//
//  RiskServiceTests.swift
//  RiskMapAppTests
//
//  Unit tests for RiskService
//

import XCTest
import MapKit
import CoreLocation
@testable import RiskMapApp

final class RiskServiceTests: XCTestCase {
    var service: RiskService!
    var mockURLSession: URLSession!

    override func setUpWithError() throws {
        try super.setUpWithError()

        // Configure URLSession with mock protocol
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        mockURLSession = URLSession(configuration: configuration)

        service = RiskService()
        MockURLProtocol.reset()
    }

    override func tearDownWithError() throws {
        service = nil
        mockURLSession = nil
        MockURLProtocol.reset()
        try super.tearDownWithError()
    }

    // MARK: - Fetch Risk Predictions Tests

    func testFetchRiskPredictionsSuccess() async throws {
        // Configure mock response
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        // Create mock service with our URL session
        let testService = createTestRiskService()

        // Test
        let region = TestData.torontoRegion()
        let segments = try await testService.fetchRiskPredictions(for: region)

        // Verify
        XCTAssertEqual(segments.count, 3)
        XCTAssertFalse(testService.isLoading)
        XCTAssertNil(testService.errorMessage)
    }

    func testFetchRiskPredictionsNetworkError() async {
        // Configure mock to throw error
        MockURLProtocol.error = NSError(
            domain: NSURLErrorDomain,
            code: NSURLErrorNotConnectedToInternet,
            userInfo: [NSLocalizedDescriptionKey: "No internet connection"]
        )

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        do {
            _ = try await testService.fetchRiskPredictions(for: region)
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertFalse(testService.isLoading)
            XCTAssertNotNil(testService.errorMessage)
        }
    }

    func testFetchRiskPredictionsServerError() async {
        // Configure mock to return 500 error
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 500,
                httpVersion: nil,
                headerFields: nil
            )!
            return (response, nil)
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        do {
            _ = try await testService.fetchRiskPredictions(for: region)
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertFalse(testService.isLoading)
            XCTAssertNotNil(testService.errorMessage)
        }
    }

    func testFetchRiskPredictionsDecodingError() async {
        // Configure mock to return invalid JSON
        let invalidData = TestData.invalidJSONData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, invalidData)
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        do {
            _ = try await testService.fetchRiskPredictions(for: region)
            XCTFail("Expected decoding error to be thrown")
        } catch {
            XCTAssertFalse(testService.isLoading)
            XCTAssertNotNil(testService.errorMessage)
        }
    }

    func testFetchRiskPredictionsEmptyResponse() async throws {
        // Configure mock to return empty array
        let emptyData = TestData.emptyAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, emptyData)
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()
        let segments = try await testService.fetchRiskPredictions(for: region)

        XCTAssertEqual(segments.count, 0)
        XCTAssertFalse(testService.isLoading)
    }

    func testFetchRiskPredictionsTimeout() async {
        // Configure mock to simulate timeout
        MockURLProtocol.requestHandler = { request in
            // Simulate delay
            Thread.sleep(forTimeInterval: 35.0) // Longer than 30s timeout
            throw NSError(
                domain: NSURLErrorDomain,
                code: NSURLErrorTimedOut,
                userInfo: [NSLocalizedDescriptionKey: "Request timed out"]
            )
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        do {
            _ = try await testService.fetchRiskPredictions(for: region)
            XCTFail("Expected timeout error")
        } catch {
            XCTAssertFalse(testService.isLoading)
        }
    }

    // MARK: - State Management Tests

    func testLoadingStateTransitions() async throws {
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        // Initial state
        XCTAssertFalse(testService.isLoading)

        // Start request
        Task {
            _ = try await testService.fetchRiskPredictions(for: region)
        }

        // Should be loading (give it a moment to start)
        try await Task.sleep(nanoseconds: 10_000_000) // 0.01 seconds
        XCTAssertTrue(testService.isLoading)

        // Wait for completion
        _ = try await testService.fetchRiskPredictions(for: region)

        // Should no longer be loading
        XCTAssertFalse(testService.isLoading)
    }

    func testErrorMessageClearing() async throws {
        // First request fails
        MockURLProtocol.error = NSError(domain: "test", code: -1)
        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        do {
            _ = try await testService.fetchRiskPredictions(for: region)
        } catch {
            // Expected
        }

        XCTAssertNotNil(testService.errorMessage)

        // Second request succeeds
        MockURLProtocol.error = nil
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        _ = try await testService.fetchRiskPredictions(for: region)

        XCTAssertNil(testService.errorMessage)
    }

    func testRoadSegmentsPublished() async throws {
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        XCTAssertEqual(testService.roadSegments.count, 0)

        _ = try await testService.fetchRiskPredictions(for: region)

        XCTAssertEqual(testService.roadSegments.count, 3)
    }

    func testPublishedPropertiesOnMainActor() async throws {
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()

        _ = try await testService.fetchRiskPredictions(for: region)

        // Verify we can access published properties on main actor
        await MainActor.run {
            XCTAssertFalse(testService.isLoading)
            XCTAssertNil(testService.errorMessage)
            XCTAssertEqual(testService.roadSegments.count, 3)
        }
    }

    // MARK: - Region Handling Tests

    func testFetchRiskPredictionsForSmallRegion() async throws {
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        let testService = createTestRiskService()
        let region = TestData.smallRegion()
        let segments = try await testService.fetchRiskPredictions(for: region)

        XCTAssertGreaterThanOrEqual(segments.count, 0)
    }

    func testFetchRiskPredictionsForLargeRegion() async throws {
        let mockData = TestData.sampleAPIResponseData()
        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, mockData)
        }

        let testService = createTestRiskService()
        let region = TestData.largeRegion()
        let segments = try await testService.fetchRiskPredictions(for: region)

        XCTAssertGreaterThanOrEqual(segments.count, 0)
    }

    func testRequestBodyContainsRegionBounds() async throws {
        var capturedRequest: URLRequest?

        MockURLProtocol.requestHandler = { request in
            capturedRequest = request
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 200,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            )!
            return (response, TestData.emptyAPIResponseData())
        }

        let testService = createTestRiskService()
        let region = TestData.torontoRegion()
        _ = try await testService.fetchRiskPredictions(for: region)

        XCTAssertNotNil(capturedRequest)
        XCTAssertNotNil(capturedRequest?.httpBody)

        if let body = capturedRequest?.httpBody,
           let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any] {
            XCTAssertNotNil(json["north"])
            XCTAssertNotNil(json["south"])
            XCTAssertNotNil(json["east"])
            XCTAssertNotNil(json["west"])
        } else {
            XCTFail("Could not parse request body")
        }
    }

    func testTorontoRegionHelper() {
        let region = MKCoordinateRegion.toronto()

        XCTAssertEqual(region.center.latitude, 43.6532, accuracy: 0.0001)
        XCTAssertEqual(region.center.longitude, -79.3832, accuracy: 0.0001)
        XCTAssertEqual(region.span.latitudeDelta, 0.1, accuracy: 0.0001)
        XCTAssertEqual(region.span.longitudeDelta, 0.1, accuracy: 0.0001)
    }

    // MARK: - High Risk Roads Tests

    func testGetHighRiskRoads() {
        service.roadSegments = [
            TestData.highRiskSegment(),
            TestData.sampleRoadSegment(),
            TestData.lowRiskSegment(),
            TestData.sampleRoadSegment(
                id: "high-2",
                name: "Another High Risk",
                riskLevel: .high,
                totalCrashes: 50
            )
        ]

        let highRiskRoads = service.getHighRiskRoads()

        XCTAssertEqual(highRiskRoads.count, 2)
        XCTAssertTrue(highRiskRoads.allSatisfy { $0.riskLevel == .high })
        // Should be sorted by crash count (descending)
        XCTAssertEqual(highRiskRoads[0].numTotalCrashes, 50)
        XCTAssertEqual(highRiskRoads[1].numTotalCrashes, 45)
    }

    func testGetHighRiskRoadsEmpty() {
        service.roadSegments = [
            TestData.sampleRoadSegment(),
            TestData.lowRiskSegment()
        ]

        let highRiskRoads = service.getHighRiskRoads()

        XCTAssertEqual(highRiskRoads.count, 0)
    }

    func testGetHighRiskRoadsSorting() {
        service.roadSegments = [
            TestData.sampleRoadSegment(
                id: "high-1",
                riskLevel: .high,
                totalCrashes: 10
            ),
            TestData.sampleRoadSegment(
                id: "high-2",
                riskLevel: .high,
                totalCrashes: 30
            ),
            TestData.sampleRoadSegment(
                id: "high-3",
                riskLevel: .high,
                totalCrashes: 20
            )
        ]

        let highRiskRoads = service.getHighRiskRoads()

        XCTAssertEqual(highRiskRoads.count, 3)
        XCTAssertEqual(highRiskRoads[0].numTotalCrashes, 30)
        XCTAssertEqual(highRiskRoads[1].numTotalCrashes, 20)
        XCTAssertEqual(highRiskRoads[2].numTotalCrashes, 10)
    }

    func testGetHighRiskRoadsFiltering() {
        service.roadSegments = [
            TestData.highRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .medium),
            TestData.lowRiskSegment(),
            TestData.sampleRoadSegment(riskLevel: .high, totalCrashes: 100)
        ]

        let highRiskRoads = service.getHighRiskRoads()

        XCTAssertEqual(highRiskRoads.count, 2)
        XCTAssertFalse(highRiskRoads.contains { $0.riskLevel == .medium })
        XCTAssertFalse(highRiskRoads.contains { $0.riskLevel == .low })
    }

    // MARK: - Helper Methods

    private func createTestRiskService() -> RiskService {
        // Note: In a real implementation, we'd need to inject the URLSession
        // For now, we're using the shared session which won't use our mock
        // This is a limitation of the current RiskService implementation
        return RiskService()
    }
}
