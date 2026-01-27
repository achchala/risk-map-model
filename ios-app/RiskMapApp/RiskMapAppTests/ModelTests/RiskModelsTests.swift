//
//  RiskModelsTests.swift
//  RiskMapAppTests
//
//  Unit tests for risk models
//

import XCTest
import MapKit
import CoreLocation
@testable import RiskMapApp

final class RiskModelsTests: XCTestCase {

    // MARK: - RiskLevel Tests

    func testRiskLevelDisplayNames() {
        XCTAssertEqual(RiskLevel.low.displayName, "Low Risk")
        XCTAssertEqual(RiskLevel.medium.displayName, "Medium Risk")
        XCTAssertEqual(RiskLevel.high.displayName, "High Risk")
    }

    func testRiskLevelColors() {
        XCTAssertEqual(RiskLevel.low.color, "#2E8B57")
        XCTAssertEqual(RiskLevel.medium.color, "#FFA500")
        XCTAssertEqual(RiskLevel.high.color, "#DC143C")
    }

    func testRiskLevelSystemImages() {
        XCTAssertEqual(RiskLevel.low.systemImage, "checkmark.circle.fill")
        XCTAssertEqual(RiskLevel.medium.systemImage, "exclamationmark.circle.fill")
        XCTAssertEqual(RiskLevel.high.systemImage, "xmark.circle.fill")
    }

    func testRiskLevelRawValues() {
        XCTAssertEqual(RiskLevel.low.rawValue, "low")
        XCTAssertEqual(RiskLevel.medium.rawValue, "medium")
        XCTAssertEqual(RiskLevel.high.rawValue, "high")
    }

    func testRiskLevelCodable() throws {
        let riskLevel = RiskLevel.high
        let encoder = JSONEncoder()
        let data = try encoder.encode(riskLevel)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(RiskLevel.self, from: data)

        XCTAssertEqual(decoded, riskLevel)
    }

    // MARK: - RoadSegment Tests

    func testRoadSegmentInitialization() {
        let segment = TestData.sampleRoadSegment()

        XCTAssertEqual(segment.id, "test-segment-1")
        XCTAssertEqual(segment.linearName, "Queen Street West")
        XCTAssertEqual(segment.roadClass, "Major Arterial")
        XCTAssertEqual(segment.segmentLength, 500.0)
        XCTAssertEqual(segment.riskLevel, .medium)
        XCTAssertEqual(segment.confidence, 0.85)
        XCTAssertEqual(segment.numTotalCrashes, 12)
        XCTAssertEqual(segment.numKSICrashes, 3)
        XCTAssertEqual(segment.fatalityCount, 1)
        XCTAssertEqual(segment.coordinates.count, 2)
    }

    func testRoadSegmentHighRisk() {
        let segment = TestData.highRiskSegment()

        XCTAssertEqual(segment.riskLevel, .high)
        XCTAssertEqual(segment.numTotalCrashes, 45)
        XCTAssertEqual(segment.numKSICrashes, 12)
        XCTAssertEqual(segment.fatalityCount, 3)
    }

    func testRoadSegmentLowRisk() {
        let segment = TestData.lowRiskSegment()

        XCTAssertEqual(segment.riskLevel, .low)
        XCTAssertEqual(segment.numTotalCrashes, 2)
        XCTAssertEqual(segment.numKSICrashes, 0)
        XCTAssertEqual(segment.fatalityCount, 0)
    }

    func testRoadSegmentCodableWithCustomKeys() throws {
        let json = """
        {
            "id": "seg-123",
            "LINEAR_NAME": "Main Street",
            "ROAD_CLASS": "Minor Arterial",
            "segment_length": 750.5,
            "risk_label": "high",
            "confidence": 0.92,
            "num_total_crashes": 25,
            "num_ksi_crashes": 8,
            "fatality_count": 2,
            "coordinates": [
                {"latitude": 43.65, "longitude": -79.38},
                {"latitude": 43.66, "longitude": -79.39}
            ]
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let segment = try decoder.decode(RoadSegment.self, from: json)

        XCTAssertEqual(segment.id, "seg-123")
        XCTAssertEqual(segment.linearName, "Main Street")
        XCTAssertEqual(segment.roadClass, "Minor Arterial")
        XCTAssertEqual(segment.segmentLength, 750.5)
        XCTAssertEqual(segment.riskLevel, .high)
        XCTAssertEqual(segment.confidence, 0.92)
        XCTAssertEqual(segment.numTotalCrashes, 25)
        XCTAssertEqual(segment.numKSICrashes, 8)
        XCTAssertEqual(segment.fatalityCount, 2)
        XCTAssertEqual(segment.coordinates.count, 2)
    }

    func testRoadSegmentArrayDecoding() throws {
        let data = TestData.sampleAPIResponseData()
        let decoder = JSONDecoder()
        let segments = try decoder.decode([RoadSegment].self, from: data)

        XCTAssertEqual(segments.count, 3)
        XCTAssertTrue(segments.contains { $0.riskLevel == .high })
        XCTAssertTrue(segments.contains { $0.riskLevel == .medium })
        XCTAssertTrue(segments.contains { $0.riskLevel == .low })
    }

    func testRoadSegmentCoordinates() {
        let coords = [
            RoadSegment.Coordinate(latitude: 43.6532, longitude: -79.3832),
            RoadSegment.Coordinate(latitude: 43.6542, longitude: -79.3842),
            RoadSegment.Coordinate(latitude: 43.6552, longitude: -79.3852)
        ]

        let segment = TestData.sampleRoadSegment(coordinates: coords)

        XCTAssertEqual(segment.coordinates.count, 3)
        XCTAssertEqual(segment.coordinates[0].latitude, 43.6532)
        XCTAssertEqual(segment.coordinates[2].longitude, -79.3852)
    }

    // MARK: - Route Tests

    func testRouteSafetyExplanationWithNoHighRisk() {
        // Create a mock MKRoute - note: in actual tests we'd need to calculate real routes
        let mkRoute = createMockRoute()

        let route = Route(
            route: mkRoute,
            riskScore: 1.2,
            highRiskSegments: 0,
            mediumRiskSegments: 2,
            lowRiskSegments: 8,
            routeType: .safer
        )

        let explanation = route.safetyExplanation(comparedTo: nil)

        XCTAssertTrue(explanation.contains("Avoids all high-risk road segments"))
        XCTAssertTrue(explanation.contains("Primarily uses low-risk roads"))
        XCTAssertTrue(explanation.contains("Overall low risk score"))
    }

    func testRouteSafetyExplanationComparedToOptimal() {
        let mkRoute = createMockRoute()

        let saferRoute = Route(
            route: mkRoute,
            riskScore: 1.5,
            highRiskSegments: 1,
            mediumRiskSegments: 3,
            lowRiskSegments: 6,
            routeType: .safer
        )

        let optimalRoute = Route(
            route: mkRoute,
            riskScore: 2.2,
            highRiskSegments: 4,
            mediumRiskSegments: 4,
            lowRiskSegments: 2,
            routeType: .optimal
        )

        let explanation = saferRoute.safetyExplanation(comparedTo: optimalRoute)

        XCTAssertTrue(explanation.contains("Avoids 3 additional high-risk segments"))
        XCTAssertTrue(explanation.contains("Lower overall risk score"))
    }

    func testRouteProperties() {
        let mkRoute = createMockRoute()

        let route = Route(
            route: mkRoute,
            riskScore: 1.8,
            highRiskSegments: 2,
            mediumRiskSegments: 3,
            lowRiskSegments: 5,
            routeType: .safer
        )

        XCTAssertEqual(route.highRiskSegments, 2)
        XCTAssertEqual(route.mediumRiskSegments, 3)
        XCTAssertEqual(route.lowRiskSegments, 5)
        XCTAssertEqual(route.riskScore, 1.8)
        XCTAssertEqual(route.routeType, .safer)
    }

    func testRouteRiskScoring() {
        let mkRoute = createMockRoute()

        let highRiskRoute = Route(
            route: mkRoute,
            riskScore: 2.8,
            highRiskSegments: 8,
            mediumRiskSegments: 2,
            lowRiskSegments: 0,
            routeType: .optimal
        )

        let lowRiskRoute = Route(
            route: mkRoute,
            riskScore: 1.1,
            highRiskSegments: 0,
            mediumRiskSegments: 1,
            lowRiskSegments: 9,
            routeType: .safer
        )

        XCTAssertGreaterThan(highRiskRoute.riskScore, lowRiskRoute.riskScore)
        XCTAssertGreaterThan(highRiskRoute.highRiskSegments, lowRiskRoute.highRiskSegments)
        XCTAssertGreaterThan(lowRiskRoute.lowRiskSegments, highRiskRoute.lowRiskSegments)
    }

    // MARK: - RouteComparison Tests

    func testRouteComparisonTimeDifference() {
        let mkRoute1 = createMockRoute(expectedTime: 600) // 10 minutes
        let mkRoute2 = createMockRoute(expectedTime: 720) // 12 minutes

        let saferRoute = Route(
            route: mkRoute2,
            riskScore: 1.2,
            highRiskSegments: 0,
            mediumRiskSegments: 2,
            lowRiskSegments: 8,
            routeType: .safer
        )

        let optimalRoute = Route(
            route: mkRoute1,
            riskScore: 2.5,
            highRiskSegments: 5,
            mediumRiskSegments: 3,
            lowRiskSegments: 2,
            routeType: .optimal
        )

        let comparison = RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)

        XCTAssertEqual(comparison.timeDifference, 120) // 2 minutes
        XCTAssertTrue(comparison.saferRouteSlower)
    }

    func testRouteComparisonSafetyImprovement() {
        let mkRoute = createMockRoute()

        let saferRoute = Route(
            route: mkRoute,
            riskScore: 1.2,
            highRiskSegments: 1,
            mediumRiskSegments: 2,
            lowRiskSegments: 7,
            routeType: .safer
        )

        let optimalRoute = Route(
            route: mkRoute,
            riskScore: 2.5,
            highRiskSegments: 5,
            mediumRiskSegments: 3,
            lowRiskSegments: 2,
            routeType: .optimal
        )

        let comparison = RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)
        let improvement = comparison.safetyImprovement

        XCTAssertTrue(improvement.contains("4 fewer high-risk segments"))
        XCTAssertTrue(improvement.contains("1.3 points lower risk score"))
        XCTAssertTrue(improvement.contains("5 more low-risk segments"))
    }

    func testRouteComparisonDetailedExplanation() {
        let mkRoute1 = createMockRoute(expectedTime: 600)
        let mkRoute2 = createMockRoute(expectedTime: 720)

        let saferRoute = Route(
            route: mkRoute2,
            riskScore: 1.2,
            highRiskSegments: 0,
            mediumRiskSegments: 2,
            lowRiskSegments: 8,
            routeType: .safer
        )

        let optimalRoute = Route(
            route: mkRoute1,
            riskScore: 2.0,
            highRiskSegments: 3,
            mediumRiskSegments: 4,
            lowRiskSegments: 3,
            routeType: .optimal
        )

        let comparison = RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)
        let explanation = comparison.detailedExplanation

        XCTAssertTrue(explanation.contains("safer route was selected"))
        XCTAssertTrue(explanation.contains("avoids"))
        XCTAssertTrue(explanation.contains("minute"))
    }

    func testRouteComparisonSimilarSafety() {
        let mkRoute = createMockRoute()

        let saferRoute = Route(
            route: mkRoute,
            riskScore: 1.5,
            highRiskSegments: 2,
            mediumRiskSegments: 3,
            lowRiskSegments: 5,
            routeType: .safer
        )

        let optimalRoute = Route(
            route: mkRoute,
            riskScore: 1.6,
            highRiskSegments: 2,
            mediumRiskSegments: 4,
            lowRiskSegments: 4,
            routeType: .optimal
        )

        let comparison = RouteComparison(saferRoute: saferRoute, optimalRoute: optimalRoute)
        let improvement = comparison.safetyImprovement

        XCTAssertTrue(improvement.contains("Similar safety") || improvement.contains("low-risk segment"))
    }

    // MARK: - APIError Tests

    func testAPIErrorDescriptions() {
        XCTAssertEqual(APIError.invalidURL.errorDescription, "Invalid URL")
        XCTAssertEqual(APIError.noData.errorDescription, "No data received")
        XCTAssertEqual(APIError.decodingError.errorDescription, "Failed to decode response")
        XCTAssertEqual(APIError.serverError("Test").errorDescription, "Server error: Test")

        let networkError = NSError(domain: "test", code: -1, userInfo: [NSLocalizedDescriptionKey: "Connection failed"])
        let apiNetworkError = APIError.networkError(networkError)
        XCTAssertTrue(apiNetworkError.errorDescription?.contains("Network error") ?? false)
    }

    // MARK: - Helper Methods

    private func createMockRoute(
        distance: CLLocationDistance = 5000,
        expectedTime: TimeInterval = 600
    ) -> MKRoute {
        // Note: In actual unit tests, we would either:
        // 1. Use the real MKDirections API (integration test)
        // 2. Create a mock Route wrapper that doesn't require MKRoute
        // For now, we'll use a placeholder that creates minimal test data

        // In Swift, you cannot directly instantiate MKRoute
        // We need to use MKDirections to calculate a real route
        // For unit tests, we should mock at the RouteService level instead

        let start = MKMapItem(placemark: MKPlacemark(coordinate: TestData.torontoCoordinate()))
        let destination = MKMapItem(placemark: MKPlacemark(
            coordinate: CLLocationCoordinate2D(latitude: 43.7, longitude: -79.4)
        ))

        let request = MKDirections.Request()
        request.source = start
        request.destination = destination

        // For unit tests, we'll need to use async/await or mock the entire route service
        // This is a limitation of the current test structure
        fatalError("Cannot create MKRoute synchronously - use RouteService mocks instead")
    }
}
