//
//  RiskMapAppTests.swift
//  RiskMapAppTests
//
//  Created by Adriel De Vera on 2026-01-26.
//

import Testing
import Foundation
import CoreLocation
import MapKit
@testable import RiskMapApp

// MARK: - RiskLevel Tests
@Suite("RiskLevel Tests")
struct RiskLevelTests {

    @Test("RiskLevel display names are correct")
    func testDisplayNames() async throws {
        #expect(RiskLevel.low.displayName == "Low Risk")
        #expect(RiskLevel.medium.displayName == "Medium Risk")
        #expect(RiskLevel.high.displayName == "High Risk")
    }

    @Test("RiskLevel colors are correct hex values")
    func testColors() async throws {
        #expect(RiskLevel.low.color == "#2E8B57")      // green
        #expect(RiskLevel.medium.color == "#FFA500")   // orange
        #expect(RiskLevel.high.color == "#DC143C")     // crimson red
    }

    @Test("RiskLevel system images are correct")
    func testSystemImages() async throws {
        #expect(RiskLevel.low.systemImage == "checkmark.circle.fill")
        #expect(RiskLevel.medium.systemImage == "exclamationmark.circle.fill")
        #expect(RiskLevel.high.systemImage == "xmark.circle.fill")
    }

    @Test("RiskLevel can be encoded and decoded")
    func testCodable() throws {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        // Test all risk levels
        for level in RiskLevel.allCases {
            let encoded = try encoder.encode(level)
            let decoded = try decoder.decode(RiskLevel.self, from: encoded)
            #expect(decoded == level)
        }
    }

    @Test("RiskLevel decodes from string correctly")
    func testStringDecoding() throws {
        let jsonData = "\"low\"".data(using: .utf8)!
        let decoded = try JSONDecoder().decode(RiskLevel.self, from: jsonData)
        #expect(decoded == .low)
    }
}

// MARK: - RoadSegment Tests
@Suite("RoadSegment Model Tests")
struct RoadSegmentTests {

    @Test("RoadSegment decodes from JSON with custom CodingKeys")
    func testDecodingWithCodingKeys() throws {
        let json = """
        {
            "id": "seg123",
            "LINEAR_NAME": "King Street",
            "ROAD_CLASS": "Major Arterial",
            "segment_length": 500.5,
            "risk_label": "high",
            "confidence": 0.85,
            "num_total_crashes": 10,
            "num_ksi_crashes": 3,
            "fatality_count": 1,
            "coordinates": [
                {"latitude": 43.65, "longitude": -79.38},
                {"latitude": 43.66, "longitude": -79.37}
            ]
        }
        """

        let data = json.data(using: .utf8)!
        let segment = try JSONDecoder().decode(RoadSegment.self, from: data)

        #expect(segment.id == "seg123")
        #expect(segment.linearName == "King Street")
        #expect(segment.roadClass == "Major Arterial")
        #expect(segment.segmentLength == 500.5)
        #expect(segment.riskLevel == .high)
        #expect(segment.confidence == 0.85)
        #expect(segment.numTotalCrashes == 10)
        #expect(segment.numKSICrashes == 3)
        #expect(segment.fatalityCount == 1)
        #expect(segment.coordinates.count == 2)
    }

    @Test("RoadSegment coordinates decode correctly")
    func testCoordinatesDecoding() throws {
        let json = """
        {
            "id": "seg1",
            "LINEAR_NAME": "Test St",
            "ROAD_CLASS": "Local",
            "segment_length": 100.0,
            "risk_label": "low",
            "confidence": 0.6,
            "num_total_crashes": 0,
            "num_ksi_crashes": 0,
            "fatality_count": 0,
            "coordinates": [
                {"latitude": 43.6532, "longitude": -79.3832}
            ]
        }
        """

        let data = json.data(using: .utf8)!
        let segment = try JSONDecoder().decode(RoadSegment.self, from: data)

        #expect(segment.coordinates.count == 1)
        #expect(segment.coordinates[0].latitude == 43.6532)
        #expect(segment.coordinates[0].longitude == -79.3832)
    }
}

// MARK: - RiskService Tests
@Suite("RiskService Tests", .serialized)
struct RiskServiceTests {

    @Test("RiskService initializes with empty state")
    func testInitialState() async throws {
        let service = await RiskService()

        let isEmpty = await service.roadSegments.isEmpty
        let loading = await service.isLoading
        let error = await service.errorMessage

        #expect(isEmpty)
        #expect(loading == false)
        #expect(error == nil)
    }

    @Test("getHighRiskRoads filters by high risk level")
    func testGetHighRiskRoadsFiltering() async throws {
        let service = await RiskService()

        // Create mock segments
        let highRiskSegment = createMockSegment(
            id: "high1",
            riskLevel: .high,
            crashes: 10
        )
        let mediumRiskSegment = createMockSegment(
            id: "med1",
            riskLevel: .medium,
            crashes: 5
        )
        let lowRiskSegment = createMockSegment(
            id: "low1",
            riskLevel: .low,
            crashes: 2
        )

        await MainActor.run {
            service.roadSegments = [highRiskSegment, mediumRiskSegment, lowRiskSegment]
        }

        let highRiskRoads = await service.getHighRiskRoads()

        #expect(highRiskRoads.count == 1)
        #expect(highRiskRoads[0].riskLevel == .high)
    }

    @Test("getHighRiskRoads sorts by crash count descending")
    func testGetHighRiskRoadsSorting() async throws {
        let service = await RiskService()

        let highRisk1 = createMockSegment(id: "high1", riskLevel: .high, crashes: 5)
        let highRisk2 = createMockSegment(id: "high2", riskLevel: .high, crashes: 15)
        let highRisk3 = createMockSegment(id: "high3", riskLevel: .high, crashes: 10)

        await MainActor.run {
            service.roadSegments = [highRisk1, highRisk2, highRisk3]
        }

        let sorted = await service.getHighRiskRoads()

        #expect(sorted[0].numTotalCrashes == 15)
        #expect(sorted[1].numTotalCrashes == 10)
        #expect(sorted[2].numTotalCrashes == 5)
    }

    @Test("getHighRiskRoads returns empty array when no high risk segments")
    func testGetHighRiskRoadsEmpty() async throws {
        let service = await RiskService()

        let lowRisk = createMockSegment(id: "low1", riskLevel: .low, crashes: 2)

        await MainActor.run {
            service.roadSegments = [lowRisk]
        }

        let highRiskRoads = await service.getHighRiskRoads()
        #expect(highRiskRoads.isEmpty)
    }

    // Helper function to create mock segments
    private func createMockSegment(id: String, riskLevel: RiskLevel, crashes: Int) -> RoadSegment {
        return RoadSegment(
            id: id,
            linearName: "Test Street",
            roadClass: "Major Arterial",
            segmentLength: 500.0,
            riskLevel: riskLevel,
            confidence: 0.8,
            numTotalCrashes: crashes,
            numKSICrashes: 1,
            fatalityCount: 0,
            coordinates: [
                RoadSegment.Coordinate(latitude: 43.65, longitude: -79.38)
            ]
        )
    }
}

// MARK: - Route Model Tests
@Suite("Route Model Tests")
struct RouteModelTests {

    @Test("Route safety explanation generation - avoids high risk",
          .disabled("MKRoute cannot be mocked in unit tests - requires integration testing"))
    func testSafetyExplanationAvoidsHighRisk() async throws {
        let optimalMockRoute = createMockMKRoute()
        let saferMockRoute = createMockMKRoute()

        let optimalRoute = Route(
            route: optimalMockRoute,
            riskScore: 2.5,
            highRiskSegments: 3,
            mediumRiskSegments: 2,
            lowRiskSegments: 1,
            routeType: .optimal
        )

        let saferRoute = Route(
            route: saferMockRoute,
            riskScore: 1.2,
            highRiskSegments: 0,
            mediumRiskSegments: 3,
            lowRiskSegments: 3,
            routeType: .safer
        )

        let explanation = saferRoute.safetyExplanation(comparedTo: optimalRoute)

        #expect(explanation.contains("Completely avoids high-risk roads"))
    }

    @Test("Route safety explanation - reduces high risk segments",
          .disabled("MKRoute cannot be mocked in unit tests - requires integration testing"))
    func testSafetyExplanationReducesHighRisk() async throws {
        let optimalMockRoute = createMockMKRoute()
        let saferMockRoute = createMockMKRoute()

        let optimalRoute = Route(
            route: optimalMockRoute,
            riskScore: 2.5,
            highRiskSegments: 5,
            mediumRiskSegments: 2,
            lowRiskSegments: 1,
            routeType: .optimal
        )

        let saferRoute = Route(
            route: saferMockRoute,
            riskScore: 1.8,
            highRiskSegments: 2,
            mediumRiskSegments: 4,
            lowRiskSegments: 2,
            routeType: .safer
        )

        let explanation = saferRoute.safetyExplanation(comparedTo: optimalRoute)

        #expect(explanation.contains("3"))  // Should mention 3 fewer high-risk segments
    }

    @Test("Route distance and time properties work",
          .disabled("MKRoute cannot be mocked in unit tests - requires integration testing"))
    func testRouteProperties() async throws {
        let mockRoute = createMockMKRoute()

        let route = Route(
            route: mockRoute,
            riskScore: 1.5,
            highRiskSegments: 1,
            mediumRiskSegments: 2,
            lowRiskSegments: 3,
            routeType: .safer
        )

        // Basic property access should not crash
        let _ = route.estimatedTime
        let _ = route.distance
        let _ = route.polyline
        let _ = route.steps
    }

    // Helper to create mock MKRoute
    private func createMockMKRoute() -> MKRoute {
        // Create a simple directions request to get a real MKRoute
        // Note: This is a simplified mock - in real tests you might use a more sophisticated mock
        let source = MKMapItem(placemark: MKPlacemark(coordinate: CLLocationCoordinate2D(latitude: 43.65, longitude: -79.38)))
        let destination = MKMapItem(placemark: MKPlacemark(coordinate: CLLocationCoordinate2D(latitude: 43.66, longitude: -79.37)))

        let request = MKDirections.Request()
        request.source = source
        request.destination = destination

        // For unit tests, we can't easily create MKRoute synchronously
        // In practice, you'd either:
        // 1. Use a mock framework
        // 2. Test the RouteService integration tests separately
        // 3. Create a protocol for MKRoute and use dependency injection

        // For now, return a basic route (this will fail in actual execution)
        // This is a limitation of testing MapKit
        fatalError("MKRoute cannot be easily mocked - use integration tests for route-specific logic")
    }
}

// MARK: - API Error Tests
@Suite("API Error Tests")
struct APIErrorTests {

    @Test("APIError types are defined")
    func testAPIErrorTypes() async throws {
        // Test that API errors can be created
        let invalidURL = APIError.invalidURL
        let serverError = APIError.serverError("Test error")
        let decodingError = APIError.decodingError

        // Basic check that errors exist
        #expect(invalidURL != nil)
        #expect(serverError != nil)
        #expect(decodingError != nil)
    }
}

// MARK: - RiskPredictionResponse Tests
@Suite("RiskPredictionResponse Tests")
struct RiskPredictionResponseTests {

    @Test("RiskPredictionResponse decodes from JSON")
    func testDecoding() throws {
        let json = """
        {
            "riskLevel": "high",
            "confidence": 0.85,
            "probabilities": {
                "low": 0.05,
                "medium": 0.10,
                "high": 0.85
            }
        }
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(RiskPredictionResponse.self, from: data)

        #expect(response.riskLevel == .high)
        #expect(response.confidence == 0.85)
        #expect(response.probabilities.low == 0.05)
        #expect(response.probabilities.medium == 0.10)
        #expect(response.probabilities.high == 0.85)
    }

    @Test("RiskPredictionResponse with optional segmentInfo")
    func testDecodingWithOptionalSegment() throws {
        let json = """
        {
            "riskLevel": "medium",
            "confidence": 0.70,
            "probabilities": {
                "low": 0.15,
                "medium": 0.70,
                "high": 0.15
            },
            "segmentInfo": null
        }
        """

        let data = json.data(using: .utf8)!
        let response = try JSONDecoder().decode(RiskPredictionResponse.self, from: data)

        #expect(response.segmentInfo == nil)
    }
}
