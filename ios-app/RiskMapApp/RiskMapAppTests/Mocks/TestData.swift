//
//  TestData.swift
//  RiskMapAppTests
//
//  Sample test data helpers for unit tests
//

import Foundation
import CoreLocation
import MapKit
@testable import RiskMapApp

struct TestData {
    // MARK: - Sample Road Segments

    static func sampleRoadSegment(
        id: String = "test-segment-1",
        name: String = "Queen Street West",
        roadClass: String = "Major Arterial",
        length: Double = 500.0,
        riskLevel: RiskLevel = .medium,
        confidence: Double = 0.85,
        totalCrashes: Int = 12,
        ksiCrashes: Int = 3,
        fatalities: Int = 1,
        coordinates: [RoadSegment.Coordinate]? = nil
    ) -> RoadSegment {
        let coords = coordinates ?? [
            RoadSegment.Coordinate(latitude: 43.6532, longitude: -79.3832),
            RoadSegment.Coordinate(latitude: 43.6542, longitude: -79.3842)
        ]

        return RoadSegment(
            id: id,
            linearName: name,
            roadClass: roadClass,
            segmentLength: length,
            riskLevel: riskLevel,
            confidence: confidence,
            numTotalCrashes: totalCrashes,
            numKSICrashes: ksiCrashes,
            fatalityCount: fatalities,
            coordinates: coords
        )
    }

    static func highRiskSegment() -> RoadSegment {
        sampleRoadSegment(
            id: "high-risk-1",
            name: "Dangerous Avenue",
            riskLevel: .high,
            confidence: 0.92,
            totalCrashes: 45,
            ksiCrashes: 12,
            fatalities: 3
        )
    }

    static func lowRiskSegment() -> RoadSegment {
        sampleRoadSegment(
            id: "low-risk-1",
            name: "Safe Street",
            riskLevel: .low,
            confidence: 0.88,
            totalCrashes: 2,
            ksiCrashes: 0,
            fatalities: 0
        )
    }

    // MARK: - API Response Data

    static func sampleAPIResponseData() -> Data {
        let segments = [
            sampleRoadSegment(),
            highRiskSegment(),
            lowRiskSegment()
        ]

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .useDefaultKeys
        return try! encoder.encode(segments)
    }

    static func emptyAPIResponseData() -> Data {
        let emptyArray: [RoadSegment] = []
        return try! JSONEncoder().encode(emptyArray)
    }

    static func invalidJSONData() -> Data {
        return "{invalid json}".data(using: .utf8)!
    }

    // MARK: - Test Locations

    static func torontoCoordinate() -> CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: 43.6532, longitude: -79.3832)
    }

    static func torontoRegion() -> MKCoordinateRegion {
        MKCoordinateRegion(
            center: torontoCoordinate(),
            span: MKCoordinateSpan(latitudeDelta: 0.1, longitudeDelta: 0.1)
        )
    }

    static func smallRegion() -> MKCoordinateRegion {
        MKCoordinateRegion(
            center: torontoCoordinate(),
            span: MKCoordinateSpan(latitudeDelta: 0.01, longitudeDelta: 0.01)
        )
    }

    static func largeRegion() -> MKCoordinateRegion {
        MKCoordinateRegion(
            center: torontoCoordinate(),
            span: MKCoordinateSpan(latitudeDelta: 1.0, longitudeDelta: 1.0)
        )
    }

    // MARK: - Sample Routes

    static func createMockMKRoute(
        distance: CLLocationDistance = 5000,
        expectedTime: TimeInterval = 600,
        name: String = "Test Route"
    ) -> MKRoute {
        // Note: Creating a real MKRoute requires MKDirections calculation
        // For unit tests, we'll need to use the actual routing API or mock at a higher level
        // This is a placeholder that shows the intent
        fatalError("Cannot create MKRoute directly - use mock at RouteService level")
    }
}
