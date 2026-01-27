//
//  GeometryHelpers.swift
//  RiskMapApp
//
//  Geometry utilities for calculating distances between points and polylines
//

import Foundation
import CoreLocation
import MapKit

/// Calculate the distance from a point to a line segment
/// - Parameters:
///   - point: The tap location coordinate
///   - lineStart: Starting coordinate of the line segment
///   - lineEnd: Ending coordinate of the line segment
/// - Returns: Distance in meters between the point and the line segment
func distanceToLineSegment(
    point: CLLocationCoordinate2D,
    lineStart: RoadSegment.Coordinate,
    lineEnd: RoadSegment.Coordinate
) -> CLLocationDistance {
    let pointLoc = CLLocation(latitude: point.latitude, longitude: point.longitude)
    let startLoc = CLLocation(latitude: lineStart.latitude, longitude: lineStart.longitude)
    let endLoc = CLLocation(latitude: lineEnd.latitude, longitude: lineEnd.longitude)

    // Calculate the deltas for the line segment
    let dx = lineEnd.longitude - lineStart.longitude
    let dy = lineEnd.latitude - lineStart.latitude
    let segmentLengthSq = dx * dx + dy * dy

    // Handle degenerate case where line segment is actually a point
    if segmentLengthSq == 0 {
        return pointLoc.distance(from: startLoc)
    }

    // Calculate projection parameter t (represents position along line segment)
    // t=0 means at start, t=1 means at end, 0<t<1 means somewhere in between
    let t = ((point.longitude - lineStart.longitude) * dx +
             (point.latitude - lineStart.latitude) * dy) / segmentLengthSq

    // Clamp t to [0, 1] to ensure we stay on the segment (not extending beyond endpoints)
    let tClamped = max(0, min(1, t))

    // Calculate the closest point on the segment
    let closestLat = lineStart.latitude + tClamped * dy
    let closestLon = lineStart.longitude + tClamped * dx
    let closestLoc = CLLocation(latitude: closestLat, longitude: closestLon)

    // Return geographic distance using haversine formula (via CLLocation)
    return pointLoc.distance(from: closestLoc)
}

/// Calculate the minimum distance from a point to a polyline
/// - Parameters:
///   - point: The tap location coordinate
///   - coordinates: Array of coordinates forming the polyline
/// - Returns: Minimum distance in meters from the point to any part of the polyline
func distanceToPolyline(
    point: CLLocationCoordinate2D,
    coordinates: [RoadSegment.Coordinate]
) -> CLLocationDistance {
    // Handle edge cases
    guard !coordinates.isEmpty else {
        return .infinity
    }

    // If polyline is just a single point
    if coordinates.count == 1 {
        let coord = coordinates[0]
        let pointLoc = CLLocation(latitude: point.latitude, longitude: point.longitude)
        let coordLoc = CLLocation(latitude: coord.latitude, longitude: coord.longitude)
        return pointLoc.distance(from: coordLoc)
    }

    var minDistance: CLLocationDistance = .infinity

    // Iterate through each line segment in the polyline
    for i in 0..<(coordinates.count - 1) {
        let segmentStart = coordinates[i]
        let segmentEnd = coordinates[i + 1]

        let distance = distanceToLineSegment(
            point: point,
            lineStart: segmentStart,
            lineEnd: segmentEnd
        )

        minDistance = min(minDistance, distance)
    }

    return minDistance
}

/// Find the nearest road segment to a given point
/// - Parameters:
///   - point: The tap location coordinate
///   - segments: Array of road segments to search through
///   - maxDistance: Maximum distance threshold in meters (default 50m)
/// - Returns: The nearest road segment within the threshold, or nil if none found
func findNearestSegment(
    to point: CLLocationCoordinate2D,
    in segments: [RoadSegment],
    maxDistance: CLLocationDistance = 50
) -> RoadSegment? {
    var nearestSegment: RoadSegment?
    var minDistance: CLLocationDistance = maxDistance

    let pointLoc = CLLocation(latitude: point.latitude, longitude: point.longitude)

    for segment in segments {
        // Skip segments with no coordinates
        guard !segment.coordinates.isEmpty else {
            continue
        }

        // Quick rejection: check if segment bounding box center is very far
        // This optimization skips distant segments before expensive polyline calculation
        let firstCoord = segment.coordinates[0]
        let firstLoc = CLLocation(latitude: firstCoord.latitude, longitude: firstCoord.longitude)
        let roughDistance = pointLoc.distance(from: firstLoc)

        // Skip if first point is more than 2x the max distance (rough filter)
        if roughDistance > (maxDistance * 2) {
            continue
        }

        // Calculate accurate distance to the polyline
        let distance = distanceToPolyline(point: point, coordinates: segment.coordinates)

        // Update nearest if this segment is closer
        if distance < minDistance {
            minDistance = distance
            nearestSegment = segment

            // Early exit optimization: if segment is very close (<10m), stop searching
            if minDistance < 10 {
                return nearestSegment
            }
        }
    }

    return nearestSegment
}
