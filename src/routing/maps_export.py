"""
Route export utilities for external navigation apps (Google/Apple Maps).

Key responsibilities:
- simplify polyline geometries (Douglas–Peucker)
- cap the number of waypoints (e.g., 25)
- return coordinate sequences suitable for use as waypoints
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import logging
import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring

logger = logging.getLogger(__name__)


def simplify_route_for_export(
    route_geometry: LineString,
    max_waypoints: int = 25,
    tolerance: float = 0.0001,
) -> List[Tuple[float, float]]:
    """
    Simplify a route geometry and cap the number of waypoints.

    PHASE 5 — MAP DISPLAY (ROUTE GEOMETRY)
    After Dijkstra returns a path, the raw geometry is a dense sequence of
    coordinates from every road segment. This function prepares it for the
    iOS map by:

    Step 1 — Simplify (Douglas-Peucker algorithm):
        Removes redundant points that don't meaningfully change the route shape.
        tolerance=0.0001 degrees ≈ 10 meters — keeps the line visually accurate
        while reducing point count for rendering performance.

    Step 2 — Cap waypoints:
        External navigation apps (Google/Apple Maps) have waypoint limits.
        If the simplified route still has >25 points, evenly samples 25 points
        using numpy linspace to preserve the route's overall shape.

    Step 3 — Coordinate swap:
        Shapely stores coordinates as (lon, lat) internally.
        The iOS app expects (lat, lon) — so we swap the order here.

    Args:
        route_geometry: shapely LineString in EPSG:4326 (lon/lat).
        max_waypoints: maximum number of points for external apps.
        tolerance: Douglas-Peucker tolerance in degrees (~10m at city scale).

    Returns:
        List of (lat, lon) pairs suitable for use as waypoints.
    """
    if not isinstance(route_geometry, LineString):
        raise ValueError("route_geometry must be a LineString.")

    # Step 1: Remove redundant shape points while preserving visible route shape
    simplified = route_geometry.simplify(tolerance, preserve_topology=False)

    # Step 2: Extract coordinates — Shapely gives (lon, lat), we need (lat, lon) for iOS
    coords = list(simplified.coords)
    if not coords:
        return []

    # Step 3: Cap number of waypoints by evenly sampling across the route
    if len(coords) > max_waypoints:
        indices = np.linspace(0, len(coords) - 1, max_waypoints, dtype=int)
        coords = [coords[i] for i in indices]

    # Swap (lon, lat) → (lat, lon) to match iOS/Google Maps convention
    waypoints = [(lat, lon) for lon, lat in coords]
    logger.info("Simplified route to %d waypoints (max=%d).", len(waypoints), max_waypoints)
    return waypoints


def stitch_segment_geometries(geoms: Iterable[LineString]) -> LineString:
    """
    Stitch a sequence of segment LineStrings into a single continuous route.

    PHASE 5 — MAP DISPLAY (GEOMETRY STITCHING)
    Dijkstra returns a list of node IDs. Each consecutive node pair maps to an
    edge, and each edge has a 'geometry' attribute (a LineString for that segment).

    This function joins those per-segment LineStrings into one continuous
    LineString for the full route — so the iOS map can draw a single polyline
    rather than managing dozens of individual segments.

    Gap handling:
        If two adjacent segments share an endpoint (coords match), the duplicate
        point is dropped to avoid jagged artifacts on the map.
        Minor gaps (where the last point of one segment doesn't exactly equal the
        first point of the next) are tolerated — coordinates are just appended.

    Assumes segments are already ordered along the route (as returned by path_edges).
    """
    geoms = list(geoms)
    if not geoms:
        raise ValueError("No geometries provided to stitch.")

    coords: List[Tuple[float, float]] = []
    for geom in geoms:
        if not isinstance(geom, LineString):
            continue
        if not coords:
            # First segment — take all coordinates
            coords.extend(list(geom.coords))
        else:
            segment_coords = list(geom.coords)
            # If this segment starts where the last one ended, skip the duplicate point
            if coords[-1] == segment_coords[0]:
                coords.extend(segment_coords[1:])
            else:
                coords.extend(segment_coords)

    return LineString(coords)

