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

    Args:
        route_geometry: shapely LineString in EPSG:4326 (lon/lat).
        max_waypoints: maximum number of points for external apps.
        tolerance: Douglas–Peucker tolerance in degrees (~10m at city scale).

    Returns:
        List of (lat, lon) pairs suitable for use as waypoints.
    """
    if not isinstance(route_geometry, LineString):
        raise ValueError("route_geometry must be a LineString.")

    # 1. Simplify geometry
    simplified = route_geometry.simplify(tolerance, preserve_topology=False)

    # 2. Extract coordinates (lon, lat) → (lat, lon)
    coords = list(simplified.coords)
    if not coords:
        return []

    # 3. Cap number of waypoints
    if len(coords) > max_waypoints:
        indices = np.linspace(0, len(coords) - 1, max_waypoints, dtype=int)
        coords = [coords[i] for i in indices]

    waypoints = [(lat, lon) for lon, lat in coords]
    logger.info("Simplified route to %d waypoints (max=%d).", len(waypoints), max_waypoints)
    return waypoints


def stitch_segment_geometries(geoms: Iterable[LineString]) -> LineString:
    """
    Stitch a sequence of segment LineStrings into a single continuous route.

    Assumes segments are ordered along the route; minor gaps are tolerated.
    """
    geoms = list(geoms)
    if not geoms:
        raise ValueError("No geometries provided to stitch.")

    # Concatenate coordinates in order
    coords: List[Tuple[float, float]] = []
    for geom in geoms:
        if not isinstance(geom, LineString):
            continue
        if not coords:
            coords.extend(list(geom.coords))
        else:
            # Avoid duplicating the first point if it matches last
            segment_coords = list(geom.coords)
            if coords[-1] == segment_coords[0]:
                coords.extend(segment_coords[1:])
            else:
                coords.extend(segment_coords)

    return LineString(coords)

