"""
Routing graph construction and risk-aware edge weighting.

This module builds a directed graph from the road network using
FROM_INTERSECTION_ID / TO_INTERSECTION_ID and CENTRELINE_ID as the
stable segment identifier, and provides helpers for:

- computing edge travel times from segment length and road class
- converting crash rate estimates λ into expected crashes per edge
- forming combined time + risk edge weights for safer routing
- finding fastest vs safer routes with Dijkstra
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Tuple

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PHASE 1 — TRAVEL TIME ESTIMATION
# These are the assumed speeds for each road type in Toronto.
# No live traffic data is used — these are fixed at startup.
# They feed directly into the travel_time_hours edge attribute on every segment.
# ---------------------------------------------------------------------------
DEFAULT_SPEEDS_KMH: Dict[str, float] = {
    "arterial": 50.0,       # Major roads like Yonge, Bloor
    "collector": 40.0,      # Secondary roads feeding into arterials
    "local": 30.0,          # Residential streets
    "minor_arterial": 45.0, # Smaller arterials between major roads
}


def estimate_travel_time_hours(
    segment_length_m: float,
    road_class: str,
    default_speeds_kmh: Dict[str, float] | None = None,
) -> float:
    """
    Estimate travel time (in hours) from length and road class.

    PHASE 1 — TRAVEL TIME ESTIMATION
    Called once per segment at graph-build time. The result is stored as the
    'travel_time_hours' edge attribute and is used by Dijkstra for the fastest route.

    Formula: travel_time_hours = segment_length_m / (speed_kmh / 3.6) / 3600

    Units:
    - segment_length_m: meters
    - speed: km/h
    - return: hours
    """
    if default_speeds_kmh is None:
        default_speeds_kmh = DEFAULT_SPEEDS_KMH

    # Look up speed by road class; default to 40 km/h if not found
    speed_kmh = default_speeds_kmh.get(str(road_class).lower(), 40.0)
    speed_ms = speed_kmh / 3.6  # convert km/h → m/s
    if speed_ms <= 0:
        # Fallback in pathological case
        speed_ms = 10.0

    travel_time_seconds = segment_length_m / speed_ms
    return travel_time_seconds / 3600.0


def build_road_graph(road_network: gpd.GeoDataFrame) -> nx.DiGraph:
    """
    Build a directed graph from the road network.

    PHASE 1 — GRAPH CONSTRUCTION
    Called once at server startup. Converts the Toronto Centreline GeoDataFrame
    into a networkx DiGraph where:
      - Each node  = an intersection (FROM_INTERSECTION_ID / TO_INTERSECTION_ID)
      - Each edge  = a road segment between those intersections (CENTRELINE_ID)

    Edge attributes stored for use downstream:
      - segment_id        → used to look up ML crash rates (λ) per segment
      - geometry          → used to draw the route on the map
      - length_m          → raw segment length in meters
      - road_class        → determines assumed travel speed
      - travel_time_hours → pre-computed cost for fastest-route Dijkstra

    One-way handling:
        - If ONEWAY_DIR_CODE indicates one-way, add edge in that direction only.
        - Otherwise, add reverse edge as well (same segment_id).
    """
    required_cols = {"CENTRELINE_ID", "segment_length", "ROAD_CLASS"}
    missing_required = required_cols - set(road_network.columns)
    if missing_required:
        raise ValueError(f"Road network missing required columns: {missing_required}")

    if "FROM_INTERSECTION_ID" not in road_network.columns or "TO_INTERSECTION_ID" not in road_network.columns:
        raise ValueError(
            "Road network must contain FROM_INTERSECTION_ID and TO_INTERSECTION_ID "
            "to build a routing graph."
        )

    # Each node in the graph is an intersection ID (a plain number from the Centreline data).
    # Edges are the road segments connecting those intersections.
    G = nx.DiGraph()

    for _, seg in road_network.iterrows():
        from_node = seg["FROM_INTERSECTION_ID"]   # intersection at the start of this segment
        to_node = seg["TO_INTERSECTION_ID"]        # intersection at the end of this segment
        segment_id = seg["CENTRELINE_ID"]          # stable ID used to look up ML crash rates later
        length_m = float(seg["segment_length"])
        road_class = seg.get("ROAD_CLASS", "unknown")
        geom = seg.geometry

        if isinstance(geom, LineString):
            geometry = geom
        elif isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
            # Merge all parts into one LineString for routing
            # (some segments in the Centreline data are stored as MultiLineStrings)
            coords = []
            for g in geom.geoms:
                coords.extend(list(g.coords))
            if len(coords) < 2:
                continue
            geometry = LineString(coords)
        else:
            continue  # skip segments with no usable geometry

        # Skip segments with invalid intersection IDs (can't be added to the graph)
        try:
            if from_node is None or to_node is None or np.isnan(from_node) or np.isnan(to_node):
                continue
        except (TypeError, ValueError):
            continue

        # Pre-compute travel time so it's ready for Dijkstra at query time
        t_hours = estimate_travel_time_hours(length_m, str(road_class))

        G.add_edge(
            from_node,
            to_node,
            segment_id=segment_id,
            geometry=geometry,
            length_m=length_m,
            road_class=road_class,
            travel_time_hours=t_hours,  # cost for fastest-route Dijkstra
        )

        # One-way streets only get a single directed edge (from → to).
        # Bidirectional streets get a second reverse edge with the same attributes.
        oneway_code = str(seg.get("ONEWAY_DIR_CODE", "")).upper()
        is_one_way = oneway_code in {"NB", "SB", "EB", "WB", "ONE_WAY"}

        if not is_one_way:
            # Add reverse edge with same segment_id and attributes
            G.add_edge(
                to_node,
                from_node,
                segment_id=segment_id,
                geometry=geometry,
                length_m=length_m,
                road_class=road_class,
                travel_time_hours=t_hours,
            )

    logger.info(
        "Built road graph with %d nodes and %d edges.",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def apply_risk_to_edge_costs(
    G: nx.DiGraph,
    lambda_per_hour: Dict[Hashable, float],
    beta_hours_per_expected_crash: float = 0.1,
) -> None:
    """
    Annotate edges with expected crashes and combined cost.

    PHASE 3 — RISK WEIGHTING
    Called on every route request (after multipliers are applied) to attach a
    risk-adjusted cost to every edge. The safer-route Dijkstra minimises
    'risk_weight_hours' instead of raw travel time.

    How the penalty works:
        expected_crashes  = λ × travel_time_hours
            → "how many crashes would we expect on this segment during this trip?"

        risk_weight_hours = travel_time_hours + β × expected_crashes
            → makes risky segments appear "slower" so Dijkstra avoids them

    β (beta) defaults to 0.1 h/expected crash — tunable per request.
    A higher β makes the safer route avoid risk more aggressively, even at the
    cost of more travel time.

    Parameters:
        G: graph with 'segment_id' and 'travel_time_hours' on each edge
        lambda_per_hour: mapping from segment_id to crash rate λ (crashes/hour)
                         (already multiplied by weather/time factors before this call)
        beta_hours_per_expected_crash: how much time-penalty to assign per expected crash

    Edge attributes added:
        - expected_crashes  → used in the route risk summary returned to the app
        - risk_weight_hours → used as the Dijkstra weight for the safer route
    """
    def _get_lam(seg_id, lam_dict):
        """Look up lambda with type normalization (int/np.int64/float key mismatch)."""
        if seg_id is None:
            return 0.0
        v = lam_dict.get(seg_id, None)
        if v is not None:
            return float(v)
        try:
            v = lam_dict.get(int(seg_id), None)
            if v is not None:
                return float(v)
        except (ValueError, TypeError):
            pass
        return 0.0  # segment not in λ map → treat as zero risk

    for u, v, data in G.edges(data=True):
        seg_id = data.get("segment_id")
        travel_time = float(data.get("travel_time_hours", 0.0))
        lam = _get_lam(seg_id, lambda_per_hour)

        expected_crashes = lam * travel_time  # expected crash count on this edge during this trip
        data["expected_crashes"] = expected_crashes
        # risk_weight_hours is what the safer-route Dijkstra minimises
        data["risk_weight_hours"] = travel_time + beta_hours_per_expected_crash * expected_crashes


def find_fastest_route(
    G: nx.DiGraph,
    start_node: Hashable,
    end_node: Hashable,
) -> List[Hashable]:
    """
    Find the fastest route (time-only) using Dijkstra on travel_time_hours.

    PHASE 3 — DIJKSTRA (FASTEST)
    Ignores crash risk entirely. Only cares about how long each segment takes
    to drive based on road class speed assumptions. This is the "ignore safety"
    baseline shown to the user alongside the safer route.
    """
    path = nx.dijkstra_path(G, source=start_node, target=end_node, weight="travel_time_hours")
    return path


def find_safer_route(
    G: nx.DiGraph,
    start_node: Hashable,
    end_node: Hashable,
) -> List[Hashable]:
    """
    Find a safer route using Dijkstra on combined time + risk weight.

    PHASE 3 — DIJKSTRA (SAFER)
    Uses risk_weight_hours = travel_time + β × expected_crashes as the edge cost.
    Segments with higher crash rates appear "slower", so Dijkstra naturally routes
    around them — even if the physical travel time is similar.

    Assumes apply_risk_to_edge_costs() has been called first so that each edge
    already has a 'risk_weight_hours' attribute.
    """
    path = nx.dijkstra_path(G, source=start_node, target=end_node, weight="risk_weight_hours")
    return path


def path_edges(G: nx.DiGraph, path: Iterable[Hashable]) -> List[Tuple[Hashable, Hashable, Dict]]:
    """
    Convenience helper to get edge data along a node path.
    """
    nodes = list(path)
    edges: List[Tuple[Hashable, Hashable, Dict]] = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        data = G[u][v]
        edges.append((u, v, data))
    return edges


def calculate_route_risk(
    G: nx.DiGraph,
    path: Iterable[Hashable],
) -> Dict[str, float]:
    """
    Aggregate expected crashes and route crash probability for a path.

    PHASE 4 — ROUTE RISK SCORE
    After Dijkstra picks a path, this function rolls up all the per-edge
    expected crash counts into a single route-level risk score.

    Poisson model:
        Λ = Σ(λ_i × t_i)          total expected crashes across all edges
        P = 1 − e^{−Λ}            probability of at least one crash on this route

    This is the standard Poisson "at least one event" formula. It works because
    crashes on individual segments are assumed to be rare, independent events.

    Assumes edges along the path already have 'expected_crashes' and
    'travel_time_hours' set (i.e., apply_risk_to_edge_costs was called first).

    Returns:
        {
            "expected_crashes": Σ(λ_i * t_i),
            "route_probability": 1 - exp(-Σ(λ_i * t_i)),
            "total_travel_time_hours": Σ(t_i),
        }
    """
    edges = path_edges(G, path)
    total_expected = 0.0
    total_time = 0.0
    for _, _, data in edges:
        t = float(data.get("travel_time_hours", 0.0))
        ec = float(data.get("expected_crashes", 0.0))
        total_time += t
        total_expected += ec

    # Poisson CDF: probability of at least one crash on the full route
    route_probability = 1.0 - float(np.exp(-total_expected))
    return {
        "expected_crashes": total_expected,
        "route_probability": route_probability,
        "total_travel_time_hours": total_time,
    }


def build_node_geometry(road_network: gpd.GeoDataFrame) -> Dict[Hashable, Point]:
    """
    Derive approximate node coordinates for each intersection ID from segment endpoints.

    PHASE 1 — NODE GEOMETRY
    Graph nodes are just numeric IDs — they have no coordinates attached.
    This function creates a separate lookup dict (node_id → lat/lon Point) by
    reading the first/last coordinate of each segment's geometry.

    Why it's needed:
        When the user taps a destination on the map, the app sends a lat/lon.
        We need to convert that into a node ID the graph understands.
        snap_to_graph() uses this dict to find the nearest node.

    Returns:
        dict: intersection_id -> Point (in EPSG:4326, i.e. standard lat/lon)
    """
    # Re-project to lat/lon so snap_to_graph can compare against user GPS coordinates
    roads_geo = road_network.to_crs("EPSG:4326")

    node_coords: Dict[Hashable, Point] = {}

    for _, seg in roads_geo.iterrows():
        geom = seg.geometry
        if isinstance(geom, LineString):
            start_pt = Point(geom.coords[0])  # first vertex = FROM intersection location
            end_pt = Point(geom.coords[-1])    # last vertex  = TO intersection location
        elif isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
            first_line = geom.geoms[0]
            last_line = geom.geoms[-1]
            start_pt = Point(first_line.coords[0])
            end_pt = Point(last_line.coords[-1])
        else:
            continue

        from_node = seg["FROM_INTERSECTION_ID"]
        to_node = seg["TO_INTERSECTION_ID"]
        try:
            if from_node is None or to_node is None or np.isnan(from_node) or np.isnan(to_node):
                continue
        except (TypeError, ValueError):
            continue

        # First segment to define a node wins; they should be very close anyway.
        # An intersection is shared by multiple segments — all converge at the same point.
        if from_node not in node_coords:
            node_coords[from_node] = start_pt
        if to_node not in node_coords:
            node_coords[to_node] = end_pt

    logger.info("Derived geometry for %d intersection nodes.", len(node_coords))
    return node_coords


def snap_to_graph(
    user_point: Point,
    node_coords: Dict[Hashable, Point],
    max_distance_m: float = 300.0,
) -> Hashable:
    """
    Snap a user location to the nearest graph node within a maximum distance.

    PHASE 2 — SNAP TO GRAPH
    The user's origin and destination arrive as GPS coordinates (lat/lon).
    The graph only understands intersection IDs (node numbers).
    This function bridges that gap by finding the nearest intersection node
    to the user's point.

    If nothing is within 300 m, the request is rejected — this prevents routing
    to/from locations outside the Toronto road network coverage area.

    Distance approximation:
        Uses degrees × 111,000 to convert to meters (flat-earth approximation).
        Accurate enough for city-scale snap — error is <1% within Toronto.
    """
    if not node_coords:
        raise ValueError("No node coordinates available to snap to.")

    closest_node: Hashable | None = None
    min_dist_m = float("inf")

    for node_id, node_point in node_coords.items():
        # Convert degree distance to approximate meters (1° ≈ 111 km at Toronto's latitude)
        dist_deg = user_point.distance(node_point)
        dist_m = dist_deg * 111_000.0
        if dist_m < max_distance_m and dist_m < min_dist_m:
            min_dist_m = dist_m
            closest_node = node_id

    if closest_node is None:
        raise ValueError(f"No graph node found within {max_distance_m:.1f}m of user location.")

    return closest_node


