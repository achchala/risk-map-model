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

from typing import Dict, Hashable, Iterable, List, Optional, Tuple

import logging

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

logger = logging.getLogger(__name__)


# Default speeds (km/h) by road class, used when no explicit speed limits exist.
DEFAULT_SPEEDS_KMH: Dict[str, float] = {
    "arterial": 50.0,
    "collector": 40.0,
    "local": 30.0,
    "minor_arterial": 45.0,
}


def estimate_travel_time_hours(
    segment_length_m: float,
    road_class: str,
    default_speeds_kmh: Dict[str, float] | None = None,
) -> float:
    """
    Estimate travel time (in hours) from length and road class.

    Units:
    - segment_length_m: meters
    - speed: km/h
    - return: hours
    """
    if default_speeds_kmh is None:
        default_speeds_kmh = DEFAULT_SPEEDS_KMH

    speed_kmh = default_speeds_kmh.get(str(road_class).lower(), 40.0)
    speed_ms = speed_kmh / 3.6
    if speed_ms <= 0:
        # Fallback in pathological case
        speed_ms = 10.0

    travel_time_seconds = segment_length_m / speed_ms
    return travel_time_seconds / 3600.0


def build_road_graph(road_network: gpd.GeoDataFrame) -> nx.DiGraph:
    """
    Build a directed graph from the road network.

    Nodes:
        intersection IDs (FROM_INTERSECTION_ID / TO_INTERSECTION_ID)

    Edges:
        attributes:
        - segment_id (CENTRELINE_ID)
        - geometry (LineString)
        - length_m (segment_length)
        - road_class (ROAD_CLASS)
        - travel_time_hours

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

    G = nx.DiGraph()

    for _, seg in road_network.iterrows():
        from_node = seg["FROM_INTERSECTION_ID"]
        to_node = seg["TO_INTERSECTION_ID"]
        segment_id = seg["CENTRELINE_ID"]
        length_m = float(seg["segment_length"])
        road_class = seg.get("ROAD_CLASS", "unknown")
        geom = seg.geometry

        if isinstance(geom, LineString):
            geometry = geom
        elif isinstance(geom, MultiLineString) and len(geom.geoms) > 0:
            # Merge all parts into one LineString for routing
            coords = []
            for g in geom.geoms:
                coords.extend(list(g.coords))
            if len(coords) < 2:
                continue
            geometry = LineString(coords)
        else:
            continue

        # Skip segments with invalid intersection IDs
        try:
            if from_node is None or to_node is None or np.isnan(from_node) or np.isnan(to_node):
                continue
        except (TypeError, ValueError):
            continue

        t_hours = estimate_travel_time_hours(length_m, str(road_class))

        G.add_edge(
            from_node,
            to_node,
            segment_id=segment_id,
            geometry=geometry,
            length_m=length_m,
            road_class=road_class,
            travel_time_hours=t_hours,
        )

        # Handle one-way vs two-way. If there's no clear one-way code,
        # default to bidirectional.
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
    default_lam_per_hour: Optional[float] = None,
    risk_multiplier: float = 1.0,
) -> None:
    """
    Annotate edges with expected crashes and combined cost.

    Parameters:
        G: graph with 'segment_id' and 'travel_time_hours' on each edge
        lambda_per_hour: mapping from segment_id to crash rate λ (crashes/hour)
        beta_hours_per_expected_crash: risk-avoidance coefficient. Higher values
            penalize high-λ segments more strongly in the combined weight.
        default_lam_per_hour: for segments not in lambda_per_hour, use this instead of 0.
            If None, uses 0 (unknown = zero risk). Use median of known λ to avoid
            biasing safer path toward segments with no data.
        risk_multiplier: scales the risk penalty (e.g. 1.35 for rain). Used so that
            weather/time conditions change route selection; default 1.0 = no scaling.

    Edge attributes added:
        - expected_crashes
        - normalized_risk
        - risk_penalty_hours
        - risk_weight_hours
    """
    default_lam = float(default_lam_per_hour) if default_lam_per_hour is not None else 0.0
    positive_lams = np.array(
        [float(v) for v in lambda_per_hour.values() if float(v) > 0.0],
        dtype=float,
    )
    # Use a typical non-zero λ as the network baseline so routing reacts to
    # relative risk, not just raw λ magnitudes (which are very small in practice).
    baseline_lam = (
        float(np.median(positive_lams))
        if positive_lams.size > 0
        else max(default_lam, 1e-12)
    )
    baseline_lam = max(baseline_lam, 1e-12)
    NORMALIZED_RISK_CAP = 1.5

    def _get_lam(seg_id, lam_dict):
        """Look up lambda with type normalization (int/np.int64/float key mismatch)."""
        if seg_id is None:
            return default_lam
        v = lam_dict.get(seg_id, None)
        if v is not None:
            return float(v)
        try:
            v = lam_dict.get(int(seg_id), None)
            if v is not None:
                return float(v)
        except (ValueError, TypeError):
            pass
        return default_lam

    for u, v, data in G.edges(data=True):
        seg_id = data.get("segment_id")
        travel_time = float(data.get("travel_time_hours", 0.0))
        lam = _get_lam(seg_id, lambda_per_hour)

        expected_crashes = lam * travel_time  # dimensionless expected count
        raw_normalized = float(np.log1p(max(lam, 0.0) / baseline_lam))
        normalized_risk = min(raw_normalized, NORMALIZED_RISK_CAP)
        risk_penalty_hours = (
            beta_hours_per_expected_crash * travel_time * normalized_risk * risk_multiplier
        )
        data["expected_crashes"] = expected_crashes
        data["normalized_risk"] = normalized_risk
        data["risk_penalty_hours"] = risk_penalty_hours
        data["risk_weight_hours"] = travel_time + risk_penalty_hours


def find_fastest_route(
    G: nx.DiGraph,
    start_node: Hashable,
    end_node: Hashable,
) -> List[Hashable]:
    """
    Find the fastest route (time-only) using Dijkstra on travel_time_hours.
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

    Assumes `apply_risk_to_edge_costs` has been called so that each edge
    has a 'risk_weight_hours' attribute.
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

    Assumes edges along the path already have 'expected_crashes' and
    'travel_time_hours' set.

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

    route_probability = 1.0 - float(np.exp(-total_expected))
    return {
        "expected_crashes": total_expected,
        "route_probability": route_probability,
        "total_travel_time_hours": total_time,
    }


def build_node_geometry(road_network: gpd.GeoDataFrame) -> Dict[Hashable, Point]:
    """
    Derive approximate node coordinates for each intersection ID from segment endpoints.

    Returns:
        dict: intersection_id -> Point (in EPSG:4326)
    """
    # Work in geographic CRS so distances can be approximated later
    roads_geo = road_network.to_crs("EPSG:4326")

    node_coords: Dict[Hashable, Point] = {}

    for _, seg in roads_geo.iterrows():
        geom = seg.geometry
        if isinstance(geom, LineString):
            start_pt = Point(geom.coords[0])
            end_pt = Point(geom.coords[-1])
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

    Distances are approximated by converting degrees to meters (~111km per degree),
    which is sufficient at city scale for snap-to-graph.
    """
    if not node_coords:
        raise ValueError("No node coordinates available to snap to.")

    closest_node: Hashable | None = None
    min_dist_m = float("inf")

    for node_id, node_point in node_coords.items():
        # Approximate meters using a simple WGS84 scale
        dist_deg = user_point.distance(node_point)
        dist_m = dist_deg * 111_000.0
        if dist_m < max_distance_m and dist_m < min_dist_m:
            min_dist_m = dist_m
            closest_node = node_id

    if closest_node is None:
        raise ValueError(f"No graph node found within {max_distance_m:.1f}m of user location.")

    return closest_node


