"""
Graph-based multi-curve extraction from skeleton images.

Converts a 1-pixel skeleton into an adjacency graph, finds junctions
and endpoints, compresses chains into a segment graph, then uses
k-best path search with smoothness / curvature scoring to extract
multiple non-overlapping curves.

This module is used by ``bw_pipeline.extract_bw_curves`` when
``config.use_graph_extraction`` is enabled.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from core.config import BWPipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# 8-connectivity offsets (dy, dx)
_NBRS_8 = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]


# ====================================================================
# 1. Build skeleton graph
# ====================================================================

def build_skeleton_graph(
    skeleton: np.ndarray,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Build adjacency list from skeleton using 8-connectivity.

    Parameters
    ----------
    skeleton : (H, W) bool array

    Returns
    -------
    dict mapping (x, y) -> list of (x, y) neighbors
    """
    h, w = skeleton.shape
    ys, xs = np.where(skeleton)
    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for x, y in pixel_set:
        neighbors: List[Tuple[int, int]] = []
        for dy, dx in _NBRS_8:
            nx, ny = x + dx, y + dy
            if (nx, ny) in pixel_set:
                neighbors.append((nx, ny))
        graph[(x, y)] = neighbors

    return graph


# ====================================================================
# 2. Classify nodes
# ====================================================================

def classify_nodes(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Classify graph nodes into endpoints, junctions, and regular.

    Returns
    -------
    (endpoints, junctions, regular)
        - endpoints : degree == 1
        - junctions : degree >= 3
        - regular   : degree == 2
    """
    endpoints: Set[Tuple[int, int]] = set()
    junctions: Set[Tuple[int, int]] = set()
    regular: Set[Tuple[int, int]] = set()

    for node, nbrs in graph.items():
        deg = len(nbrs)
        if deg == 1:
            endpoints.add(node)
        elif deg >= 3:
            junctions.add(node)
        else:
            regular.add(node)

    return endpoints, junctions, regular


# ====================================================================
# 3. Segment edge (compressed graph edge)
# ====================================================================

class SegmentEdge:
    """An edge in the compressed skeleton graph.

    Represents a chain of skeleton pixels between two special nodes
    (endpoints or junctions).
    """
    __slots__ = (
        "node_a", "node_b", "pixels", "length",
        "curvature", "cost", "x_span", "y_span",
    )

    def __init__(
        self,
        node_a: Tuple[int, int],
        node_b: Tuple[int, int],
        pixels: List[Tuple[int, int]],
    ):
        self.node_a = node_a
        self.node_b = node_b
        self.pixels = pixels
        self.length = len(pixels)
        self.curvature = _compute_curvature(pixels)
        if pixels:
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            self.x_span = max(xs) - min(xs)
            self.y_span = max(ys) - min(ys)
        else:
            self.x_span = 0
            self.y_span = 0
        self.cost = 0.0  # computed later by compute_edge_costs


# ====================================================================
# 4. Compress skeleton into segment graph
# ====================================================================

def compress_skeleton(
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    endpoints: Set[Tuple[int, int]],
    junctions: Set[Tuple[int, int]],
) -> Tuple[Dict[Tuple[int, int], List[SegmentEdge]], List[SegmentEdge]]:
    """Compress chains between special nodes into segment edges.

    Each segment is a contiguous chain of pixels between two endpoints
    or junctions.  The resulting compressed graph has a much smaller
    node count than the pixel-level graph.

    Returns
    -------
    (adjacency, all_edges)
        adjacency : dict mapping special_node -> list of SegmentEdge
        all_edges : flat list of all SegmentEdge objects
    """
    special = set(endpoints) | set(junctions)
    adjacency: Dict[Tuple[int, int], List[SegmentEdge]] = defaultdict(list)
    all_edges: List[SegmentEdge] = []
    visited_starts: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()

    for start_node in special:
        if start_node not in graph:
            continue
        for first_nbr in graph[start_node]:
            if (start_node, first_nbr) in visited_starts:
                continue

            # Walk along the chain until the next special node or dead end
            chain = [start_node, first_nbr]
            visited_on_chain = {start_node, first_nbr}
            current = first_nbr

            while current not in special or current == first_nbr:
                if current in special and current != first_nbr:
                    break
                next_node = None
                for nbr in graph.get(current, []):
                    if nbr not in visited_on_chain:
                        next_node = nbr
                        break
                if next_node is None:
                    break
                chain.append(next_node)
                visited_on_chain.add(next_node)
                current = next_node

            end_node = chain[-1]

            # Accept edge if end is a special node or a dead-end
            if end_node in special or len(graph.get(end_node, [])) <= 1:
                if end_node not in special:
                    endpoints.add(end_node)
                    special.add(end_node)

                edge = SegmentEdge(start_node, end_node, chain)
                adjacency[start_node].append(edge)
                adjacency[end_node].append(edge)
                all_edges.append(edge)

                visited_starts.add((start_node, first_nbr))
                if len(chain) >= 2:
                    visited_starts.add((end_node, chain[-2]))

    return dict(adjacency), all_edges


# ====================================================================
# 5. Curvature computation
# ====================================================================

def _compute_curvature(pixels: List[Tuple[int, int]]) -> float:
    """Total curvature (sum of angle changes) along a pixel chain.

    Returns total curvature in radians.  Sub-samples long chains for
    speed.
    """
    if len(pixels) < 3:
        return 0.0

    step = max(1, len(pixels) // 60)
    sampled = pixels[::step]

    total = 0.0
    for i in range(1, len(sampled) - 1):
        x0, y0 = sampled[i - 1]
        x1, y1 = sampled[i]
        x2, y2 = sampled[i + 1]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1

        len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue

        cross = dx1 * dy2 - dy1 * dx2
        dot = dx1 * dx2 + dy1 * dy2
        total += abs(math.atan2(cross, dot))

    return total


# ====================================================================
# 6. Edge cost computation
# ====================================================================

def compute_edge_costs(
    edges: List[SegmentEdge],
    config: BWPipelineConfig,
) -> None:
    """Assign traversal cost to each segment edge.

    cost = length + curvature_penalty * curvature
    """
    for edge in edges:
        curv_cost = config.curvature_penalty * edge.curvature
        edge.cost = float(edge.length) + curv_cost


# ====================================================================
# 7. K-best path search (modified Dijkstra with diversity)
# ====================================================================

def find_k_shortest_paths(
    adjacency: Dict[Tuple[int, int], List[SegmentEdge]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    k: int,
    config: BWPipelineConfig,
) -> List[Tuple[float, List[Tuple[int, int]], List[SegmentEdge]]]:
    """Find *k* diverse shortest paths between *start* and *end*.

    Uses repeated Dijkstra with increasing penalties on previously used
    edges to encourage diversity without the full cost of Yen's
    algorithm (the compressed graph is small enough for this to work).

    Returns
    -------
    list of (cost, pixel_path, edge_list) sorted by cost
    """
    results: List[Tuple[float, List[Tuple[int, int]], List[SegmentEdge]]] = []
    used_edge_ids: List[Set[int]] = []
    _counter = 0  # tie-breaking for heapq

    for _iteration in range(k):
        # Extra cost for edges already used by previous results
        extra_cost: Dict[int, float] = {}
        for prev_set in used_edge_ids:
            for eid in prev_set:
                extra_cost[eid] = extra_cost.get(eid, 0) + config.shared_pixel_penalty * 10

        # Dijkstra on the compressed graph
        pq: List[Tuple[float, int, Tuple[int, int], List[SegmentEdge]]] = []
        _counter += 1
        heapq.heappush(pq, (0.0, _counter, start, []))
        visited: Dict[Tuple[int, int], float] = {}
        best_path: Optional[Tuple[float, List[SegmentEdge]]] = None

        while pq:
            cost, _cnt, node, edge_path = heapq.heappop(pq)
            if node in visited and visited[node] <= cost:
                continue
            visited[node] = cost

            if node == end:
                best_path = (cost, edge_path)
                break

            for edge in adjacency.get(node, []):
                other = edge.node_b if edge.node_a == node else edge.node_a
                if other in visited:
                    continue

                eid = id(edge)
                edge_extra = extra_cost.get(eid, 0.0)
                junction_cost = (
                    config.junction_penalty
                    if len(adjacency.get(other, [])) >= 3
                    else 0.0
                )

                new_cost = cost + edge.cost + edge_extra + junction_cost
                if other not in visited or visited[other] > new_cost:
                    _counter += 1
                    heapq.heappush(pq, (new_cost, _counter, other, edge_path + [edge]))

        if best_path is None:
            break

        bp_cost, bp_edges = best_path
        pixel_path = _edges_to_pixel_path(bp_edges, start, end)
        results.append((bp_cost, pixel_path, bp_edges))

        used_edge_ids.append({id(e) for e in bp_edges})

    return results


# ====================================================================
# 8. Convert edge list → pixel polyline
# ====================================================================

def _edges_to_pixel_path(
    edges: List[SegmentEdge],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Reconstruct an ordered pixel polyline from segment edges."""
    if not edges:
        return [start, end] if start != end else [start]

    path: List[Tuple[int, int]] = []
    current = start

    for edge in edges:
        if edge.node_a == current:
            pixels = edge.pixels
        elif edge.node_b == current:
            pixels = list(reversed(edge.pixels))
        else:
            # Proximity fallback
            da = abs(edge.node_a[0] - current[0]) + abs(edge.node_a[1] - current[1])
            db = abs(edge.node_b[0] - current[0]) + abs(edge.node_b[1] - current[1])
            pixels = edge.pixels if da <= db else list(reversed(edge.pixels))

        # Skip first pixel if it duplicates the last of path
        start_idx = 1 if path and pixels and pixels[0] == path[-1] else 0
        path.extend(pixels[start_idx:])
        current = pixels[-1] if pixels else current

    return path


# ====================================================================
# 9. Non-overlapping curve selection
# ====================================================================

def select_non_overlapping_curves(
    candidates: List[Tuple[float, List[Tuple[int, int]], Any]],
    max_curves: int,
    min_path_length: int,
    shared_pixel_penalty: float,
) -> List[List[Tuple[int, int]]]:
    """Greedily select up to *max_curves* non-overlapping paths.

    Algorithm:
      1. Filter by minimum length.
      2. Sort by cost / x_span (prefer cheap, long paths).
      3. Greedily pick the best path and penalise candidates that
         share pixels with it.
    """
    if not candidates:
        return []

    valid = [
        (cost, pixels, extra)
        for cost, pixels, extra in candidates
        if len(pixels) >= min_path_length
    ]
    if not valid:
        return []

    # Sort by cost normalised by x-span (prefer cost-effective spanning)
    def _sort_key(c: Tuple[float, List[Tuple[int, int]], Any]) -> float:
        cost, px, _ = c
        if not px:
            return float("inf")
        xs = [p[0] for p in px]
        x_span = max(xs) - min(xs) + 1
        return cost / max(x_span, 1)

    valid.sort(key=_sort_key)

    selected: List[List[Tuple[int, int]]] = []
    used_pixels: Set[Tuple[int, int]] = set()

    for cost, pixels, _ in valid:
        if len(selected) >= max_curves:
            break
        pixel_set = set(pixels)
        overlap = len(pixel_set & used_pixels)
        overlap_ratio = overlap / max(len(pixel_set), 1)

        if overlap_ratio > 0.3:
            continue

        selected.append(pixels)
        used_pixels |= pixel_set

    return selected


# ====================================================================
# 10. Snap point to nearest graph node
# ====================================================================

def _snap_to_node(
    point: Tuple[int, int],
    graph: Dict[Tuple[int, int], List[Tuple[int, int]]],
    radius: int = 20,
) -> Optional[Tuple[int, int]]:
    """Snap a point to the nearest graph node within *radius*."""
    if point in graph:
        return point

    best: Optional[Tuple[int, int]] = None
    best_dist = float("inf")
    px, py = point

    for node in graph:
        dx = node[0] - px
        dy = node[1] - py
        dist = dx * dx + dy * dy
        if dist < best_dist and dist <= radius * radius:
            best_dist = dist
            best = node

    return best


# ====================================================================
# 11. Main entry point
# ====================================================================

def extract_curves_graph(
    skeleton: np.ndarray,
    binary: np.ndarray,
    num_curves: int,
    config: BWPipelineConfig,
    *,
    anchors: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[str, Any]]:
    """Extract multiple curves from skeleton using graph-based approach.

    Parameters
    ----------
    skeleton : (H, W) bool
        Skeletonised binary image (plot-area crop).
    binary : (H, W) bool
        Cleaned binary image before skeletonisation.
    num_curves : int
        Target number of curves to extract.
    config : BWPipelineConfig
    anchors : optional
        Start/end anchors in **local** (crop) coordinates.

    Returns
    -------
    (curves_dict, debug_info)
        curves_dict : Dict[int, list of (x, y)] in local coords
        debug_info  : dict with endpoints, junctions, segment info
    """
    logger.info("extract_curves_graph: building skeleton graph")

    graph = build_skeleton_graph(skeleton)
    if not graph:
        return {}, {"error": "empty skeleton graph"}

    endpoints, junctions, regular = classify_nodes(graph)
    logger.info(
        "extract_curves_graph: %d endpoints, %d junctions, %d regular",
        len(endpoints), len(junctions), len(regular),
    )

    # Compress into segment graph
    adjacency, all_edges = compress_skeleton(graph, endpoints, junctions)
    compute_edge_costs(all_edges, config)
    logger.info(
        "extract_curves_graph: %d segments, %d special nodes",
        len(all_edges), len(adjacency),
    )

    debug_info: Dict[str, Any] = {
        "num_endpoints": len(endpoints),
        "num_junctions": len(junctions),
        "num_segments": len(all_edges),
        "endpoints": list(endpoints)[:50],
        "junctions": list(junctions)[:50],
    }

    h, w = skeleton.shape
    all_candidates: List[Tuple[float, List[Tuple[int, int]], List[SegmentEdge]]] = []

    # ── Anchor-guided path search ──
    if anchors:
        for start, end in anchors:
            start_node = _snap_to_node(start, graph)
            end_node = _snap_to_node(end, graph)
            if start_node and end_node and start_node != end_node:
                paths = find_k_shortest_paths(
                    adjacency, start_node, end_node, config.k_paths, config,
                )
                all_candidates.extend(paths)
    else:
        # ── Auto-detect: pair left-side endpoints with right-side endpoints ──
        left_eps = sorted(
            [e for e in endpoints if e[0] < w * 0.25], key=lambda p: p[1],
        )
        right_eps = sorted(
            [e for e in endpoints if e[0] > w * 0.75], key=lambda p: p[1],
        )

        # Fallback: split all endpoints into left/right thirds
        if len(left_eps) < 1 or len(right_eps) < 1:
            sorted_eps = sorted(endpoints, key=lambda p: p[0])
            n = len(sorted_eps)
            if n >= 2:
                left_eps = sorted_eps[: max(1, n // 3)]
                right_eps = sorted_eps[max(1, 2 * n // 3) :]

        # Cap pair count to avoid combinatorial blow-up
        max_pairs = min(len(left_eps) * len(right_eps), 50)
        pair_count = 0

        for le in left_eps:
            for re in right_eps:
                if pair_count >= max_pairs:
                    break
                x_span = abs(re[0] - le[0])
                if x_span < w * 0.3:
                    continue
                paths = find_k_shortest_paths(
                    adjacency, le, re, config.k_paths, config,
                )
                all_candidates.extend(paths)
                pair_count += 1

    logger.info("extract_curves_graph: %d candidate paths", len(all_candidates))

    # Select non-overlapping curves
    target = max(num_curves, 1)
    selected = select_non_overlapping_curves(
        all_candidates,
        min(target, config.max_curves),
        config.min_path_length_px,
        config.shared_pixel_penalty,
    )

    curves_dict: Dict[int, List[Tuple[int, int]]] = {}
    for i, pixels in enumerate(selected):
        curves_dict[i] = pixels

    debug_info["num_candidates"] = len(all_candidates)
    debug_info["num_selected"] = len(selected)

    logger.info("extract_curves_graph: selected %d curves", len(curves_dict))
    return curves_dict, debug_info
