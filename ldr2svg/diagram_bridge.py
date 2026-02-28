"""diagram_bridge.py — Extract diagrams-library graph and build LDraw scene."""

import json
import math
import runpy
import statistics

import numpy as np
import diagrams

from .parts import Piece

# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------

def extract_graph(script_path: str) -> dict:
    """Run a diagrams script, capture the dot graph, return graphviz JSON layout.

    Monkey-patches ``diagrams.Diagram`` so the context manager captures the
    dot graph without writing any output files or launching a viewer.
    """
    _captured: list = []
    _OrigDiagram = diagrams.Diagram

    class _CaptureDiagram(_OrigDiagram):
        def __exit__(self, *args):
            _captured.append(self.dot)
            # Skip super().__exit__() — no file output, no graphviz render

    setattr(diagrams, "Diagram", _CaptureDiagram)
    try:
        runpy.run_path(script_path, run_name="__main__")
    finally:
        setattr(diagrams, "Diagram", _OrigDiagram)

    if not _captured:
        raise RuntimeError(f"No Diagram context captured from {script_path!r}")

    dot = _captured[0]
    return json.loads(dot.pipe(format="json", engine="dot"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# LDraw color palette for clusters: blue, red, green, orange, light-blue, tan
_CLUSTER_PALETTE = [1, 4, 2, 25, 41, 22]

# LDraw heights in LDU (1 LDU = 0.4 mm; 1 stud = 8 mm = 20 LDU)
_PLATE_H_LDU = 8   # height of a 2×2 plate  (3022, height=1/3 brick)
_BRICK_H_LDU = 24  # height of a 2×2 brick  (3003, height=1   brick)


def _provider_color(image_path: str) -> int:
    """Derive a LDraw color from a diagrams provider icon path."""
    for key, color in [("k8s", 1), ("gcp", 4), ("aws", 25), ("azure", 41), ("onprem", 2)]:
        if key in image_path:
            return color
    return 7  # gray default


def _median_nn_dist(xs: list[float], ys: list[float]) -> float:
    """Median nearest-neighbor distance among a set of 2D points."""
    if len(xs) < 2:
        return 1.0
    dists = [
        min(math.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(len(xs)) if j != i)
        for i in range(len(xs))
    ]
    return statistics.median(dists)


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_ldr_scene(
    graph: dict,
) -> tuple[list[Piece], list[tuple], list[dict]]:
    """Parse graphviz JSON layout and build LDraw scene data.

    Parameters
    ----------
    graph:
        Dict returned by ``extract_graph()``.

    Returns
    -------
    pieces
        List of :class:`~ldr2svg.parts.Piece` objects (node bricks + platform
        tiles) ready to be rendered by :func:`~ldr2svg.ldr2png_svg.build_pngs`.
    edge_positions
        List of ``(from_pos, to_pos)`` numpy arrays in LDraw coordinates at
        floor level (Y = 0) — used to draw SVG floor arrows.
    node_data
        List of dicts, one per node::

            {
              "pos":       np.array([ldx, node_y, ldz]),  # LDraw world pos
              "icon_path": str | None,                    # absolute PNG path
              "label":     str,
              "half_w":    int,                           # half-side in LDU
            }
    """
    objects = graph.get("objects", [])
    edges   = graph.get("edges",   [])

    # ── Step 1: split node objects vs cluster subgraphs ───────────────────
    node_objs    = [o for o in objects if "pos" in o]
    cluster_objs = [o for o in objects if "nodes" in o]

    gvid_to_node = {o["_gvid"]: o for o in node_objs}

    # For each node, find the innermost cluster (fewest members) that contains it
    node_cluster: dict[int, str] = {}
    for gvid in gvid_to_node:
        best_name: str | None = None
        best_size = math.inf
        for cl in cluster_objs:
            if gvid in cl["nodes"] and len(cl["nodes"]) < best_size:
                best_name = cl["name"]
                best_size = len(cl["nodes"])
        if best_name is not None:
            node_cluster[gvid] = best_name

    # Assign colors to clusters in appearance order
    unique_clusters: list[str] = list(dict.fromkeys(
        node_cluster[gvid]
        for gvid in gvid_to_node
        if gvid in node_cluster
    ))
    cluster_color: dict[str, int] = {
        name: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)]
        for i, name in enumerate(unique_clusters)
    }

    # Nesting depth: number of clusters whose node set strictly contains this one.
    # depth=0 → outermost, depth=1 → one level in, etc.
    # Platforms are stacked at Y = -(depth+1) * PLATE_H so inner clusters appear raised.
    cluster_node_sets = {cl["name"]: set(cl["nodes"]) for cl in cluster_objs}
    cluster_depth: dict[str, int] = {
        cl["name"]: sum(
            1 for other in cluster_objs
            if cluster_node_sets[other["name"]] > cluster_node_sets[cl["name"]]
        )
        for cl in cluster_objs
    }

    # ── Step 2: normalize graphviz positions to LDraw grid ────────────────
    # Graphviz pos string is "x,y" in points (1 pt = 1/72 inch).
    # We scale so median nearest-neighbour distance ≥ 80 LDU (4 studs),
    # then snap to 20-LDU (1-stud) increments.
    gvx = [float(o["pos"].split(",")[0]) for o in node_objs]
    gvy = [float(o["pos"].split(",")[1]) for o in node_objs]
    nn_dist = _median_nn_dist(gvx, gvy)
    scale = 80.0 / nn_dist if nn_dist > 1e-6 else 1.0

    def snap(v: float) -> int:
        return round(v * scale / 20) * 20

    gvid_to_ld: dict[int, tuple[int, int]] = {}
    for obj in node_objs:
        gx = float(obj["pos"].split(",")[0])
        gy = float(obj["pos"].split(",")[1])
        gvid_to_ld[obj["_gvid"]] = (snap(gx), snap(-gy))  # negate Y: gv↑ → LDraw -Z

    # ── Steps 3–4: node pieces ────────────────────────────────────────────
    # LDraw Y convention: floor = Y=0; pieces above floor have negative Y.
    # Piece pos[1] = top-face Y.
    #   Lone brick on floor:               pos_Y = -BRICK_H
    #   Brick on cluster platform (depth d): pos_Y = -(d+1)*PLATE_H - BRICK_H
    pieces: list[Piece] = []
    node_data: list[dict] = []

    for obj in node_objs:
        gvid = obj["_gvid"]
        ldx, ldz = gvid_to_ld[gvid]
        in_cluster = gvid in node_cluster

        color = (cluster_color[node_cluster[gvid]] if in_cluster
                 else _provider_color(obj.get("image", "")))

        if in_cluster:
            depth = cluster_depth[node_cluster[gvid]]
            node_y = float(-(depth + 1) * _PLATE_H_LDU - _BRICK_H_LDU)
        else:
            node_y = float(-_BRICK_H_LDU)
        pos = np.array([float(ldx), node_y, float(ldz)])

        pieces.append(Piece(part="3003", color=color, pos=pos, rot=np.eye(3)))
        node_data.append({
            "pos":       pos,
            "icon_path": obj.get("image") or None,
            "label":     obj.get("label", ""),
            "half_w":    20,   # 2×2 brick → half-side = 20 LDU
        })

    # ── Step 5: cluster platform pieces (2×2 plates, stacked by depth) ────
    # Depth-d cluster → plate pos_Y = -(d+1)*PLATE_H.
    # Outermost (d=0) sits on the floor; each inner level is one plate higher.
    #
    # Padding grows with distance from the deepest cluster so that each outer
    # platform extends exactly one more stud beyond its immediate children:
    #   pad = tile * (max_depth − depth + 1)
    # → deepest: pad=40  (1 stud beyond brick edge)
    # → one level up: pad=80 (1 stud beyond inner platform edge)
    # → outermost: pad = tile * (max_depth + 1)
    tile      = 40  # 40 LDU = 2 studs = one 2×2 tile width
    max_depth = max(cluster_depth.values(), default=0)

    for cl in cluster_objs:
        cl_gvids = [g for g in cl["nodes"] if g in gvid_to_ld]
        if not cl_gvids:
            continue

        cl_xs = [gvid_to_ld[g][0] for g in cl_gvids]
        cl_zs = [gvid_to_ld[g][1] for g in cl_gvids]

        depth = cluster_depth[cl["name"]]
        pad   = tile * (max_depth - depth + 1)
        x0s = math.floor((min(cl_xs) - pad) / tile) * tile
        x1s = math.ceil ((max(cl_xs) + pad) / tile) * tile
        z0s = math.floor((min(cl_zs) - pad) / tile) * tile
        z1s = math.ceil ((max(cl_zs) + pad) / tile) * tile

        plate_y = float(-(depth + 1) * _PLATE_H_LDU)
        cl_c    = cluster_color.get(cl["name"], 15)

        x = x0s
        while x < x1s:
            z = z0s
            while z < z1s:
                # Piece origin = center of 40×40 LDU tile
                pos = np.array([float(x + tile // 2), plate_y, float(z + tile // 2)])
                pieces.append(Piece(part="3022", color=cl_c, pos=pos, rot=np.eye(3)))
                z += tile
            x += tile

    # ── Step 7: edge positions for floor arrows ───────────────────────────
    edge_positions: list[tuple] = []
    for e in edges:
        tail_gvid = e.get("tail")
        head_gvid = e.get("head")
        if tail_gvid in gvid_to_ld and head_gvid in gvid_to_ld:
            tlx, tlz = gvid_to_ld[tail_gvid]
            hlx, hlz = gvid_to_ld[head_gvid]
            edge_positions.append((
                np.array([float(tlx), 0.0, float(tlz)]),
                np.array([float(hlx), 0.0, float(hlz)]),
            ))

    return pieces, edge_positions, node_data
