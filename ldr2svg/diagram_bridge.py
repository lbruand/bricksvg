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
) -> tuple[list[Piece], list[tuple], list[dict], list[tuple[str, list[Piece]]]]:
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
    piece_groups
        Ordered list of ``(group_name, pieces)`` tuples suitable for wrapping
        in SVG ``<g>`` elements.  Clusters appear outermost-first; lone nodes
        are last under the name ``"lone"``.  Platform tiles come before the
        node brick within each cluster group.
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

    tile      = 20  # 20 LDU = 1 stud = one 1×1 tile width
    max_depth = max(cluster_depth.values(), default=0)

    # Direct parent of each cluster: the smallest cluster whose node set
    # strictly contains this one.  Used to clip child platforms so they never
    # spill into sibling territory.
    cluster_parent: dict[str, str | None] = {}
    for cl in cluster_objs:
        containing = [
            other["name"] for other in cluster_objs
            if cluster_node_sets[other["name"]] > cluster_node_sets[cl["name"]]
        ]
        cluster_parent[cl["name"]] = (
            min(containing, key=lambda n: len(cluster_node_sets[n]))
            if containing else None
        )

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

    # ── Step 2c: pre-compute sibling-bounded, parent-clipped platform extents ─
    # Build per-cluster node-position lists (only nodes with LDraw positions).
    cluster_node_xs: dict[str, list[int]] = {}
    cluster_node_zs: dict[str, list[int]] = {}
    for cl in cluster_objs:
        cl_ld = [g for g in cl["nodes"] if g in gvid_to_ld]
        if cl_ld:
            cluster_node_xs[cl["name"]] = [gvid_to_ld[g][0] for g in cl_ld]
            cluster_node_zs[cl["name"]] = [gvid_to_ld[g][1] for g in cl_ld]

    # Process outermost-first so parent extents exist when children are clipped.
    # Additionally cap each cluster's extent at the midpoint to the nearest
    # sibling node so that sibling platforms never overlap (e.g. Pods ⊄ MySQL).
    cluster_tile_extent: dict[str, tuple[int, int, int, int]] = {}
    for d in range(max_depth + 1):
        depth_clusters = [cl for cl in cluster_objs if cluster_depth[cl["name"]] == d]
        for cl in depth_clusters:
            name = cl["name"]
            if name not in cluster_node_xs:
                continue
            xs  = cluster_node_xs[name]
            zs  = cluster_node_zs[name]
            pad = tile * (max_depth - d + 2)
            x0 = math.floor((min(xs) - pad) / tile) * tile
            x1 = math.ceil ((max(xs) + pad) / tile) * tile
            z0 = math.floor((min(zs) - pad) / tile) * tile
            z1 = math.ceil ((max(zs) + pad) / tile) * tile

            # Cap at midpoint to each sibling (same parent, same depth).
            for sib in depth_clusters:
                sib_name = sib["name"]
                if sib_name == name or sib_name not in cluster_node_xs:
                    continue
                if cluster_parent[sib_name] != cluster_parent[name]:
                    continue
                sib_xs = cluster_node_xs[sib_name]
                sib_zs = cluster_node_zs[sib_name]
                if min(sib_xs) > max(xs):   # sibling is to the right
                    mid = math.floor((max(xs) + min(sib_xs)) / 2 / tile) * tile
                    x1 = min(x1, mid)
                elif max(sib_xs) < min(xs): # sibling is to the left
                    mid = math.ceil((max(sib_xs) + min(xs)) / 2 / tile) * tile
                    x0 = max(x0, mid)
                if min(sib_zs) > max(zs):   # sibling is further back
                    mid = math.floor((max(zs) + min(sib_zs)) / 2 / tile) * tile
                    z1 = min(z1, mid)
                elif max(sib_zs) < min(zs): # sibling is in front
                    mid = math.ceil((max(sib_zs) + min(zs)) / 2 / tile) * tile
                    z0 = max(z0, mid)

            # Clip to parent's tile extent.
            parent = cluster_parent[name]
            if parent and parent in cluster_tile_extent:
                px0, px1, pz0, pz1 = cluster_tile_extent[parent]
                x0, x1 = max(x0, px0), min(x1, px1)
                z0, z1 = max(z0, pz0), min(z1, pz1)

            cluster_tile_extent[name] = (x0, x1, z0, z1)

    # ── Step 2b: push lone nodes outside cluster platform footprints ───────
    # A lone node whose 2×2 brick footprint (center ±20 LDU) overlaps a cluster
    # platform is displaced to the nearer outside edge plus one-stud clearance.
    for gvid in list(gvid_to_ld):
        if gvid in node_cluster:
            continue  # clustered nodes sit on their platform intentionally
        ldx, ldz = gvid_to_ld[gvid]
        for px0, px1, pz0, pz1 in cluster_tile_extent.values():
            bx0, bx1 = ldx - 20, ldx + 20
            bz0, bz1 = ldz - 20, ldz + 20
            if not (bx0 < px1 and bx1 > px0 and bz0 < pz1 and bz1 > pz0):
                continue
            # Move to the nearest edge (+1-stud clearance so brick clears it)
            dx_left = ldx - px0
            dx_right = px1 - ldx
            dz_near = ldz - pz0
            dz_far  = pz1 - ldz
            if min(dx_left, dx_right) <= min(dz_near, dz_far):
                ldx = round(
                    (px0 - 40 if dx_left <= dx_right else px1 + 40) / 20
                ) * 20
            else:
                ldz = round(
                    (pz0 - 40 if dz_near <= dz_far else pz1 + 40) / 20
                ) * 20
            gvid_to_ld[gvid] = (ldx, ldz)
            break

    # ── Steps 3–4: node pieces ────────────────────────────────────────────
    # LDraw Y convention: floor = Y=0; pieces above floor have negative Y.
    # Piece pos[1] = top-face Y.
    #   Lone brick on floor:               pos_Y = -BRICK_H
    #   Brick on cluster platform (depth d): pos_Y = -(d+1)*PLATE_H - BRICK_H
    pieces: list[Piece] = []
    node_data: list[dict] = []
    cluster_node_bricks: dict[str, list[Piece]] = {}
    lone_node_bricks:   list[Piece] = []

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

        piece = Piece(part="3003", color=color, pos=pos, rot=np.eye(3))
        pieces.append(piece)
        node_data.append({
            "pos":       pos,
            "icon_path": obj.get("image") or None,
            "label":     obj.get("label", ""),
            "half_w":    20,   # 2×2 brick → half-side = 20 LDU
        })

        if in_cluster:
            cluster_node_bricks.setdefault(node_cluster[gvid], []).append(piece)
        else:
            lone_node_bricks.append(piece)

    # ── Step 5: cluster platform pieces (1×1 plates, stacked by depth) ────
    # 1×1 plates (3024, tile=20 LDU) give 1-stud snap precision so each
    # outer platform can show exactly one stud of its colour around its children.
    #
    # Depth-d cluster → plate pos_Y = -(d+1)*PLATE_H.
    # Outermost (d=0) sits on the floor; each inner level is one plate higher.
    #
    # Padding formula (tile=20 LDU = 1 stud):
    #   pad = tile × (max_depth − depth + 2)
    # → deepest cluster:  pad = 2×tile = 40 LDU  (brick half 20 + 1 stud border)
    # → each outer level adds one stud, so the border between adjacent levels
    #   is exactly 1 stud.
    cluster_platform_tiles: dict[str, list[Piece]] = {}

    for cl in cluster_objs:
        if cl["name"] not in cluster_tile_extent:
            continue
        x0s, x1s, z0s, z1s = cluster_tile_extent[cl["name"]]

        depth   = cluster_depth[cl["name"]]
        plate_y = float(-(depth + 1) * _PLATE_H_LDU)
        cl_c    = cluster_color.get(cl["name"], 15)

        x = x0s
        while x < x1s:
            z = z0s
            while z < z1s:
                # Piece origin = centre of 20×20 LDU tile (1×1 plate)
                pos = np.array([float(x + tile // 2), plate_y, float(z + tile // 2)])
                piece = Piece(part="3024", color=cl_c, pos=pos, rot=np.eye(3))
                pieces.append(piece)
                cluster_platform_tiles.setdefault(cl["name"], []).append(piece)
                z += tile
            x += tile

    # ── Step 6: piece_groups — ordered list of (name, pieces) for <g> tags ─
    # Outermost cluster first so inner-cluster tiles render on top in the SVG.
    # Within each group: platform tiles first, then the node brick.
    piece_groups: list[tuple[str, list[Piece]]] = []
    for d in range(max_depth + 1):
        for cl in cluster_objs:
            name = cl["name"]
            if cluster_depth[name] != d:
                continue
            group: list[Piece] = (cluster_platform_tiles.get(name, []) +
                                  cluster_node_bricks.get(name, []))
            if group:
                piece_groups.append((name, group))
    if lone_node_bricks:
        piece_groups.append(("lone", lone_node_bricks))

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

    return pieces, edge_positions, node_data, piece_groups
