# CLAUDE.md — ldr2svg project guide

## What this project does

Two rendering pipelines, both producing isometric LEGO SVGs:

1. **LDR → SVG** (`ldr2svg scene.ldr`): parse a LeoCAD/LDraw `.ldr` file, render each
   piece to a PNG via OpenSCAD + LEGO.scad, compose into a single SVG.

2. **diagrams → SVG** (`scripts/diagram2svg.py diagram.py`): run a
   [diagrams-library](https://diagrams.mingrammer.com/) Python script, capture the
   graphviz layout, map nodes → LEGO bricks and clusters → platform tiles, render with
   the same OpenSCAD pipeline, compose with floor arrows, icons, and labels.

---

## Coordinate systems

### LDraw (`.ldr` files, `Piece.pos`)
- X right, Y **down**, Z toward viewer
- Units: LDU (1 LDU = 0.4 mm; 1 stud = 8 mm = 20 LDU, 1 brick height = 24 LDU, 1 plate = 8 LDU)
- Floor is at Y = 0; pieces above floor have **negative Y**

### OpenSCAD (`LEGO.scad`)
- X right, Y forward, Z **up**
- Units: mm
- Transform `_T` in `projection.py` maps LDraw → OpenSCAD: `os = _T @ ld * 0.4`

### Screen / SVG
- X right, Y **down** (SVG convention)
- `project_ldraw(pos_ld)` returns `(screen_x, screen_y, depth)` — screen_y increases downward
- `PX_PER_MM` converts OpenSCAD mm to pixels

---

## Camera

True isometric: camera along `(1,1,1)/√3`.
- `CAMERA_RX = arccos(1/√3) ≈ 54.74°`  (OpenSCAD `rx` from top)
- `CAMERA_RZ = 45°` (spin)
- Horizontal LDraw axes project at exactly ±30° on screen
- `CAMERA_D = 300 mm`, `IMG_PX = 800`, `FOV = 22.5°`

---

## Key files

| File | Purpose |
|------|---------|
| `ldr2svg/parts.py` | `Piece` dataclass, `PART_MAP`, LDraw colour table, `.ldr` parser |
| `ldr2svg/projection.py` | LDraw↔OpenSCAD transform, `project_ldraw`, `PX_PER_MM` |
| `ldr2svg/scad.py` | `.scad` generation (`make_scad`), OpenSCAD render, background removal |
| `ldr2svg/ldr2png_svg.py` | `build_pngs` (dedup + render loop), `main` CLI entry point |
| `ldr2svg/compose.py` | `compose_svg`, `_project_pieces`, `_canvas_bounds`, `_build_defs` |
| `ldr2svg/grid.py` | Isometric floor grid (SVG lines) |
| `ldr2svg/diagram_bridge.py` | Extract graphviz layout → LDraw scene (`extract_graph`, `build_ldr_scene` + step functions) |
| `ldr2svg/diagram_compose.py` | `compose_diagram_svg` (extends compose with arrows, icons, labels) |
| `scripts/diagram2svg.py` | CLI: orchestrates diagram bridge → build_pngs → compose_diagram_svg |

---

## Pipeline: LDR → SVG

```
parse_ldr(path)          → list[Piece]
build_pngs(pieces, dir)  → renders dict  (label → (img, ax, ay, piece_list))
compose_svg(renders, out) → SVG file
```

`renders` key is a human-readable label (part + color + rotation) used to deduplicate:
identical-looking pieces share one `<image>` in `<defs>`, referenced via `<use>`.

## Pipeline: diagrams → SVG

```
extract_graph(script)           → graphviz JSON (positions in points)
build_ldr_scene(graph)          → (pieces, edge_positions, node_data, piece_groups)
build_pngs(pieces, dir)         → renders dict
compose_diagram_svg(renders, …) → SVG file
```

### `build_ldr_scene` step functions (in order)

| Function | Input | Output |
|----------|-------|--------|
| `_parse_objects` | graph | `node_objs, cluster_objs` |
| `_compute_cluster_metadata` | node_objs, cluster_objs | `node_cluster, cluster_color, cluster_depth, cluster_parent` |
| `_layout_positions` | node_objs | `gvid_to_ld` (snapped LDraw X/Z per gvid) |
| `_compute_platform_extents` | cluster_objs, gvid_to_ld, cluster_depth, cluster_parent | `cluster_tile_extent` |
| `_displace_lone_nodes` | gvid_to_ld, node_cluster, cluster_tile_extent | *(mutates gvid_to_ld)* |
| `_build_node_pieces` | node_objs, gvid_to_ld, node_cluster, cluster_color, cluster_depth | pieces, node_data, cluster_node_bricks, lone_node_bricks |
| `_build_platform_pieces` | cluster_objs, cluster_tile_extent, cluster_depth, cluster_color | pieces, cluster_platform_tiles |
| `_assemble_piece_groups` | cluster_objs, cluster_depth, cluster_node_bricks, cluster_platform_tiles, lone_node_bricks | `piece_groups` |
| `_build_edge_positions` | edges, gvid_to_ld | `edge_positions` |

### LDraw heights for diagram pieces
- Platform tile (3024, 1×1 plate): `_PLATE_H_LDU = 8`
- Node brick (3003, 2×2 brick): `_BRICK_H_LDU = 24`
- Cluster at depth `d`: plate Y = `-(d+1)*8`, node brick Y = `-(d+1)*8 - 24`
- Outermost cluster depth = 0; each nested level adds 1

### SVG layer order in `compose_diagram_svg`
1. Background rect
2. Isometric floor grid
3. Floor arrows (beneath bricks)
4. Brick `<use>` elements grouped by cluster in `<g id="cluster_…">` (back-to-front)
5. Icons (affine-mapped to brick top face)
6. Isometric labels (skewed text at brick front-floor foot)

---

## Testing

```bash
uv run pytest test/ -m "not slow"   # fast (no OpenSCAD)
xvfb-run uv run pytest test/ -v     # full (needs openscad + xvfb)
uv run ruff check ldr2svg/ test/
uv run ty check ldr2svg/
```

Tests are organised in classes, one class per function or logical group.
Step functions in `diagram_bridge.py` each have a corresponding `Test<StepName>` class.

---

## Coding style

- **Prefer list/dict/set comprehensions and functional constructs** (`map`, `filter`,
  generator expressions, `sorted`, `dict.fromkeys`) over imperative loops.
- **Avoid `continue` and `break`**. Restructure the logic using `if`/`else`, filtering
  comprehensions, `next(… for …)`, or early-returning helper functions instead.
- Keep functions small and focused on one thing. `build_ldr_scene` is an orchestrator;
  each step lives in its own private function with a minimal interface.
- Prefer `dict` / `set` lookups over linear searches when the key is known.
- Use `math.floor` / `math.ceil` explicitly when snapping to a grid; avoid ambiguous
  integer division for negative values.
