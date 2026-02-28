"""diagram_compose.py — SVG composition for diagram2svg output."""

import io
import base64
import math

import numpy as np
import svgwrite
import svgwrite.container
from PIL import Image

from .compose import _project_pieces, _canvas_bounds, _build_defs
from .projection import project_ldraw, PX_PER_MM
from .grid import _grid_params, _grid_corner_sx_sy, _draw_isometric_grid
from .parts import Piece


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def _proj_canvas(
    pos_ld: np.ndarray,
    cx,
    cy,
) -> tuple[float, float]:
    """Project a LDraw position and convert to canvas (SVG) coordinates."""
    sx, sy, _ = project_ldraw(pos_ld)
    return cx(sx * PX_PER_MM), cy(sy * PX_PER_MM)


# ---------------------------------------------------------------------------
# Arrow defs + drawing
# ---------------------------------------------------------------------------

def _add_arrow_defs(dwg: svgwrite.Drawing) -> None:
    """Add a filled arrowhead marker to <defs>."""
    marker = svgwrite.container.Marker(
        id="arrow",
        insert=(5, 3),
        size=(6, 6),
        orient="auto",
    )
    marker.add(dwg.polygon(points=[(0, 0), (6, 3), (0, 6)], fill="#666"))
    dwg.defs.add(marker)


def _draw_floor_arrows(
    dwg: svgwrite.Drawing,
    arrows: list[tuple],
    cx,
    cy,
    shorten_px: float = 25.0,
) -> None:
    """Draw directed floor arrows connecting nodes."""
    grp = dwg.g(id="arrows")
    for from_pos, to_pos in arrows:
        x0, y0 = _proj_canvas(from_pos, cx, cy)
        x1, y1 = _proj_canvas(to_pos,   cx, cy)
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        # Shorten at the head end to avoid overlapping the destination brick
        x1s = x1 - ux * shorten_px
        y1s = y1 - uy * shorten_px
        line = dwg.line(
            start=(f"{x0:.1f}", f"{y0:.1f}"),
            end=(f"{x1s:.1f}", f"{y1s:.1f}"),
            stroke="#888",
            stroke_width="1.5",
        )
        line.attribs["marker-end"] = "url(#arrow)"
        grp.add(line)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Icon overlay
# ---------------------------------------------------------------------------

def _draw_icons(
    dwg: svgwrite.Drawing,
    node_data: list[dict],
    cx,
    cy,
) -> None:
    """Embed provider icons on the top face of each node brick.

    Each icon is affine-mapped from a W×H rectangle to the isometric top-face
    parallelogram using an SVG ``matrix(a,b,c,d,e,f)`` transform.
    """
    grp = dwg.g(id="icons")
    for nd in node_data:
        icon_path = nd.get("icon_path")
        if not icon_path:
            continue
        try:
            icon = Image.open(icon_path).convert("RGBA")
        except Exception:
            continue
        W, H = icon.size
        if W == 0 or H == 0:
            continue

        pos  = nd["pos"]
        hw   = nd["half_w"]
        ldx  = float(pos[0])
        top_y = float(pos[1])   # top-face Y in LDraw
        ldz  = float(pos[2])

        # Top-face corners in LDraw (Y = top of brick)
        A = np.array([ldx - hw, top_y, ldz - hw])   # back-left
        B = np.array([ldx + hw, top_y, ldz - hw])   # back-right
        D = np.array([ldx - hw, top_y, ldz + hw])   # front-left

        ax, ay = _proj_canvas(A, cx, cy)
        bx, by = _proj_canvas(B, cx, cy)
        dx, dy = _proj_canvas(D, cx, cy)

        # Affine: image (0,0)→A, (W,0)→B, (0,H)→D
        ma = (bx - ax) / W
        mb = (by - ay) / W
        mc = (dx - ax) / H
        md = (dy - ay) / H
        me = ax
        mf = ay

        buf = io.BytesIO()
        icon.save(buf, format="PNG")
        uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        img_el = dwg.image(uri, insert=("0px", "0px"), size=(f"{W}px", f"{H}px"))
        img_el.attribs["preserveAspectRatio"] = "none"
        img_el.attribs["transform"] = (
            f"matrix({ma:.6f},{mb:.6f},{mc:.6f},{md:.6f},{me:.2f},{mf:.2f})"
        )
        grp.add(img_el)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Isometric labels
# ---------------------------------------------------------------------------

def _draw_labels(
    dwg: svgwrite.Drawing,
    node_data: list[dict],
    cx,
    cy,
) -> None:
    """Draw isometric-skewed text labels at the front-floor foot of each brick."""
    grp = dwg.g(id="labels")
    for nd in node_data:
        label = nd.get("label", "")
        if not label:
            continue
        pos = nd["pos"]
        hw  = nd["half_w"]
        ldx = float(pos[0])
        ldz = float(pos[2])

        # Anchor: front-floor corner of the brick (at floor Y=0)
        anchor = np.array([ldx, 0.0, ldz + hw])
        lx, ly = _proj_canvas(anchor, cx, cy)

        text_el = dwg.text(
            label,
            insert=(0, 0),
            font_size="9",
            fill="#333",
            text_anchor="middle",
        )
        text_el.attribs["transform"] = (
            f"translate({lx:.1f},{ly:.1f}) scale(1,0.5) rotate(-30)"
        )
        grp.add(text_el)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

def compose_diagram_svg(
    renders: dict,
    output: str,
    arrows: list[tuple],
    node_data: list[dict],
    padding: int = 60,
) -> None:
    """Compose the final LEGO diagram SVG.

    Layer order:
    1. Background rect
    2. Isometric floor grid
    3. Floor arrows  (before bricks → appear as ground markings)
    4. Brick ``<use>`` elements (back-to-front)
    5. Icons on top faces
    6. Isometric labels
    """
    pngs: list[tuple[Piece, int, int, float, float]] = [
        (piece, *img.size, ax, ay)
        for img, ax, ay, piece_list in renders.values()
        for piece in piece_list
    ]
    projected = _project_pieces(pngs)

    grid = _grid_params(renders)
    grid_corners = _grid_corner_sx_sy(*grid) if grid else []

    # Include floor-arrow endpoints in the canvas bounds calculation
    extra_pts: list[tuple[float, float]] = []
    for from_pos, to_pos in arrows:
        for p in (from_pos, to_pos):
            sx, sy, _ = project_ldraw(p)
            extra_pts.append((sx * PX_PER_MM, sy * PX_PER_MM))

    W, H, min_x, min_y = _canvas_bounds(
        projected, padding, extra_sx_sy=grid_corners + extra_pts
    )

    def cx(val: float) -> float: return val - min_x + padding
    def cy(val: float) -> float: return val - min_y + padding

    dwg = svgwrite.Drawing(output, size=(f"{W}px", f"{H}px"))

    # 1. Background
    dwg.add(dwg.rect((0, 0), ("100%", "100%"), fill="#f8f8f0"))

    # 2. Grid
    if grid:
        fy, gx0, gx1, gz0, gz1 = grid
        _draw_isometric_grid(dwg, fy, gx0, gx1, gz0, gz1, cx, cy)

    # Arrow marker in <defs>
    _add_arrow_defs(dwg)

    # 3. Floor arrows
    _draw_floor_arrows(dwg, arrows, cx, cy)

    # 4. Brick images in defs + placed instances (back-to-front)
    defs = _build_defs(dwg, renders)
    for sx_px, sy_px, _, _, _, _, label in projected:
        def_id, ax, ay = defs[label]
        dwg.add(dwg.use(
            f"#{def_id}",
            insert=(f"{cx(sx_px) - ax:.1f}px", f"{cy(sy_px) - ay:.1f}px"),
        ))

    # 5. Icons on top faces
    _draw_icons(dwg, node_data, cx, cy)

    # 6. Labels
    _draw_labels(dwg, node_data, cx, cy)

    dwg.save(pretty=True)
    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
