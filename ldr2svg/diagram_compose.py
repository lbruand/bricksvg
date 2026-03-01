"""diagram_compose.py — SVG composition for diagram2svg output."""

import io
import base64
import math

import numpy as np
import svgwrite
import svgwrite.container
from PIL import Image

from .compose import _project_pieces, _project_piece, _canvas_bounds, _build_defs
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


def _make_floor_arrow(
    dwg: svgwrite.Drawing,
    from_pos: np.ndarray,
    to_pos: np.ndarray,
    cx,
    cy,
    shorten_px: float,
):
    """Return a shortened SVG line for one arrow, or None if the endpoints coincide."""
    x0, y0 = _proj_canvas(from_pos, cx, cy)
    x1, y1 = _proj_canvas(to_pos,   cx, cy)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    ux, uy = dx / length, dy / length
    line = dwg.line(
        start=(f"{x0:.1f}", f"{y0:.1f}"),
        end=(f"{x1 - ux * shorten_px:.1f}", f"{y1 - uy * shorten_px:.1f}"),
        stroke="#888",
        stroke_width="1.5",
    )
    line.attribs["marker-end"] = "url(#arrow)"
    return line


def _draw_floor_arrows(
    dwg: svgwrite.Drawing,
    arrows: list[tuple],
    cx,
    cy,
    shorten_px: float = 25.0,
) -> None:
    """Draw directed floor arrows connecting nodes."""
    grp = dwg.g(id="arrows")
    lines = filter(None, (_make_floor_arrow(dwg, fp, tp, cx, cy, shorten_px)
                          for fp, tp in arrows))
    for line in lines:
        grp.add(line)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Icon overlay
# ---------------------------------------------------------------------------

def _load_icon(icon_path: str | None) -> Image.Image | None:
    """Load an icon image and return it, or None if missing, unreadable, or empty."""
    if not icon_path:
        return None
    try:
        icon = Image.open(icon_path).convert("RGBA")
    except Exception:
        return None
    W, H = icon.size
    return icon if W > 0 and H > 0 else None


def _icon_element(
    dwg: svgwrite.Drawing,
    icon: Image.Image,
    nd: dict,
    cx,
    cy,
):
    """Build the affine-transformed SVG <image> element for one node icon."""
    W, H = icon.size
    pos   = nd["pos"]
    hw    = nd["half_w"]
    ldx   = float(pos[0])
    top_y = float(pos[1])
    ldz   = float(pos[2])

    # Top-face corners in LDraw (Y = top of brick)
    A = np.array([ldx - hw, top_y, ldz - hw])   # back-left
    B = np.array([ldx + hw, top_y, ldz - hw])   # back-right
    D = np.array([ldx - hw, top_y, ldz + hw])   # front-left

    ax, ay = _proj_canvas(A, cx, cy)
    bx, by = _proj_canvas(B, cx, cy)
    dx, dy = _proj_canvas(D, cx, cy)

    # Affine: image (0,0)→A, (W,0)→B, (0,H)→D
    buf = io.BytesIO()
    icon.save(buf, format="PNG")
    uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    img_el = dwg.image(uri, insert=("0px", "0px"), size=(f"{W}px", f"{H}px"))
    img_el.attribs["preserveAspectRatio"] = "none"
    img_el.attribs["transform"] = (
        f"matrix({(bx-ax)/W:.6f},{(by-ay)/W:.6f},"
        f"{(dx-ax)/H:.6f},{(dy-ay)/H:.6f},{ax:.2f},{ay:.2f})"
    )
    return img_el


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
        icon = _load_icon(nd.get("icon_path"))
        if icon is not None:
            grp.add(_icon_element(dwg, icon, nd, cx, cy))
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
    for nd in (n for n in node_data if n.get("label", "")):
        pos = nd["pos"]
        hw  = nd["half_w"]
        ldx = float(pos[0])
        ldz = float(pos[2])

        # Anchor: front-floor corner of the brick (at floor Y=0)
        lx, ly = _proj_canvas(np.array([ldx, 0.0, ldz + hw]), cx, cy)

        text_el = dwg.text(
            nd["label"],
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
    piece_groups: list[tuple[str, list[Piece]]] | None = None,
    padding: int = 60,
) -> None:
    """Compose the final LEGO diagram SVG.

    Layer order:
    1. Background rect
    2. Isometric floor grid
    3. Brick ``<use>`` elements (globally sorted back-to-front across all groups)
    4. Icons on top faces
    5. Isometric labels
    6. Arrows (on top of everything so they are always visible)
    """
    pngs: list[tuple[Piece, int, int, float, float]] = [
        (piece, *img.size, ax, ay)
        for img, ax, ay, piece_list in renders.values()
        for piece in piece_list
    ]
    projected = _project_pieces(pngs)

    grid = _grid_params(renders)
    grid_corners = _grid_corner_sx_sy(*grid) if grid else []

    extra_pts = [
        (sx * PX_PER_MM, sy * PX_PER_MM)
        for from_pos, to_pos in arrows
        for p in (from_pos, to_pos)
        for sx, sy, _ in (project_ldraw(p),)
    ]

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

    # 3. Brick images in defs + placed instances (back-to-front, grouped by cluster)
    defs = _build_defs(dwg, renders)

    piece_proj = {
        id(piece): _project_piece(piece, iw, ih, ax, ay)
        for piece, iw, ih, ax, ay in pngs
    }

    def _use(sx_px: float, sy_px: float, label: str) -> None:
        def_id, dax, day = defs[label]
        dwg.add(dwg.use(
            f"#{def_id}",
            insert=(f"{cx(sx_px) - dax:.1f}px", f"{cy(sy_px) - day:.1f}px"),
        ))

    if piece_groups is not None:
        # Globally sort all pieces across all groups back-to-front so pieces
        # from different groups (e.g. lone nodes vs cluster bricks) never
        # occlude each other incorrectly.
        all_rows = sorted(
            [piece_proj[id(p)] for _, group_pieces in piece_groups
             for p in group_pieces if id(p) in piece_proj],
            key=lambda r: (-r[1], r[0]),
        )
        for _depth, _ldy, sx_px, sy_px, _ax, _ay, _iw, _ih, label in all_rows:
            _use(sx_px, sy_px, label)
    else:
        for sx_px, sy_px, _, _, _, _, label in projected:
            _use(sx_px, sy_px, label)

    # 4. Icons on top faces
    _draw_icons(dwg, node_data, cx, cy)

    # 5. Labels
    _draw_labels(dwg, node_data, cx, cy)

    # 6. Arrows
    _draw_floor_arrows(dwg, arrows, cx, cy)

    dwg.save(pretty=True)
    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
