"""diagram_compose.py — SVG composition for diagram2svg output."""

import math

import numpy as np
import svgwrite
import svgwrite.container
from PIL import Image

from .compose import (
    _project_pieces, _project_piece, _canvas_bounds,
    _build_defs, _build_defs_masked,
    _piece_label_no_color,
    _img_to_data_uri, _place_masked_piece, _draw_grid,
)
from .parts import Piece, ldraw_rgb
from .projection import project_ldraw, PX_PER_MM
from .grid import _grid_params, _grid_corner_sx_sy


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
    img_el = dwg.image(_img_to_data_uri(icon), insert=("0px", "0px"), size=(f"{W}px", f"{H}px"))
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
    """Draw labels on the left-hand (front/+Z) face of each node brick.

    The face transform is: scaleY(cos30°) → skewX(30°) → rotate(30° CW),
    which maps the text baseline along the isometric +X axis and character
    height straight down in screen space.
    Combined matrix: matrix(0.866025, 0.5, 0, 1, tx, ty).
    Anchor: top-centre of the front (+Z) face of the tile.
    """
    grp = dwg.g(id="labels")
    for nd in (n for n in node_data if n.get("label", "")):
        pos = nd["pos"]
        hw  = nd["half_w"]
        ldx = float(pos[0])
        ldy = float(pos[1])
        ldz = float(pos[2])

        # Anchor at the top-centre of the front face of the tile/brick
        lx, ly = _proj_canvas(np.array([ldx, ldy, ldz + hw]), cx, cy)

        text_el = dwg.text(
            nd["label"],
            insert=(0, 0),
            font_size="9",
            fill="#333",
            text_anchor="middle",
        )
        # scaleY(cos30°) → skewX(30°) → rotate(30° CW)
        text_el.attribs["transform"] = (
            f"matrix(0.866025,0.5,0,1,{lx:.1f},{ly:.1f})"
        )
        grp.add(text_el)
    dwg.add(grp)


# ---------------------------------------------------------------------------
# Cluster labels
# ---------------------------------------------------------------------------

def _draw_cluster_labels(
    dwg: svgwrite.Drawing,
    cluster_data: list[dict],
    cx,
    cy,
) -> None:
    """Draw isometric-skewed text labels at the front edge of each cluster platform."""
    grp = dwg.g(id="cluster_labels")
    for nd in (n for n in cluster_data if n.get("label", "")):
        pos = nd["pos"]
        lx, ly = _proj_canvas(pos, cx, cy)
        # Shift f (screen-Y translation) to vertically centre the glyphs on the
        # tile. b+d=1 so adding δ to f produces a pure +δ px screen-Y shift.
        # Two contributions to the needed offset:
        #   glyph em-centre above baseline: d×0.4×fs = 0.5×0.4×fs ≈ 0.08×fs
        #   tile front face (26 px high) pushes the piece visual centre 13 px
        #   below the projected top-face anchor → ~0.42×fs at fs=40
        # Total ≈ 0.5×fs centres the text on the visible tile piece.
        _font_size = 40
        ly += _font_size * 0.5
        text_el = dwg.text(
            nd["label"],
            insert=(0, 0),
            font_size=str(_font_size),
            font_weight="bold",
            fill="#444",
            text_anchor="middle",
        )
        # Top-face isometric matrix: col1=(a,b)=+X dir, col2=(c,d)=+Z dir.
        # Glyphs lie flat on the horizontal tile surface.
        text_el.attribs["transform"] = (
            f"matrix(0.866025,0.5,-0.866025,0.5,{lx:.1f},{ly:.1f})"
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
    cluster_data: list[dict] | None = None,
    padding: int = 60,
    masked: bool = False,
) -> None:
    """Compose the final isometric brick SVG.

    Layer order:
    1. Background rect
    2. Isometric floor grid
    3. Brick elements (globally sorted back-to-front across all groups)
    4. Icons on top faces
    5. Isometric labels
    6. Arrows (on top of everything so they are always visible)

    When *masked* is ``True``, pieces are rendered using a white base image
    stored once per unique (part, rotation) in ``<defs>``.  Each instance is
    composed as::

        <g style="isolation: isolate" transform="translate(x,y)">
          <rect fill="#color" mask="url(#alpha-…)"/>   <!-- coloured base -->
          <use href="#shadow-…" style="mix-blend-mode: multiply"/>  <!-- shadows -->
        </g>

    This avoids re-rendering the same geometry in multiple colours.
    """
    pngs: list[tuple[Piece, int, int, float, float]] = [
        (piece, *img.size, ax, ay)
        for img, ax, ay, piece_list in renders.values()
        for piece in piece_list
    ]
    if masked:
        projected = _project_pieces(pngs, label_fn=_piece_label_no_color)
    else:
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
    _draw_grid(dwg, grid, cx, cy)

    # Arrow marker in <defs>
    _add_arrow_defs(dwg)

    # 3. Brick images in defs + placed instances (back-to-front, grouped by cluster)
    if masked:
        defs_m = _build_defs_masked(dwg, renders)
        piece_proj = {
            id(piece): _project_piece(piece, iw, ih, ax, ay,
                                      label_fn=_piece_label_no_color)
            for piece, iw, ih, ax, ay in pngs
        }
        piece_hex: dict[int, str] = {
            id(piece): "#{:02x}{:02x}{:02x}".format(*ldraw_rgb(piece.color))
            for piece, *_ in pngs
        }

        def _place_masked(grp, piece: Piece, sx_px: float, sy_px: float, label: str) -> None:
            shadow_id, mask_id, dax, day, iw, ih = defs_m[label]
            _place_masked_piece(dwg, grp, shadow_id, mask_id, dax, day, iw, ih,
                                sx_px, sy_px, piece_hex[id(piece)], cx, cy)

        defs: dict[str, tuple[str, float, float]] = {}  # not used in masked path
    else:
        defs = _build_defs(dwg, renders)
        piece_proj = {
            id(piece): _project_piece(piece, iw, ih, ax, ay)
            for piece, iw, ih, ax, ay in pngs
        }

    def _row_key(p: Piece) -> tuple:
        r = piece_proj[id(p)]
        return (-r[1], r[0])   # sort back-to-front: by -ldy then depth

    def _place_piece(grp, piece: Piece) -> None:
        """Place one piece into *grp*, using masked or normal technique."""
        row = piece_proj[id(piece)]
        sx_px, sy_px, label = row[2], row[3], row[8]
        if masked:
            _place_masked(grp, piece, sx_px, sy_px, label)
        else:
            def_id, dax, day = defs[label]
            grp.add(dwg.use(
                f"#{def_id}",
                insert=(f"{cx(sx_px) - dax:.1f}px", f"{cy(sy_px) - day:.1f}px"),
            ))

    def _use(sx_px: float, sy_px: float, label: str) -> None:
        def_id, dax, day = defs[label]
        dwg.add(dwg.use(
            f"#{def_id}",
            insert=(f"{cx(sx_px) - dax:.1f}px", f"{cy(sy_px) - day:.1f}px"),
        ))

    if piece_groups is not None:
        # Pass 1: platform tiles (3024 / 3070b) per cluster — each cluster in its
        # own named <g> so they can be targeted by CSS / inspected in an SVG editor.
        # Platforms are flat on the floor so no cross-cluster depth conflict.
        for group_name, group_pieces in piece_groups:
            platform_pieces = sorted(
                [p for p in group_pieces
                 if p.part in ("3024", "3070b") and id(p) in piece_proj],
                key=_row_key,
            )
            if not platform_pieces:
                continue
            grp = dwg.g(id=f"platform_{group_name}")
            for p in platform_pieces:
                _place_piece(grp, p)
            dwg.add(grp)

        # Pass 2: node bricks + tiles (non-3024/3070b) per cluster in named
        # <g> groups, sorted back-to-front within each group.
        for group_name, group_pieces in piece_groups:
            node_pieces = sorted(
                [p for p in group_pieces
                 if p.part not in ("3024", "3070b") and id(p) in piece_proj],
                key=_row_key,
            )
            if not node_pieces:
                continue
            grp = dwg.g(id=f"nodes_{group_name}")
            for p in node_pieces:
                _place_piece(grp, p)
            dwg.add(grp)
    else:
        if masked:
            # projected is sorted; map back to piece objects via piece_proj
            pid_to_piece = {id(p): p for p, *_ in pngs}
            for sx_px, sy_px, _, _, _, _, label in projected:
                piece_obj = next(
                    pid_to_piece[pid] for pid, row in piece_proj.items()
                    if row[2] == sx_px and row[8] == label
                )
                grp = dwg.g()
                _place_masked(grp, piece_obj, sx_px, sy_px, label)
                dwg.add(grp)
        else:
            for sx_px, sy_px, _, _, _, _, label in projected:
                _use(sx_px, sy_px, label)

    # 4. Icons on top faces
    _draw_icons(dwg, node_data, cx, cy)

    # 5. Node labels
    _draw_labels(dwg, node_data, cx, cy)

    # 6. Cluster labels
    if cluster_data:
        _draw_cluster_labels(dwg, cluster_data, cx, cy)

    # 7. Arrows
    _draw_floor_arrows(dwg, arrows, cx, cy)

    dwg.save(pretty=True)
    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
