"""compose.py — SVG composition: projection, layout, and piece placement."""

import io
import re
import base64
import hashlib
from pathlib import Path

import svgwrite
import svgwrite.image
from PIL import Image

from .parts import Piece
from .projection import project_ldraw, PX_PER_MM
from .grid import _grid_params, _grid_corner_sx_sy, _draw_isometric_grid


def _piece_label(piece: Piece) -> str:
    """Human-readable cache-key label for SVG comments."""
    def fmt_val(v: float) -> str:
        return f"{round(v)}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"
    rows = "[" + ",".join(
        "[" + ",".join(fmt_val(v) for v in row) + "]"
        for row in piece.rot
    ) + "]"
    return f"{piece.part} color={piece.color} rot={rows}"


def _project_piece(
    piece: Piece, iw: int, ih: int, anchor_x: float, anchor_y: float,
) -> tuple[float, float, float, float, float, float, int, int, str]:
    """Project one piece to a screen-space row: (depth, ldy, sx_px, sy_px, ax, ay, iw, ih, label)."""
    sx, sy, depth = project_ldraw(piece.pos)
    return (depth, float(piece.pos[1]), sx * PX_PER_MM, sy * PX_PER_MM,
            anchor_x, anchor_y, iw, ih, _piece_label(piece))


def _project_pieces(
    pngs: list[tuple[Piece, int, int, float, float]],
) -> list[tuple[float, float, float, float, int, int, str]]:
    """Project each piece to screen coords and sort back-to-front.

    Returns (sx_px, sy_px, ax, ay, iw, ih, label) per piece.
    """
    rows = sorted([_project_piece(*t) for t in pngs], key=lambda t: (-t[1], t[0]))
    return [(sx, sy, ax, ay, iw, ih, label)
            for _, _, sx, sy, ax, ay, iw, ih, label in rows]


def _canvas_bounds(
    projected: list[tuple],
    padding: int,
    extra_sx_sy: list[tuple[float, float]] = (),
) -> tuple[int, int, float, float]:
    """Return (W, H, min_x, min_y) for the SVG canvas."""
    xs = ([sx - ax       for sx, _,  ax, _,  iw, _,  _ in projected] +
          [sx - ax + iw  for sx, _,  ax, _,  iw, _,  _ in projected] +
          [sx for sx, _ in extra_sx_sy])
    ys = ([sy - ay       for _,  sy, _,  ay, _,  ih, _ in projected] +
          [sy - ay + ih  for _,  sy, _,  ay, _,  ih, _ in projected] +
          [sy for _, sy in extra_sx_sy])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = int(max_x - min_x + 2 * padding)
    H = int(max_y - min_y + 2 * padding)
    return W, H, min_x, min_y


def _make_dwg_image(
    label: str, img: Image.Image, anchor_x: float, anchor_y: float,
    dwg: svgwrite.Drawing,
) -> tuple[str, float, float, svgwrite.image.Image]:
    """Encode img as a base64 PNG and return (def_id, ax, ay, dwg_image)."""
    part_name = label.split()[0]
    def_id    = f"{part_name}-{hashlib.sha256(label.encode()).hexdigest()[:8]}"
    iw, ih    = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return def_id, anchor_x, anchor_y, dwg.image(uri, insert=("0px", "0px"),
                                                  size=(f"{iw}px", f"{ih}px"), id=def_id)


def _build_defs(
    dwg: svgwrite.Drawing,
    renders: dict[str, tuple[Image.Image, float, float, list]],
) -> dict[str, tuple[str, float, float]]:
    """Add each unique piece image to <defs> once; return label→(def_id, ax, ay)."""
    defs = {label: _make_dwg_image(label, img, anchor_x, anchor_y, dwg)
            for label, (img, anchor_x, anchor_y, _pieces) in renders.items()}
    for _, (_, _, _, dwg_image) in defs.items():
        dwg.defs.add(dwg_image)
    return {k: (def_id, ax, ay) for k, (def_id, ax, ay, _) in defs.items()}


def _inject_def_comments(output: str, defs: dict[str, tuple]) -> None:
    """Insert <!-- label --> before each <image element in <defs>."""
    it = iter(defs.keys())
    svg_text = Path(output).read_text()
    svg_text = re.sub(r"(<image\b)", lambda m: f"<!-- {next(it)} -->\n      {m.group(1)}", svg_text)
    Path(output).write_text(svg_text)


def compose_svg(
    renders: dict[str, tuple[Image.Image, float, float, list[Piece]]],
    output: str,
    padding: int = 60,
) -> None:
    pngs: list[tuple[Piece, int, int, float, float]] = [
        (piece, *img.size, ax, ay)
        for img, ax, ay, piece_list in renders.values()
        for piece in piece_list
    ]
    projected = _project_pieces(pngs)

    grid = _grid_params(renders)
    grid_corners = _grid_corner_sx_sy(*grid) if grid else []

    W, H, min_x, min_y = _canvas_bounds(projected, padding, grid_corners)

    def cx(val): return val - min_x + padding
    def cy(val): return val - min_y + padding

    dwg = svgwrite.Drawing(output, size=(f"{W}px", f"{H}px"))
    dwg.add(dwg.rect((0, 0), ("100%", "100%"), fill="#f8f8f0"))

    if grid:
        _draw_isometric_grid(dwg, *grid, cx, cy)

    defs = _build_defs(dwg, renders)

    for sx_px, sy_px, _, _, _, _, label in projected:
        def_id, ax, ay = defs[label]
        dwg.add(dwg.use(f"#{def_id}", insert=(f"{cx(sx_px) - ax:.1f}px", f"{cy(sy_px) - ay:.1f}px")))

    dwg.save(pretty=True)
    _inject_def_comments(output, defs)
    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
