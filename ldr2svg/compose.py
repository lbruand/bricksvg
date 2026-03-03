"""compose.py — SVG composition: projection, layout, and piece placement."""

import io
import re
import base64
import hashlib
from pathlib import Path
from typing import Callable

import svgwrite
import svgwrite.container
import svgwrite.image
from PIL import Image

from .parts import Piece, ldraw_rgb
from .projection import project_ldraw, PX_PER_MM
from .grid import _grid_params, _grid_corner_sx_sy, _draw_isometric_grid


class _SvgEl(svgwrite.container.Group):
    """Minimal wrapper to emit any SVG element tag, bypassing svgwrite's validator.

    svgwrite validates attribute names and values both at assignment time
    and at serialisation time.  For filter primitives (feColorMatrix etc.)
    and other elements not in its allowlist this raises ValueError/TypeError.
    Overriding ``get_xml`` lets us write the element directly to ElementTree
    without going through the validator.
    """
    def __init__(self, tag: str, **attribs):
        self.elementname = tag
        super().__init__(id=attribs.pop("id", None))
        self.attribs.update(attribs)

    def get_xml(self):
        import xml.etree.ElementTree as ET
        xml_el = ET.Element(self.elementname)
        for k, v in self.attribs.items():
            if v is not None:
                xml_el.set(k, str(v))
        for child in self.elements:
            xml_el.append(child.get_xml())
        return xml_el


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _fmt_rot_rows(piece: Piece) -> str:
    """Format the rotation matrix as a compact string for use in piece labels."""
    def fmt_val(v: float) -> str:
        return f"{round(v)}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"
    return "[" + ",".join(
        "[" + ",".join(fmt_val(v) for v in row) + "]"
        for row in piece.rot
    ) + "]"


def _hash_label(label: str) -> str:
    """Return an 8-character SHA-256 hex digest of *label*."""
    return hashlib.sha256(label.encode()).hexdigest()[:8]


def _img_to_data_uri(img: Image.Image) -> str:
    """Encode *img* as a base64 PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"



def _draw_grid(
    dwg: svgwrite.Drawing,
    grid: tuple | None,
    cx: Callable[[float], float],
    cy: Callable[[float], float],
) -> None:
    """Draw the isometric floor grid if grid params are present."""
    if grid:
        fy, gx0, gx1, gz0, gz1 = grid
        _draw_isometric_grid(dwg, fy, gx0, gx1, gz0, gz1, cx, cy)


# ---------------------------------------------------------------------------
# Piece label / projection
# ---------------------------------------------------------------------------

def _piece_label(piece: Piece) -> str:
    """Human-readable cache-key label (includes colour)."""
    return f"{piece.part} color={piece.color} rot={_fmt_rot_rows(piece)}"


def _piece_label_no_color(piece: Piece) -> str:
    """Cache-key excluding colour — one render per part+rotation."""
    return f"{piece.part} rot={_fmt_rot_rows(piece)}"


def _project_piece(
    piece: Piece,
    iw: int,
    ih: int,
    anchor_x: float,
    anchor_y: float,
    label_fn: Callable[[Piece], str] = _piece_label,
) -> tuple[float, float, float, float, float, float, int, int, str]:
    """Project one piece to a screen-space row: (depth, ldy, sx_px, sy_px, ax, ay, iw, ih, label)."""
    sx, sy, depth = project_ldraw(piece.pos)
    return (depth, float(piece.pos[1]), sx * PX_PER_MM, sy * PX_PER_MM,
            anchor_x, anchor_y, iw, ih, label_fn(piece))


def _project_pieces(
    pngs: list[tuple[Piece, int, int, float, float]],
    label_fn: Callable[[Piece], str] = _piece_label,
) -> list[tuple[float, float, float, float, int, int, str]]:
    """Project each piece to screen coords and sort back-to-front.

    Returns (sx_px, sy_px, ax, ay, iw, ih, label) per piece.
    """
    rows = sorted([_project_piece(*t, label_fn=label_fn) for t in pngs],
                  key=lambda t: (-t[1], t[0]))
    return [(sx, sy, ax, ay, iw, ih, label)
            for _, _, sx, sy, ax, ay, iw, ih, label in rows]


def _canvas_bounds(
    projected: list[tuple],
    padding: int,
    extra_sx_sy: list[tuple[float, float]] | None = None,
) -> tuple[int, int, float, float]:
    """Return (W, H, min_x, min_y) for the SVG canvas."""
    extra = extra_sx_sy or []
    xs = ([sx - ax       for sx, _,  ax, _,  iw, _,  _ in projected] +
          [sx - ax + iw  for sx, _,  ax, _,  iw, _,  _ in projected] +
          [sx for sx, _ in extra])
    ys = ([sy - ay       for _,  sy, _,  ay, _,  ih, _ in projected] +
          [sy - ay + ih  for _,  sy, _,  ay, _,  ih, _ in projected] +
          [sy for _, sy in extra])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = int(max_x - min_x + 2 * padding)
    H = int(max_y - min_y + 2 * padding)
    return W, H, min_x, min_y


# ---------------------------------------------------------------------------
# SVG <defs> builders
# ---------------------------------------------------------------------------

def _make_dwg_image(
    label: str, img: Image.Image, anchor_x: float, anchor_y: float,
    dwg: svgwrite.Drawing,
) -> tuple[str, float, float, svgwrite.image.Image]:
    """Encode img as a base64 PNG and return (def_id, ax, ay, dwg_image)."""
    part_name = label.split()[0]
    def_id    = f"{part_name}-{_hash_label(label)}"
    iw, ih    = img.size
    return def_id, anchor_x, anchor_y, dwg.image(
        _img_to_data_uri(img), insert=("0px", "0px"),
        size=(f"{iw}px", f"{ih}px"), id=def_id,
    )


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


def _build_defs_masked(
    dwg: svgwrite.Drawing,
    renders: dict[str, tuple[Image.Image, float, float, list]],
) -> dict[str, tuple[str, float, float, int, int]]:
    """Store each white render once in <defs>; return label → (img_id, ax, ay, iw, ih)."""
    result = {}
    for label, (img, ax, ay, _pieces) in renders.items():
        iw, ih = img.size
        part_name = label.split()[0]
        img_id = f"grayscale-{part_name}-{_hash_label(label)}"
        dwg.defs.add(dwg.image(
            _img_to_data_uri(img), insert=("0", "0"),
            size=(str(iw), str(ih)), id=img_id,
        ))
        result[label] = (img_id, ax, ay, iw, ih)
    return result


def _build_duotone_filters(
    dwg: svgwrite.Drawing,
    hex_colors: set[str],
) -> dict[str, str]:
    """Create one feColorMatrix filter per unique brick color; return hex → filter_id.

    Replicates PIL ``ImageChops.multiply``: white pixels become the brick colour;
    shadow pixels darken proportionally.  ``feColorMatrix`` row ``r 0 0 0 0``
    maps input-R (= grey value of the white render) to output-R × r.
    """
    result = {}
    for hex_color in hex_colors:
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        fid = f"duotone-{hex_color[1:]}"
        filt = _SvgEl("filter", id=fid)
        filt["color-interpolation-filters"] = "sRGB"
        filt.add(_SvgEl(
            "feColorMatrix",
            type="matrix",
            values=f"{r:.4f} 0 0 0 0  {g:.4f} 0 0 0 0  {b:.4f} 0 0 0 0  0 0 0 1 0",
        ))
        dwg.defs.add(filt)
        result[hex_color] = fid
    return result


def _inject_def_comments(output: str, defs: dict[str, tuple]) -> None:
    """Insert <!-- label --> before each <image element in <defs>."""
    it = iter(defs.keys())
    svg_text = Path(output).read_text()
    svg_text = re.sub(r"(<image\b)", lambda m: f"<!-- {next(it)} -->\n      {m.group(1)}", svg_text)
    Path(output).write_text(svg_text)


# ---------------------------------------------------------------------------
# Top-level compose
# ---------------------------------------------------------------------------

def compose_svg(
    renders: dict[str, tuple[Image.Image, float, float, list[Piece]]],
    output: str,
    padding: int = 60,
    masked: bool = False,
) -> None:
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

    W, H, min_x, min_y = _canvas_bounds(projected, padding, grid_corners)

    def cx(val): return val - min_x + padding
    def cy(val): return val - min_y + padding

    dwg = svgwrite.Drawing(output, size=(f"{W}px", f"{H}px"))
    dwg.add(dwg.rect((0, 0), ("100%", "100%"), fill="#f8f8f0"))
    _draw_grid(dwg, grid, cx, cy)

    if masked:
        defs_m = _build_defs_masked(dwg, renders)
        hex_colors = {
            "#{:02x}{:02x}{:02x}".format(*ldraw_rgb(p.color))
            for p, *_ in pngs
        }
        filters = _build_duotone_filters(dwg, hex_colors)
        piece_proj = {
            id(p): _project_piece(p, iw, ih, ax, ay, label_fn=_piece_label_no_color)
            for p, iw, ih, ax, ay in pngs
        }
        lsx_to_piece: dict[tuple[str, float], Piece] = {
            (row[8], row[2]): p
            for p, *_ in pngs
            for row in (piece_proj[id(p)],)
        }
        for sx_px, sy_px, _, _, _, _, label in projected:
            piece = lsx_to_piece[(label, sx_px)]
            img_id, dax, day, _, _ = defs_m[label]
            hex_color = "#{:02x}{:02x}{:02x}".format(*ldraw_rgb(piece.color))
            use_el = dwg.use(
                f"#{img_id}",
                insert=(f"{cx(sx_px) - dax:.1f}px", f"{cy(sy_px) - day:.1f}px"),
            )
            use_el.attribs["filter"] = f"url(#{filters[hex_color]})"
            dwg.add(use_el)
        dwg.save(pretty=True)
    else:
        defs = _build_defs(dwg, renders)
        for sx_px, sy_px, _, _, _, _, label in projected:
            def_id, ax, ay = defs[label]
            dwg.add(dwg.use(f"#{def_id}", insert=(f"{cx(sx_px) - ax:.1f}px", f"{cy(sy_px) - ay:.1f}px")))
        dwg.save(pretty=True)
        _inject_def_comments(output, defs)

    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
