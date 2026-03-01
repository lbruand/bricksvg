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


# svgwrite has no built-in Mask element; reuse Group with the right tag name.
class _SvgMask(svgwrite.container.Group):
    elementname = "mask"


def _piece_label(piece: Piece) -> str:
    """Human-readable cache-key label for SVG comments."""
    def fmt_val(v: float) -> str:
        return f"{round(v)}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"
    rows = "[" + ",".join(
        "[" + ",".join(fmt_val(v) for v in row) + "]"
        for row in piece.rot
    ) + "]"
    return f"{piece.part} color={piece.color} rot={rows}"


def _piece_label_no_color(piece: Piece) -> str:
    """Cache-key excluding color — for masked/white rendering (one render per part+rotation)."""
    def fmt_val(v: float) -> str:
        return f"{round(v)}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"
    rows = "[" + ",".join(
        "[" + ",".join(fmt_val(v) for v in row) + "]"
        for row in piece.rot
    ) + "]"
    return f"{piece.part} rot={rows}"


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


def _build_defs_masked(
    dwg: svgwrite.Drawing,
    renders: dict[str, tuple[Image.Image, float, float, list]],
) -> dict[str, tuple[str, str, float, float, int, int]]:
    """For masked rendering: add shadow images + alpha masks to <defs>.

    Each unique white render gets:
    - An ``<image id="shadow-{sha}">`` element (the white render itself)
    - A ``<mask id="alpha-{sha}">`` element whose content is the alpha channel
      of the white render encoded as a greyscale PNG.

    In the SVG, each piece instance is composed as::

        <g style="isolation: isolate" transform="translate(x, y)">
          <rect … fill="#color" mask="url(#alpha-{sha})"/>
          <use href="#shadow-{sha}" style="mix-blend-mode: multiply"/>
        </g>

    The rect supplies the target colour; the white render (with multiply
    blend) darkens the colour where shadow pixels appear, leaving highlight
    pixels unchanged.

    Returns label → (shadow_id, mask_id, ax, ay, iw, ih).
    """
    result = {}
    for label, (img, ax, ay, _pieces) in renders.items():
        iw, ih = img.size
        sha = hashlib.sha256(label.encode()).hexdigest()[:8]

        # Shadow image (white render) — referenced by <use> with multiply blend
        shadow_id = f"shadow-{sha}"
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        shadow_img = dwg.image(uri, insert=("0", "0"),
                               size=(str(iw), str(ih)), id=shadow_id)
        dwg.defs.add(shadow_img)

        # Alpha mask — greyscale PNG of the alpha channel; white = opaque
        mask_id = f"alpha-{sha}"
        alpha_ch = img.split()[3]          # PIL Image, mode='L'
        alpha_buf = io.BytesIO()
        alpha_ch.save(alpha_buf, format="PNG")
        alpha_uri = f"data:image/png;base64,{base64.b64encode(alpha_buf.getvalue()).decode()}"
        mask_el = _SvgMask(id=mask_id)
        mask_el["maskContentUnits"] = "userSpaceOnUse"
        mask_el["maskUnits"] = "userSpaceOnUse"
        mask_el["x"] = "0"
        mask_el["y"] = "0"
        mask_el["width"] = str(iw)
        mask_el["height"] = str(ih)
        mask_el.add(dwg.image(alpha_uri, insert=("0", "0"), size=(str(iw), str(ih))))
        dwg.defs.add(mask_el)

        result[label] = (shadow_id, mask_id, ax, ay, iw, ih)
    return result


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

    if grid:
        fy, gx0, gx1, gz0, gz1 = grid
        _draw_isometric_grid(dwg, fy, gx0, gx1, gz0, gz1, cx, cy)

    if masked:
        defs_m = _build_defs_masked(dwg, renders)
        piece_proj = {
            id(p): _project_piece(p, iw, ih, ax, ay, label_fn=_piece_label_no_color)
            for p, iw, ih, ax, ay in pngs
        }
        # (label, sx_px) → piece — used to recover colour when placing
        lsx_to_piece: dict[tuple[str, float], Piece] = {
            (row[8], row[2]): p
            for p, *_ in pngs
            for row in (piece_proj[id(p)],)
        }
        for sx_px, sy_px, _, _, _, _, label in projected:
            piece = lsx_to_piece[(label, sx_px)]
            shadow_id, mask_id, dax, day, iw, ih = defs_m[label]
            r, g, b = ldraw_rgb(piece.color)
            x = f"{cx(sx_px) - dax:.1f}"
            y = f"{cy(sy_px) - day:.1f}"
            grp = dwg.g(style="isolation: isolate", transform=f"translate({x},{y})")
            rect = dwg.rect(insert=("0", "0"), size=(str(iw), str(ih)),
                            fill=f"#{r:02x}{g:02x}{b:02x}")
            rect.attribs["mask"] = f"url(#{mask_id})"
            grp.add(rect)
            use_el = dwg.use(f"#{shadow_id}")
            use_el.attribs["style"] = "mix-blend-mode: multiply"
            grp.add(use_el)
            dwg.add(grp)
        dwg.save(pretty=True)
    else:
        defs = _build_defs(dwg, renders)
        for sx_px, sy_px, _, _, _, _, label in projected:
            def_id, ax, ay = defs[label]
            dwg.add(dwg.use(f"#{def_id}", insert=(f"{cx(sx_px) - ax:.1f}px", f"{cy(sy_px) - ay:.1f}px")))
        dwg.save(pretty=True)
        _inject_def_comments(output, defs)

    print(f"Saved: {output}  ({W}×{H} px, {len(projected)} pieces)")
