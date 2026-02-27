#!/usr/bin/env python3
"""ldr2png_svg.py - Render each LDraw piece with LEGO.scad/OpenSCAD, then compose into SVG."""

import sys
import math
import argparse
import subprocess
import tempfile
import io
import re
import base64
import hashlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PIL import Image
import svgwrite
import svgwrite.image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEGOLIB = Path(__file__).parent / "LEGO.scad"

# ---------------------------------------------------------------------------
# Camera parameters (OpenSCAD)
# ---------------------------------------------------------------------------
# True isometric: camera along (1,1,1)/√3 ↔ rx = arccos(1/√3) ≈ 54.74°, rz = 45°.
# All three world axes project with equal foreshortening; horizontal axes
# appear at exactly ±30° on screen.
CAMERA_RX = math.degrees(math.acos(1 / math.sqrt(3)))  # ≈ 54.74°
CAMERA_RZ = 45.0   # degrees spin (around Z)
CAMERA_D  = 300.0  # camera distance (mm) — controls scale
IMG_PX    = 800    # render each piece into a square IMG_PX × IMG_PX PNG

# OpenSCAD ortho scale: viewport covers 2*D*tan(fov/2) mm → IMG_PX pixels.
# OpenSCAD default full perspective FOV = 22.5°.
_OPENSCAD_FOV_DEG = 22.5
PX_PER_MM = IMG_PX / (2 * CAMERA_D * math.tan(math.radians(_OPENSCAD_FOV_DEG / 2)))

# ---------------------------------------------------------------------------
# Isometric grid
# ---------------------------------------------------------------------------
GRID_STEP   = 20      # LDU = 1 LEGO stud = 8 mm
GRID_MARGIN = 2       # extra grid cells around scene bounding box
GRID_COLOR  = "#c0b8b0"
GRID_WIDTH  = "0.5"

# ---------------------------------------------------------------------------
# LDraw colour table
# ---------------------------------------------------------------------------
LDRAW_COLORS: dict[int, tuple[int, int, int]] = {
    0: (5, 19, 29),      1: (0, 85, 191),    2: (37, 122, 62),   3: (0, 133, 85),
    4: (201, 26, 9),     5: (200, 112, 160),  6: (88, 58, 32),    7: (155, 155, 155),
    8: (100, 100, 100),  9: (180, 210, 227),  10: (75, 151, 75),  11: (85, 165, 175),
    12: (255, 119, 0),   13: (247, 188, 213), 14: (242, 205, 55), 15: (255, 255, 255),
    17: (190, 233, 15),  19: (242, 205, 161), 22: (129, 9, 110),  25: (254, 138, 24),
    28: (173, 114, 53),  70: (92, 29, 13),    71: (150, 150, 150),72: (108, 110, 104),
    85: (84, 32, 113),   86: (100, 51, 0),    92: (181, 106, 55), 288: (0, 71, 33),
    320: (114, 14, 14),  321: (54, 172, 200), 484: (160, 82, 48),
}

def ldraw_rgb(color_id: int) -> tuple[int, int, int]:
    return LDRAW_COLORS.get(color_id, (136, 136, 136))

# ---------------------------------------------------------------------------
# LDraw part → LEGO.scad block() mapping
# ---------------------------------------------------------------------------
# Sizes in studs; height is a multiplier of 9.6 mm (brick height)
# LEGO.scad block origin is bottom-left corner at (0,0,0).
# In LDraw, the local x=long axis, z=short axis, y=0 at top.

@dataclass
class PartDef:
    width: int        # studs in local X
    length: int       # studs in local Y (= -LDraw Z after axis swap)
    height: float     # 1=brick, 1/3=plate
    block_type: str   # "brick", "round", etc.
    # pre-computed dimensions (mm in OpenSCAD coords)
    @property
    def w_mm(self): return self.width  * 8.0
    @property
    def l_mm(self): return self.length * 8.0
    @property
    def h_mm(self): return self.height * 9.6

PART_MAP: dict[str, PartDef] = {
    "3666":  PartDef(6, 1, 1/3, "brick"),   # Plate  1×6 — long axis in LDraw X
    "60474": PartDef(4, 4, 1/3, "brick"),   # Plate 4×4 Round (approximated)
    "3062a": PartDef(1, 1, 1.0, "round"),   # Brick 1×1 Round
}

# ---------------------------------------------------------------------------
# Coordinate transform: LDraw ↔ OpenSCAD
# ---------------------------------------------------------------------------
# LDraw:  X right, Y down, Z toward viewer  (units: LDU)
# LEGO.scad/OpenSCAD: X right, Y forward, Z up  (units: mm)
# Mapping: os_X = ld_X * 0.4,  os_Y = -ld_Z * 0.4,  os_Z = -ld_Y * 0.4
_T = np.array([[1, 0, 0],   # os_X  = ld_X
               [0, 0,-1],   # os_Y  = -ld_Z
               [0,-1, 0]],  # os_Z  = -ld_Y
              dtype=float)

LDU_TO_MM = 0.4

def ldraw_to_os(pos_ld: np.ndarray, rot_ld: np.ndarray) -> np.ndarray:
    """Build a 4×4 OpenSCAD transform from a LDraw position+rotation."""
    R_os  = _T @ rot_ld @ _T      # rotate 3×3
    t_os  = _T @ pos_ld * LDU_TO_MM
    m = np.eye(4)
    m[:3, :3] = R_os
    m[:3,  3] = t_os
    return m

# ---------------------------------------------------------------------------
# OpenSCAD camera → 2-D projection
# ---------------------------------------------------------------------------
def _cam_matrix(rx_deg: float, rz_deg: float) -> np.ndarray:
    """R_cam = Rx(rx) @ Rz(rz) — maps OpenSCAD world → camera space."""
    rx = np.radians(rx_deg)
    rz = np.radians(rz_deg)
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(rx), -np.sin(rx) ],
                   [0, np.sin(rx),  np.cos(rx) ]])
    Rz = np.array([[ np.cos(rz), -np.sin(rz), 0],
                   [ np.sin(rz),  np.cos(rz), 0],
                   [ 0,           0,          1]])
    return Rx @ Rz

_R_CAM = _cam_matrix(-CAMERA_RX, -CAMERA_RZ)

def project_ldraw(pos_ld: np.ndarray) -> tuple[float, float, float]:
    """Project a LDraw world position to (screen_x, screen_y, depth).

    screen_y increases downward (SVG convention).
    depth increases toward the camera (larger = in front).
    """
    p_os  = _T @ pos_ld * LDU_TO_MM
    cam   = _R_CAM @ p_os
    return float(cam[0]), float(-cam[1]), float(cam[2])

# ---------------------------------------------------------------------------
# SCAD file generation
# ---------------------------------------------------------------------------
def make_scad(part: PartDef, r: tuple[int, int, int], rot_ld: np.ndarray) -> str:
    """Return a .scad file that renders *part* centered on the origin.

    The LDraw piece origin (top-face centre) is placed at (0, 0, 0) in
    OpenSCAD space.  The block therefore extends downward (negative Z).
    The LDraw rotation matrix is converted to OpenSCAD axes and applied
    via multmatrix so each piece renders in its scene orientation.
    """
    r_str = f"[{r[0]/255:.3f}, {r[1]/255:.3f}, {r[2]/255:.3f}]"
    # LEGO.scad block() already centers in X and Y; only shift Z so top is at z=0
    tz = -part.h_mm
    h_param = f"{part.height:.6f}"
    # Convert LDraw rotation → OpenSCAD rotation: R_os = T @ rot_ld @ T
    R_os = _T @ rot_ld @ _T
    def fmt_row(row):
        return f"[{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},0]"
    mat_str = (f"[{fmt_row(R_os[0])},{fmt_row(R_os[1])},"
               f"{fmt_row(R_os[2])},[0,0,0,1]]")
    return (
        f'use <{LEGOLIB}>\n'
        f'$fs = 1.0; $fa = 8;\n'
        f'color({r_str})\n'
        f'multmatrix({mat_str})\n'
        f'translate([0, 0, {tz}])\n'
        f'  block(width={part.width}, length={part.length},\n'
        f'        height={h_param}, type="{part.block_type}");\n'
    )

# ---------------------------------------------------------------------------
# OpenSCAD rendering
# ---------------------------------------------------------------------------
def render_piece(scad_src: str, png_out: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".scad", mode="w", delete=False) as f:
        f.write(scad_src)
        scad_path = Path(f.name)
    try:
        camera = f"0,0,0,{CAMERA_RX},0,{CAMERA_RZ},{CAMERA_D}"
        result = subprocess.run(
            ["openscad",
             "--camera", camera,
             "--projection", "ortho",
             "--imgsize", f"{IMG_PX},{IMG_PX}",
             "-o", str(png_out),
             str(scad_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  openscad error: {result.stderr[-400:]}", file=sys.stderr)
            return False
        return png_out.exists()
    finally:
        scad_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Background removal + tight crop
# ---------------------------------------------------------------------------
CROP_PAD = 6   # extra transparent pixels around each cropped piece

def remove_and_crop(png_path: Path) -> tuple[Image.Image, float, float]:
    """Remove background, crop tightly, return (img, anchor_x, anchor_y).

    anchor_x/y is the position of the piece origin (= image centre before
    cropping) inside the returned cropped image.
    """
    img  = Image.open(png_path).convert("RGBA")
    arr  = np.array(img)
    bg   = arr[0, 0, :3]

    # Knock out background pixels
    tol  = 12
    mask = np.all(np.abs(arr[:, :, :3].astype(int) - bg.astype(int)) <= tol, axis=2)
    arr[mask, 3] = 0

    # Find bounding box of opaque pixels
    alpha = arr[:, :, 3]
    rows  = np.any(alpha > 0, axis=1)
    cols  = np.any(alpha > 0, axis=0)
    if not rows.any():
        # Blank image — return as-is, anchor at centre
        h, w = arr.shape[:2]
        return Image.fromarray(arr, "RGBA"), w / 2, h / 2

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    # Add padding, clamp to image bounds
    H, W   = arr.shape[:2]
    r0c    = max(0, r0 - CROP_PAD)
    r1c    = min(H, r1 + CROP_PAD + 1)
    c0c    = max(0, c0 - CROP_PAD)
    c1c    = min(W, c1 + CROP_PAD + 1)

    cropped = Image.fromarray(arr[r0c:r1c, c0c:c1c], "RGBA")

    # The piece origin was at the render canvas centre (IMG_PX/2, IMG_PX/2)
    anchor_x = IMG_PX / 2 - c0c
    anchor_y = IMG_PX / 2 - r0c

    return cropped, anchor_x, anchor_y

# ---------------------------------------------------------------------------
# LDraw scene parser
# ---------------------------------------------------------------------------
@dataclass
class Piece:
    part: str          # e.g. "3666"
    color: int
    pos: np.ndarray    # (3,) LDraw world position (LDU)
    rot: np.ndarray    # (3, 3) LDraw rotation matrix

def _parse_ldr_line(parts: list[str]) -> Piece:
    """Parse a tokenised LDraw type-1 line into a Piece. Caller must verify len(parts) >= 15 and parts[0] == '1'."""
    return Piece(
        part  = Path(" ".join(parts[14:])).stem.lower(),
        color = int(parts[1]),
        pos   = np.array([float(p) for p in parts[2:5]]),
        rot   = np.array([float(p) for p in parts[5:14]], dtype=float).reshape(3, 3),
    )

def parse_ldr(path: Path) -> list[Piece]:
    lines = path.read_text(errors="replace").splitlines()
    return [_parse_ldr_line(p) for p in (line.split() for line in lines)
            if p and p[0] == "1" and len(p) >= 15]

# ---------------------------------------------------------------------------
# SVG composition
# ---------------------------------------------------------------------------
def _piece_label(piece: "Piece") -> str:
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
    # Sort back-to-front: primary = LDraw Y descending (lower pieces first, elevated last),
    # secondary = cam depth ascending (farther first) for same-height pieces.
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
    label: str, img: Image.Image, anchor_x: float, anchor_y: float, dwg: svgwrite.Drawing,
) -> tuple[str, float, float, svgwrite.image.Image]:
    """Encode img as a base64 PNG and return (def_id, ax, ay, dwg_image)."""
    part_name  = label.split()[0]
    def_id     = f"{part_name}-{hashlib.sha256(label.encode()).hexdigest()[:8]}"
    iw, ih     = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    return def_id, anchor_x, anchor_y, dwg.image(uri, insert=("0px", "0px"),
                                                  size=(f"{iw}px", f"{ih}px"), id=def_id)


def _build_defs(
    dwg: svgwrite.Drawing,
    renders: dict[str, tuple["Image.Image", float, float, list]],
) -> dict[str, tuple[str, float, float]]:
    """Add each unique piece image to <defs> once; return label→(def_id, ax, ay)."""
    defs: dict[str, tuple[str, float, float, svgwrite.image.Image]] = {
        label: _make_dwg_image(label, img, anchor_x, anchor_y, dwg)
        for label, (img, anchor_x, anchor_y, _pieces) in renders.items()
    }
    for k, (_, _, _, dwg_image) in defs.items():
        dwg.defs.add(dwg_image)
    return {k: (def_id, ax, ay) for k, (def_id, ax, ay, _) in defs.items()}


def _inject_def_comments(output: str, defs: dict[str, tuple]) -> None:
    """Insert <!-- label --> before each <image element in <defs>."""
    it = iter(defs.keys())
    svg_text = Path(output).read_text()
    svg_text = re.sub(r"(<image\b)", lambda m: f"<!-- {next(it)} -->\n      {m.group(1)}", svg_text)
    Path(output).write_text(svg_text)


def _floor_y_ldu(renders: dict) -> float:
    """LDraw Y of the floor = max bottom-face Y across all known pieces."""
    ys = [float(p.pos[1]) + PART_MAP[p.part].h_mm / LDU_TO_MM
          for _img, _ax, _ay, plist in renders.values()
          for p in plist if p.part in PART_MAP]
    return max(ys, default=0.0)


def _grid_params(renders: dict) -> tuple | None:
    """Return (floor_y, gx0, gx1, gz0, gz1) in LDU, or None if no pieces."""
    all_pieces = [p for _img, _ax, _ay, plist in renders.values() for p in plist]
    if not all_pieces:
        return None
    fy  = _floor_y_ldu(renders)
    xs  = [float(p.pos[0]) for p in all_pieces]
    zs  = [float(p.pos[2]) for p in all_pieces]
    gx0 = (math.floor(min(xs) / GRID_STEP) - GRID_MARGIN) * GRID_STEP
    gx1 = (math.ceil (max(xs) / GRID_STEP) + GRID_MARGIN) * GRID_STEP
    gz0 = (math.floor(min(zs) / GRID_STEP) - GRID_MARGIN) * GRID_STEP
    gz1 = (math.ceil (max(zs) / GRID_STEP) + GRID_MARGIN) * GRID_STEP
    return fy, gx0, gx1, gz0, gz1


def _grid_corner_sx_sy(fy: float, gx0: float, gx1: float,
                       gz0: float, gz1: float) -> list[tuple[float, float]]:
    """Screen-space (sx_px, sy_px) for the four corners of the grid."""
    return [(project_ldraw(np.array([x, fy, z]))[0] * PX_PER_MM,
             project_ldraw(np.array([x, fy, z]))[1] * PX_PER_MM)
            for x, z in [(gx0, gz0), (gx0, gz1), (gx1, gz0), (gx1, gz1)]]


def _draw_isometric_grid(dwg: svgwrite.Drawing, fy: float,
                         gx0: float, gx1: float, gz0: float, gz1: float,
                         cx, cy) -> None:
    """Draw two families of isometric grid lines into dwg."""
    kw = {"stroke": GRID_COLOR, "stroke_width": GRID_WIDTH}
    grp = dwg.g(id="grid")

    def proj(x: float, z: float) -> tuple[str, str]:
        sx, sy, _ = project_ldraw(np.array([x, fy, z]))
        return f"{cx(sx * PX_PER_MM):.1f}", f"{cy(sy * PX_PER_MM):.1f}"

    zs = range(int(gz0), int(gz1) + 1, GRID_STEP)
    xs = range(int(gx0), int(gx1) + 1, GRID_STEP)

    for line in ([dwg.line(proj(gx0, z), proj(gx1, z), **kw) for z in zs] +
                 [dwg.line(proj(x, gz0), proj(x, gz1), **kw) for x in xs]):
        grp.add(line)

    dwg.add(grp)


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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _render_one(
    i: int,
    piece: Piece,
    n: int,
    tmpdir: Path,
    keep_pngs: bool,
) -> tuple[Image.Image, float, float] | None:
    """Render a single piece; return (img, ax, ay) or None on failure."""
    part = PART_MAP[piece.part]
    r, g, b = ldraw_rgb(piece.color)
    scad_src = make_scad(part, (r, g, b), piece.rot)
    png_path = tmpdir / f"piece_{i:03d}_{piece.part}.png"
    print(f"  [{i+1}/{n}] Rendering {piece.part} (color {piece.color}) …", end=" ", flush=True)
    ok = render_piece(scad_src, png_path)
    if ok:
        img, ax, ay = remove_and_crop(png_path)
        print("ok")
        if not keep_pngs:
            png_path.unlink(missing_ok=True)
        return img, ax, ay
    print("FAILED — skipping")
    return None


def build_pngs(
    pieces: list[Piece],
    tmpdir: Path,
    keep_pngs: bool = False,
) -> dict[str, tuple[Image.Image, float, float, list[Piece]]]:
    """Render each unique (part, color, rotation) once; return renders.

    renders maps label → (img, ax, ay, pieces) where pieces is the list of
    all scene pieces sharing that label, in their original order.
    """
    n = len(pieces)

    for i, piece in enumerate(pieces):
        if PART_MAP.get(piece.part) is None:
            print(f"  [{i+1}/{n}] Skipping unknown part: {piece.part}")

    known = [(i, p) for i, p in enumerate(pieces) if PART_MAP.get(p.part) is not None]
    unique_labels = list(dict.fromkeys(_piece_label(p) for _, p in known))

    # First occurrence of each label for rendering (index preserved for PNG filename)
    by_label: dict[str, tuple[int, Piece]] = {
        label: next((i, p) for i, p in known if _piece_label(p) == label)
        for label in unique_labels
    }
    # All pieces sharing each label, preserving scene order
    pieces_by_label: dict[str, list[Piece]] = {
        label: [p for _, p in known if _piece_label(p) == label]
        for label in unique_labels
    }

    # Render each unique label once, attach its piece list
    return {
        label: (*result, pieces_by_label[label])
        for label, (i, piece) in by_label.items()
        if (result := _render_one(i, piece, n, tmpdir, keep_pngs)) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render each LDraw piece with LEGO.scad, compose into SVG."
    )
    parser.add_argument("input", help="Input .ldr file")
    parser.add_argument("-o", "--output", help="Output SVG (default: <input>_lego.svg)")
    parser.add_argument("--keep-pngs", action="store_true",
                        help="Keep per-piece PNGs in a tmp directory")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + "_lego.svg"))

    pieces = parse_ldr(input_path)
    print(f"Found {len(pieces)} pieces in {input_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="ldr2png_"))
    print(f"Rendering pieces (tmpdir: {tmpdir}) …")

    renders = build_pngs(pieces, tmpdir, keep_pngs=args.keep_pngs)

    if not renders:
        print("No pieces rendered — nothing to compose.", file=sys.stderr)
        return

    compose_svg(renders, output_path, padding=60)

    if not args.keep_pngs:
        tmpdir.rmdir()
    else:
        print(f"PNGs kept in: {tmpdir}")


if __name__ == "__main__":
    main()
