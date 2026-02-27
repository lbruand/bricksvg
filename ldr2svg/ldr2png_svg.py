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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LEGOLIB = Path(__file__).parent / "LEGO.scad"

# ---------------------------------------------------------------------------
# Camera parameters (OpenSCAD)
# ---------------------------------------------------------------------------
# Standard LEGO instruction-style view:  rx=60 tilt, rz=45 spin, ortho
CAMERA_RX = 60.0   # degrees tilt (around X)
CAMERA_RZ = 45.0   # degrees spin (around Z)
CAMERA_D  = 300.0  # camera distance (mm) — controls scale
IMG_PX    = 800    # render each piece into a square IMG_PX × IMG_PX PNG

# OpenSCAD ortho scale: viewport covers 2*D*tan(fov/2) mm → IMG_PX pixels.
# OpenSCAD default full perspective FOV = 22.5°.
_OPENSCAD_FOV_DEG = 22.5
PX_PER_MM = IMG_PX / (2 * CAMERA_D * math.tan(math.radians(_OPENSCAD_FOV_DEG / 2)))

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
    piece: Piece, img: Image.Image, anchor_x: float, anchor_y: float,
) -> tuple[float, float, float, float, Image.Image, float, float, int, int, str]:
    """Project one piece to a screen-space row: (depth, ldy, sx_px, sy_px, img, ax, ay, iw, ih, label)."""
    sx, sy, depth = project_ldraw(piece.pos)
    iw, ih = img.size
    return (depth, float(piece.pos[1]), sx * PX_PER_MM, sy * PX_PER_MM,
            img, anchor_x, anchor_y, iw, ih, _piece_label(piece))


def _project_pieces(
    pngs: list[tuple[Piece, Image.Image, float, float]],
) -> list[tuple[float, float, float, float, int, int, str]]:
    """Project each piece to screen coords and sort back-to-front.

    Returns (sx_px, sy_px, ax, ay, iw, ih, label) per piece.
    """
    # Sort back-to-front: primary = LDraw Y descending (lower pieces first, elevated last),
    # secondary = cam depth ascending (farther first) for same-height pieces.
    rows = sorted([_project_piece(*t) for t in pngs], key=lambda t: (-t[1], t[0]))
    return [(sx, sy, ax, ay, iw, ih, label)
            for _, _, sx, sy, _, ax, ay, iw, ih, label in rows]


def _canvas_bounds(
    projected: list[tuple],
    padding: int,
) -> tuple[int, int, float, float]:
    """Return (W, H, min_x, min_y) for the SVG canvas."""
    xs = ([sx - ax       for sx, _,  ax, _,  iw, _,  _ in projected] +
          [sx - ax + iw  for sx, _,  ax, _,  iw, _,  _ in projected])
    ys = ([sy - ay       for _,  sy, _,  ay, _,  ih, _ in projected] +
          [sy - ay + ih  for _,  sy, _,  ay, _,  ih, _ in projected])
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    W = int(max_x - min_x + 2 * padding)
    H = int(max_y - min_y + 2 * padding)
    return W, H, min_x, min_y


def _build_defs(
    dwg: svgwrite.Drawing,
    renders: dict[str, tuple["Image.Image", float, float]],
) -> dict[str, tuple[str, float, float]]:
    """Add each unique piece image to <defs> once; return label→(def_id, ax, ay)."""
    defs: dict[str, tuple[str, float, float]] = {}
    for label, (img, anchor_x, anchor_y) in renders.items():
        part_name  = label.split()[0]
        short_hash = hashlib.sha256(label.encode()).hexdigest()[:8]
        def_id     = f"{part_name}-{short_hash}"
        iw, ih     = img.size
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        uri = f"data:image/png;base64,{b64}"
        dwg.defs.add(dwg.image(
            uri,
            insert=("0px", "0px"),
            size=(f"{iw}px", f"{ih}px"),
            id=def_id,
        ))
        defs[label] = (def_id, anchor_x, anchor_y)
    return defs


def _inject_def_comments(output: str, defs: dict[str, tuple]) -> None:
    """Insert <!-- label --> before each <image element in <defs>."""
    it = iter(defs.keys())
    svg_text = Path(output).read_text()
    svg_text = re.sub(r"(<image\b)", lambda m: f"<!-- {next(it)} -->\n      {m.group(1)}", svg_text)
    Path(output).write_text(svg_text)


def compose_svg(
    pngs: list[tuple[Piece, Image.Image, float, float]],
    renders: dict[str, tuple[Image.Image, float, float]],
    output: str,
    padding: int = 60,
) -> None:
    projected          = _project_pieces(pngs)
    W, H, min_x, min_y = _canvas_bounds(projected, padding)

    def cx(val): return val - min_x + padding
    def cy(val): return val - min_y + padding

    dwg = svgwrite.Drawing(output, size=(f"{W}px", f"{H}px"))
    dwg.add(dwg.rect((0, 0), ("100%", "100%"), fill="#f8f8f0"))

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
def build_pngs(
    pieces: list[Piece],
    tmpdir: Path,
    keep_pngs: bool = False,
) -> tuple[list[tuple[Piece, Image.Image, float, float]], dict[str, tuple[Image.Image, float, float]]]:
    """Render each piece to a PNG (with caching); return (pngs, renders).

    renders maps label → (img, ax, ay) for each unique (part, color, rotation).
    """
    pngs: list[tuple[Piece, Image.Image, float, float]] = []
    renders: dict[str, tuple[Image.Image, float, float]] = {}

    for i, piece in enumerate(pieces):
        part = PART_MAP.get(piece.part)
        if part is None:
            print(f"  [{i+1}/{len(pieces)}] Skipping unknown part: {piece.part}")
        else:
            label = _piece_label(piece)
            if label in renders:
                img, anchor_x, anchor_y = renders[label]
                pngs.append((piece, img, anchor_x, anchor_y))
                print(f"  [{i+1}/{len(pieces)}] Rendering {piece.part} (color {piece.color}) … cached")
            else:
                r, g, b = ldraw_rgb(piece.color)
                scad_src = make_scad(part, (r, g, b), piece.rot)
                png_path = tmpdir / f"piece_{i:03d}_{piece.part}.png"

                print(f"  [{i+1}/{len(pieces)}] Rendering {piece.part} (color {piece.color}) …",
                      end=" ", flush=True)
                ok = render_piece(scad_src, png_path)
                if ok:
                    img, anchor_x, anchor_y = remove_and_crop(png_path)
                    renders[label] = (img, anchor_x, anchor_y)
                    pngs.append((piece, img, anchor_x, anchor_y))
                    print("ok")
                    if not keep_pngs:
                        png_path.unlink(missing_ok=True)
                else:
                    print("FAILED — skipping")

    return pngs, renders


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

    pngs, renders = build_pngs(pieces, tmpdir, keep_pngs=args.keep_pngs)

    if not pngs:
        print("No pieces rendered — nothing to compose.", file=sys.stderr)
        return

    compose_svg(pngs, renders, output_path, padding=60)

    if not args.keep_pngs:
        tmpdir.rmdir()
    else:
        print(f"PNGs kept in: {tmpdir}")


if __name__ == "__main__":
    main()
