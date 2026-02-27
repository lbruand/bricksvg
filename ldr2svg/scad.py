"""scad.py — SCAD file generation, OpenSCAD rendering, background removal."""

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .parts import PartDef
from .projection import CAMERA_RX, CAMERA_RZ, CAMERA_D, IMG_PX, _T

LEGOLIB  = Path(__file__).parent / "LEGO.scad"
CROP_PAD = 6   # extra transparent pixels around each cropped piece

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
    r_str   = f"[{r[0]/255:.3f}, {r[1]/255:.3f}, {r[2]/255:.3f}]"
    tz      = -part.h_mm
    h_param = f"{part.height:.6f}"
    R_os    = _T @ rot_ld @ _T

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
def remove_and_crop(png_path: Path) -> tuple[Image.Image, float, float]:
    """Remove background, crop tightly, return (img, anchor_x, anchor_y).

    anchor_x/y is the position of the piece origin (= image centre before
    cropping) inside the returned cropped image.
    """
    img  = Image.open(png_path).convert("RGBA")
    arr  = np.array(img)
    bg   = arr[0, 0, :3]

    tol  = 12
    mask = np.all(np.abs(arr[:, :, :3].astype(int) - bg.astype(int)) <= tol, axis=2)
    arr[mask, 3] = 0

    alpha = arr[:, :, 3]
    rows  = np.any(alpha > 0, axis=1)
    cols  = np.any(alpha > 0, axis=0)
    if not rows.any():
        h, w = arr.shape[:2]
        return Image.fromarray(arr, "RGBA"), w / 2, h / 2

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    H, W = arr.shape[:2]
    r0c  = max(0, r0 - CROP_PAD)
    r1c  = min(H, r1 + CROP_PAD + 1)
    c0c  = max(0, c0 - CROP_PAD)
    c1c  = min(W, c1 + CROP_PAD + 1)

    cropped  = Image.fromarray(arr[r0c:r1c, c0c:c1c], "RGBA")
    anchor_x = IMG_PX / 2 - c0c
    anchor_y = IMG_PX / 2 - r0c
    return cropped, anchor_x, anchor_y
