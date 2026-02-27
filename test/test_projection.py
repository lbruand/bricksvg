#!/usr/bin/env python3
"""test_projection.py — Integration test for the LDraw→screen projection.

Renders all LDraw pieces at their true 3D world positions in a single OpenSCAD
scene (the "reference render").  Small coloured spheres are placed at each
piece's LDraw origin.  project_ldraw() is then used to project the same
origins and the pixel error is asserted to be within tolerance.

Also usable as a standalone script:
    uv run python test/test_projection.py [file.ldr] [out_annotated.png]
"""

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from ldr2svg.parts import parse_ldr, PART_MAP, ldraw_rgb
from ldr2svg.projection import _T, LDU_TO_MM, PX_PER_MM, CAMERA_RX, CAMERA_RZ, CAMERA_D, IMG_PX, project_ldraw
from ldr2svg.scad import LEGOLIB

# ---------------------------------------------------------------------------
# reference-scene SCAD builder
# ---------------------------------------------------------------------------
MARKER_COLORS_SCAD = [
    "[1,0,0]", "[0,1,0]", "[0,0,1]",
    "[1,1,0]", "[1,0,1]", "[0,1,1]",
]
MARKER_COLORS_PIL = [
    "red", "lime", "blue",
    "yellow", "magenta", "cyan",
]
MARKER_R_MM = 2.5   # sphere radius for origin markers

LDR_PATH = Path(__file__).parent.parent / "test.ldr"

# Tolerances
X_TOLERANCE = 10.0   # px — horizontal projection error
Y_TOLERANCE = 15.0   # px — includes ~8.8 px constant vertical offset


def build_reference_scad(pieces: list) -> str:
    """Return a SCAD string that places every known piece at its world position
    plus a small coloured sphere at each piece's LDraw origin."""
    lines = [f"use <{LEGOLIB}>", "$fs = 1.0; $fa = 8;", ""]

    for i, piece in enumerate(pieces):
        part = PART_MAP.get(piece.part)
        if part is None:
            continue

        r, g, b = ldraw_rgb(piece.color)
        piece_color = f"[{r/255:.3f},{g/255:.3f},{b/255:.3f}]"

        pos_os = _T @ piece.pos * LDU_TO_MM
        R_os   = _T @ piece.rot @ _T

        def fmt_row(row):
            return f"[{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},0]"
        mat = (f"[{fmt_row(R_os[0])},{fmt_row(R_os[1])},"
               f"{fmt_row(R_os[2])},[0,0,0,1]]")

        px, py, pz = pos_os
        h = part.h_mm

        lines += [
            f"// [{i}] {piece.part}  ld={piece.pos.astype(int).tolist()}",
            f"color({piece_color})",
            f"translate([{px:.4f},{py:.4f},{pz:.4f}])",
            f"  multmatrix({mat})",
            f"  translate([0,0,{-h:.4f}])",
            f"  block(width={part.width},length={part.length},",
            f"        height={part.height:.6f},type=\"{part.block_type}\");",
            "",
        ]

        mc = MARKER_COLORS_SCAD[i % len(MARKER_COLORS_SCAD)]
        lines += [
            f"color({mc})",
            f"translate([{px:.4f},{py:.4f},{pz:.4f}])",
            f"  sphere(r={MARKER_R_MM},$fn=16);",
            "",
        ]

    return "\n".join(lines)


def render_scad_to_png(scad_src: str, png_out: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".scad", mode="w", delete=False) as f:
        f.write(scad_src)
        scad_path = Path(f.name)
    try:
        cam = f"0,0,0,{CAMERA_RX},0,{CAMERA_RZ},{CAMERA_D}"
        result = subprocess.run(
            ["openscad",
             "--camera", cam,
             "--projection", "ortho",
             "--imgsize", f"{IMG_PX},{IMG_PX}",
             "-o", str(png_out),
             str(scad_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"OpenSCAD error:\n{result.stderr[-600:]}", file=sys.stderr)
            return False
        return png_out.exists()
    finally:
        scad_path.unlink(missing_ok=True)


def annotate(ref_png: Path, pieces: list, out_png: Path) -> None:
    """Draw projected crosshairs over the reference render."""
    img  = Image.open(ref_png).convert("RGBA")
    draw = ImageDraw.Draw(img)
    cx = cy = IMG_PX / 2

    for i, piece in enumerate(pieces):
        if piece.part not in PART_MAP:
            continue
        sx, sy, _ = project_ldraw(piece.pos)
        px = cx + sx * PX_PER_MM
        py = cy + sy * PX_PER_MM
        col = MARKER_COLORS_PIL[i % len(MARKER_COLORS_PIL)]
        R   = 10
        draw.line([(px - R*2, py), (px + R*2, py)], fill=col, width=2)
        draw.line([(px, py - R*2), (px, py + R*2)], fill=col, width=2)
        draw.ellipse([(px - R, py - R), (px + R, py + R)], outline=col, width=2)

    img.save(out_png)


# ---------------------------------------------------------------------------
# sphere-centroid measurement (ground truth from reference render)
# ---------------------------------------------------------------------------
_COLOR_RANGES = {
    "red":     ((160, 255), (  0,  80), (  0,  80)),
    "lime":    ((  0,  80), (160, 255), (  0,  80)),
    "blue":    ((  0,  80), (  0,  80), (160, 255)),
    "yellow":  ((160, 255), (160, 255), (  0,  80)),
    "magenta": ((160, 255), (  0,  80), (160, 255)),
    "cyan":    ((  0,  80), (160, 255), (160, 255)),
}


def measure_sphere_centroids(ref_png: Path) -> dict[str, tuple[float, float]]:
    """Return {color_name: (canvas_x, canvas_y)} for each colored sphere."""
    arr = np.array(Image.open(ref_png).convert("RGB"))
    result = {}
    for name, (rr, gr, br) in _COLOR_RANGES.items():
        mask = (
            (arr[:, :, 0] >= rr[0]) & (arr[:, :, 0] <= rr[1]) &
            (arr[:, :, 1] >= gr[0]) & (arr[:, :, 1] <= gr[1]) &
            (arr[:, :, 2] >= br[0]) & (arr[:, :, 2] <= br[1])
        )
        ys, xs = np.where(mask)
        if len(xs) >= 5:
            result[name] = (float(xs.mean()), float(ys.mean()))
    return result


# ---------------------------------------------------------------------------
# pytest fixtures and test
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ref_png(tmp_path_factory):
    pieces = parse_ldr(LDR_PATH)
    scad   = build_reference_scad(pieces)
    png    = tmp_path_factory.mktemp("projection") / "ref_scene.png"
    ok     = render_scad_to_png(scad, png)
    assert ok, "OpenSCAD reference render failed"
    return png


@pytest.fixture(scope="session")
def centroids(ref_png):
    return measure_sphere_centroids(ref_png)


@pytest.mark.slow
def test_projection_accuracy(centroids):
    pieces = parse_ldr(LDR_PATH)
    known  = [p for p in pieces if p.part in PART_MAP]
    cx = cy = IMG_PX / 2

    for i, piece in enumerate(known):
        sx, sy, _ = project_ldraw(piece.pos)
        proj_x = cx + sx * PX_PER_MM
        proj_y = cy + sy * PX_PER_MM
        col  = MARKER_COLORS_PIL[i % len(MARKER_COLORS_PIL)]
        meas = centroids.get(col)
        assert meas is not None, f"No sphere detected for {piece.part} ({col})"
        ex = proj_x - meas[0]
        ey = proj_y - meas[1]
        assert abs(ex) < X_TOLERANCE, (
            f"{piece.part}: X error {ex:+.1f} px exceeds ±{X_TOLERANCE} px"
        )
        assert abs(ey) < Y_TOLERANCE, (
            f"{piece.part}: Y error {ey:+.1f} px exceeds ±{Y_TOLERANCE} px"
        )


# ---------------------------------------------------------------------------
# standalone script entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Visual projection accuracy test.")
    parser.add_argument("ldr", nargs="?", default=str(LDR_PATH), help="Input .ldr file")
    parser.add_argument("--annotate", metavar="FILE", nargs="?",
                        const="/tmp/ref_annotated.png",
                        help="Save annotated image (default: /tmp/ref_annotated.png)")
    args = parser.parse_args()

    ldr_path = Path(args.ldr)
    ref_png  = Path("/tmp/ref_scene.png")

    pieces = parse_ldr(ldr_path)
    known  = [p for p in pieces if p.part in PART_MAP]
    print(f"{ldr_path}: {len(known)} known pieces")

    print("Building reference SCAD …")
    scad = build_reference_scad(pieces)

    print("Rendering reference PNG …")
    ok = render_scad_to_png(scad, ref_png)
    if not ok:
        sys.exit(1)
    print(f"  → {ref_png}")

    if args.annotate:
        ann_png = Path(args.annotate)
        annotate(ref_png, pieces, ann_png)
        print(f"  → {ann_png} (annotated)")
        print("Crosshair colour = sphere colour for each piece.")
        print("If projection is correct, every crosshair sits on its sphere centre.")
    print()

    print("Measuring sphere centroids from reference render …")
    measured = measure_sphere_centroids(ref_png)

    cx = cy = IMG_PX / 2
    print()
    print(f"{'part':8s}  {'color':8s}  {'measured px':>16s}  {'projected px':>16s}  {'error px':>12s}")
    print("-" * 72)
    for i, p in enumerate(known):
        sx, sy, depth = project_ldraw(p.pos)
        proj_x = cx + sx * PX_PER_MM
        proj_y = cy + sy * PX_PER_MM
        col = MARKER_COLORS_PIL[i % len(MARKER_COLORS_PIL)]
        meas = measured.get(col)
        if meas:
            ex, ey   = proj_x - meas[0], proj_y - meas[1]
            meas_str = f"({meas[0]:6.1f},{meas[1]:6.1f})"
            err_str  = f"({ex:+6.1f},{ey:+6.1f})"
        else:
            meas_str = "       n/a      "
            err_str  = "      n/a     "
        proj_str = f"({proj_x:6.1f},{proj_y:6.1f})"
        print(f"{p.part:8s}  {col:8s}  {meas_str:>16s}  {proj_str:>16s}  {err_str:>14s}")


if __name__ == "__main__":
    main()