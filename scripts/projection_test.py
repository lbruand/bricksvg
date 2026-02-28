#!/usr/bin/env python3
"""Standalone projection accuracy visualiser.

Renders the reference scene, measures sphere centroids, and prints a
per-piece error table.  Optionally saves an annotated PNG.

Run from the project root:
    uv run python scripts/projection_test.py [file.ldr] [--annotate [FILE]]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test.test_projection import (
    LDR_PATH,
    MARKER_COLORS_PIL,
    annotate,
    build_reference_scad,
    measure_sphere_centroids,
    render_scad_to_png,
)
from ldr2svg.parts import PART_MAP, parse_ldr
from ldr2svg.projection import IMG_PX, PX_PER_MM, project_ldraw


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