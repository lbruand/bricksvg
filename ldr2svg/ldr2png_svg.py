#!/usr/bin/env python3
"""ldr2png_svg.py - Render each LDraw piece with OpenSCAD, then compose into SVG."""

import sys
import argparse
import tempfile
from pathlib import Path

from PIL import Image

from .parts import Piece, PART_MAP, ldraw_rgb, parse_ldr
from .scad import make_scad, render_piece, remove_and_crop
from .compose import compose_svg, _piece_label


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

    by_label: dict[str, tuple[int, Piece]] = {
        label: next((i, p) for i, p in known if _piece_label(p) == label)
        for label in unique_labels
    }
    pieces_by_label: dict[str, list[Piece]] = {
        label: [p for _, p in known if _piece_label(p) == label]
        for label in unique_labels
    }

    return {
        label: (*result, pieces_by_label[label])
        for label, (i, piece) in by_label.items()
        if (result := _render_one(i, piece, n, tmpdir, keep_pngs)) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render each LDraw piece with OpenSCAD, compose into SVG."
    )
    parser.add_argument("input", help="Input .ldr file")
    parser.add_argument("-o", "--output", help="Output SVG (default: <input>_bricks.svg)")
    parser.add_argument("--keep-pngs", action="store_true",
                        help="Keep per-piece PNGs in a tmp directory")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + "_bricks.svg"))

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
