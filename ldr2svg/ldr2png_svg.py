#!/usr/bin/env python3
"""ldr2png_svg.py - Render each LDraw piece with OpenSCAD, then compose into SVG."""

import os
import sys
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

from .parts import Piece, PART_MAP, parse_ldr
from .scad import make_scad, render_piece, remove_and_crop
from .compose import compose_svg, _piece_label_no_color


def _render_one_grayscale(
    i: int,
    piece: Piece,
    n: int,
    tmpdir: Path,
    keep_pngs: bool,
) -> tuple[Image.Image, float, float] | None:
    """Render one unique (part, rotation) in grayscale; return (img, ax, ay) or None."""
    part = PART_MAP[piece.part]
    scad_src = make_scad(part, (255, 255, 255), piece.rot)
    png_path = tmpdir / f"grayscale_{i:03d}_{piece.part}.png"
    ok = render_piece(scad_src, png_path)
    if ok:
        img, ax, ay = remove_and_crop(png_path)
        print(f"  [{i+1}/{n}] Grayscale-render {piece.part} … ok")
        if not keep_pngs:
            png_path.unlink(missing_ok=True)
        return img, ax, ay
    print(f"  [{i+1}/{n}] Grayscale-render {piece.part} … FAILED — skipping")
    return None


def build_pngs_grayscale(
    pieces: list[Piece],
    tmpdir: Path,
    keep_pngs: bool = False,
    workers: int | None = None,
) -> dict[str, tuple[Image.Image, float, float, list[Piece]]]:
    """Render each unique (part, rotation) once in grayscale in parallel; return renders.

    Deduplicates by part + rotation only — colour is ignored.  All pieces
    that share the same part+rotation are grouped under one label regardless
    of their actual colour.  Colour is applied at SVG composition time via
    ``feColorMatrix`` duotone filters.

    Returns ``label → (img, ax, ay, pieces)`` where *pieces* lists every
    scene piece with that part+rotation (potentially spanning multiple colours).

    *workers* controls the thread-pool size (default: ``os.cpu_count()``).
    """
    known = [(i, p) for i, p in enumerate(pieces) if PART_MAP.get(p.part) is not None]

    for i, piece in enumerate(pieces):
        if PART_MAP.get(piece.part) is None:
            print(f"  [{i+1}/{len(pieces)}] Skipping unknown part: {piece.part}")

    unique_labels = list(dict.fromkeys(_piece_label_no_color(p) for _, p in known))
    n_unique = len(unique_labels)

    by_label: dict[str, tuple[int, Piece]] = {}
    for i, p in known:
        lbl = _piece_label_no_color(p)
        if lbl not in by_label:
            by_label[lbl] = (i, p)

    pieces_by_label: dict[str, list[Piece]] = {lbl: [] for lbl in unique_labels}
    for _, p in known:
        pieces_by_label[_piece_label_no_color(p)].append(p)

    max_workers = workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {
            executor.submit(_render_one_grayscale, label_idx, piece, n_unique, tmpdir, keep_pngs): label
            for label_idx, (label, (_, piece)) in enumerate(by_label.items())
        }
        results: dict[str, tuple[Image.Image, float, float] | None] = {
            future_to_label[f]: f.result()
            for f in as_completed(future_to_label)
        }

    return {
        label: (*r, pieces_by_label[label])
        for label in unique_labels
        if (r := results.get(label)) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render each LDraw piece with OpenSCAD, compose into SVG."
    )
    parser.add_argument("input", help="Input .ldr file")
    parser.add_argument("-o", "--output", help="Output SVG (default: <input>_bricks.svg)")
    parser.add_argument("--keep-pngs", action="store_true",
                        help="Keep per-piece PNGs in a tmp directory")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Parallel render workers (default: cpu count)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + "_bricks.svg"))

    pieces = parse_ldr(input_path)
    print(f"Found {len(pieces)} pieces in {input_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="ldr2png_"))
    print(f"Grayscale-rendering pieces (tmpdir: {tmpdir}) …")

    grayscale_renders = build_pngs_grayscale(pieces, tmpdir,
                                     keep_pngs=args.keep_pngs, workers=args.workers)
    if not grayscale_renders:
        print("No pieces rendered — nothing to compose.", file=sys.stderr)
        return

    compose_svg(grayscale_renders, output_path, padding=60)

    if not args.keep_pngs:
        tmpdir.rmdir()
    else:
        print(f"PNGs kept in: {tmpdir}")


if __name__ == "__main__":
    main()
