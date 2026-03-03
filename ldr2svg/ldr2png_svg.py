#!/usr/bin/env python3
"""ldr2png_svg.py - Render each LDraw piece with OpenSCAD, then compose into SVG."""

import os
import sys
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageChops

from .parts import Piece, PART_MAP, ldraw_rgb, parse_ldr
from .scad import make_scad, render_piece, remove_and_crop
from .compose import compose_svg, _piece_label, _piece_label_no_color


def _render_one_white(
    i: int,
    piece: Piece,
    n: int,
    tmpdir: Path,
    keep_pngs: bool,
) -> tuple[Image.Image, float, float] | None:
    """Render one unique (part, rotation) in white; return (img, ax, ay) or None."""
    part = PART_MAP[piece.part]
    scad_src = make_scad(part, (255, 255, 255), piece.rot)
    png_path = tmpdir / f"white_{i:03d}_{piece.part}.png"
    ok = render_piece(scad_src, png_path)
    if ok:
        img, ax, ay = remove_and_crop(png_path)
        print(f"  [{i+1}/{n}] White-render {piece.part} … ok")
        if not keep_pngs:
            png_path.unlink(missing_ok=True)
        return img, ax, ay
    print(f"  [{i+1}/{n}] White-render {piece.part} … FAILED — skipping")
    return None


def build_pngs_white(
    pieces: list[Piece],
    tmpdir: Path,
    keep_pngs: bool = False,
    workers: int | None = None,
) -> dict[str, tuple[Image.Image, float, float, list[Piece]]]:
    """Render each unique (part, rotation) once in white in parallel; return renders.

    Deduplicates by part + rotation only — colour is ignored.  All pieces
    that share the same part+rotation are grouped under one label regardless
    of their actual colour.  Use :func:`colorize_renders` afterwards to
    produce per-colour images via fast PIL multiplication.

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
            executor.submit(_render_one_white, label_idx, piece, n_unique, tmpdir, keep_pngs): label
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


def _colorize(white_img: Image.Image, color_id: int) -> Image.Image:
    """Return a copy of *white_img* tinted to LDraw *color_id*.

    Uses PIL ``ImageChops.multiply`` — identical maths to SVG
    ``mix-blend-mode: multiply``:  result = src × color / 255.
    Lit pixels (white) become the target colour; shadow pixels darken it
    proportionally.  The alpha channel is preserved unchanged.
    """
    r, g, b = ldraw_rgb(color_id)
    solid = Image.new("RGB", white_img.size, (r, g, b))
    colored_rgb = ImageChops.multiply(white_img.convert("RGB"), solid)
    colored_rgb.putalpha(white_img.split()[3])   # restore original alpha
    return colored_rgb.convert("RGBA")


def colorize_renders(
    white_renders: dict[str, tuple[Image.Image, float, float, list[Piece]]],
) -> dict[str, tuple[Image.Image, float, float, list[Piece]]]:
    """Produce per-(part, rotation, color) renders by PIL-multiplying white renders.

    Takes the output of :func:`build_pngs_white` and returns a renders dict
    compatible with the standard SVG composition path.

    Only one OpenSCAD render is needed per unique (part, rotation); colorisation
    is done in Python in microseconds per variant and works in all SVG viewers
    (no CSS blend modes required).
    """
    result: dict[str, tuple[Image.Image, float, float, list[Piece]]] = {}
    for _nc_label, (white_img, ax, ay, pieces) in white_renders.items():
        by_color: dict[int, list[Piece]] = {}
        for p in pieces:
            by_color.setdefault(p.color, []).append(p)
        for color_id, color_pieces in by_color.items():
            colored_img = _colorize(white_img, color_id)
            label = _piece_label(color_pieces[0])
            result[label] = (colored_img, ax, ay, color_pieces)
    return result


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
    parser.add_argument(
        "--masked", action="store_true",
        help=(
            "Colourise at SVG composition time via feColorMatrix duotone filters "
            "instead of PIL pre-colorisation. Stores one grayscale image per "
            "unique shape; each colour is a separate SVG filter."
        ),
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = args.output or str(input_path.with_name(input_path.stem + "_bricks.svg"))

    pieces = parse_ldr(input_path)
    print(f"Found {len(pieces)} pieces in {input_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="ldr2png_"))
    print(f"White-rendering pieces (tmpdir: {tmpdir}) …")

    white_renders = build_pngs_white(pieces, tmpdir,
                                     keep_pngs=args.keep_pngs, workers=args.workers)
    if not white_renders:
        print("No pieces rendered — nothing to compose.", file=sys.stderr)
        return

    if args.masked:
        renders = white_renders
    else:
        print("Colorising renders with PIL …")
        renders = colorize_renders(white_renders)

    compose_svg(renders, output_path, padding=60, masked=args.masked)

    if not args.keep_pngs:
        tmpdir.rmdir()
    else:
        print(f"PNGs kept in: {tmpdir}")


if __name__ == "__main__":
    main()
