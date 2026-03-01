#!/usr/bin/env python3
"""diagram2svg.py — Render a diagrams-library script as an isometric brick SVG.

Usage
-----
    uv run python scripts/diagram2svg.py <diagram.py> [-o output.svg]
"""

import argparse
import sys
import tempfile
from pathlib import Path

# Allow running from the project root without a package install
sys.path.insert(0, str(Path(__file__).parent.parent))

from ldr2svg.diagram_bridge import extract_graph, build_ldr_scene
from ldr2svg.diagram_compose import compose_diagram_svg
from ldr2svg.ldr2png_svg import build_pngs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a diagrams-library Python script as an isometric brick SVG."
    )
    parser.add_argument("input",  help="Input diagram.py script path")
    parser.add_argument("-o", "--output", help="Output SVG path (default: <input>_bricks.svg)")
    parser.add_argument(
        "--keep-pngs", action="store_true",
        help="Keep per-piece PNG files in the temp directory",
    )
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Parallel render workers (default: cpu count)")
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = (args.output
                   or str(input_path.with_name(input_path.stem + "_bricks.svg")))

    print(f"Extracting graph from {input_path} …")
    graph = extract_graph(str(input_path))

    print("Building LDraw scene …")
    pieces, arrows, node_data, piece_groups, cluster_data = build_ldr_scene(graph)
    print(f"  {len(pieces)} pieces, {len(arrows)} edges, {len(node_data)} nodes, {len(cluster_data)} clusters")

    tmpdir = Path(tempfile.mkdtemp(prefix="diagram2svg_"))
    print(f"Rendering pieces (tmpdir: {tmpdir}) …")
    renders = build_pngs(pieces, tmpdir, keep_pngs=args.keep_pngs, workers=args.workers)

    if not renders:
        print("No pieces rendered — nothing to compose.", file=sys.stderr)
        return

    compose_diagram_svg(renders, output_path, arrows, node_data,
                        piece_groups=piece_groups, cluster_data=cluster_data)

    if not args.keep_pngs:
        try:
            tmpdir.rmdir()
        except OSError:
            pass  # tmpdir not empty (e.g., OpenSCAD left files)
    else:
        print(f"PNGs kept in: {tmpdir}")


if __name__ == "__main__":
    main()
