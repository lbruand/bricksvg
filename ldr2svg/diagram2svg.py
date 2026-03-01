"""diagram2svg — CLI entry point: render a diagrams-library script as an isometric brick SVG."""

import argparse
import sys
from pathlib import Path

from .diagram_bridge import extract_graph
from .pipeline import run_lego_pipeline


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
    parser.add_argument(
        "--masked", action="store_true",
        help=(
            "Colourise at SVG composition time via alpha mask + "
            "mix-blend-mode:multiply instead of PIL pre-colorisation "
            "(browser/Inkscape only, not supported in LibreOffice)."
        ),
    )
    args = parser.parse_args()

    input_path  = Path(args.input).resolve()
    output_path = (args.output
                   or str(input_path.with_name(input_path.stem + "_bricks.svg")))

    print(f"Extracting graph from {input_path} …")
    graph = extract_graph(str(input_path))

    run_lego_pipeline(
        graph, output_path,
        keep_pngs=args.keep_pngs,
        workers=args.workers,
        masked=args.masked,
    )


if __name__ == "__main__":
    main()
