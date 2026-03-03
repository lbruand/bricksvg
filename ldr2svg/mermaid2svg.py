"""mermaid2svg — CLI entry point: render a Mermaid flowchart as an isometric brick SVG."""

import argparse
from pathlib import Path

from .mermaid_bridge import extract_graph
from .pipeline import run_lego_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Mermaid flowchart as an isometric brick SVG."
    )
    parser.add_argument("input",  help="Input Mermaid file path (.mmd)")
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

    run_lego_pipeline(
        graph, output_path,
        keep_pngs=args.keep_pngs,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
