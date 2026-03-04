"""pipeline.py — Shared rendering pipeline used by diagram2svg and mermaid2svg."""

import sys
import tempfile
from pathlib import Path

from .diagram_bridge import build_ldr_scene
from .diagram_compose import compose_diagram_svg
from .ldr2png_svg import build_pngs_grayscale


def run_lego_pipeline(
    graph: dict,
    output_path: str,
    *,
    keep_pngs: bool = False,
    workers: int | None = None,
) -> None:
    """Build pieces, render PNGs, and compose the isometric brick SVG.

    Parameters
    ----------
    graph:
        Dict returned by either ``diagram_bridge.extract_graph`` or
        ``mermaid_bridge.extract_graph`` — the graphviz JSON layout.
    output_path:
        Destination SVG file path.
    keep_pngs:
        If True, leave intermediate PNG files in the temp directory.
    workers:
        Number of parallel OpenSCAD render workers (default: cpu count).
    """
    print("Building LDraw scene …")
    pieces, arrows, node_data, piece_groups, cluster_data = build_ldr_scene(graph)
    print(
        f"  {len(pieces)} pieces, {len(arrows)} edges, "
        f"{len(node_data)} nodes, {len(cluster_data)} clusters"
    )

    tmpdir = Path(tempfile.mkdtemp(prefix="lego_render_"))
    print(f"Grayscale-rendering pieces (tmpdir: {tmpdir}) …")
    grayscale_renders = build_pngs_grayscale(pieces, tmpdir, keep_pngs=keep_pngs, workers=workers)

    if not grayscale_renders:
        print("No pieces rendered — nothing to compose.", file=sys.stderr)
        return

    compose_diagram_svg(
        grayscale_renders, output_path, arrows, node_data,
        piece_groups=piece_groups, cluster_data=cluster_data,
    )

    if not keep_pngs:
        try:
            tmpdir.rmdir()
        except OSError:
            pass
    else:
        print(f"PNGs kept in: {tmpdir}")
