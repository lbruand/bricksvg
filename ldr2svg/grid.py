"""grid.py — Isometric floor grid for SVG output."""

import math

import numpy as np
import svgwrite

from .parts import PART_MAP
from .projection import project_ldraw, PX_PER_MM, LDU_TO_MM

GRID_STEP   = 20      # LDU = 1 LEGO stud = 8 mm
GRID_MARGIN = 2       # extra grid cells around scene bounding box
GRID_COLOR  = "#c0b8b0"
GRID_WIDTH  = "0.5"


def _floor_y_ldu(renders: dict) -> float:
    """LDraw Y of the floor = max bottom-face Y across all known pieces."""
    ys = [float(p.pos[1]) + PART_MAP[p.part].h_mm / LDU_TO_MM
          for _img, _ax, _ay, plist in renders.values()
          for p in plist if p.part in PART_MAP]
    return max(ys, default=0.0)


def _grid_params(renders: dict) -> tuple | None:
    """Return (floor_y, gx0, gx1, gz0, gz1) in LDU, or None if no pieces."""
    all_pieces = [p for _img, _ax, _ay, plist in renders.values() for p in plist]
    if not all_pieces:
        return None
    fy  = _floor_y_ldu(renders)
    xs  = [float(p.pos[0]) for p in all_pieces]
    zs  = [float(p.pos[2]) for p in all_pieces]
    gx0 = (math.floor(min(xs) / GRID_STEP) - GRID_MARGIN) * GRID_STEP
    gx1 = (math.ceil (max(xs) / GRID_STEP) + GRID_MARGIN) * GRID_STEP
    gz0 = (math.floor(min(zs) / GRID_STEP) - GRID_MARGIN) * GRID_STEP
    gz1 = (math.ceil (max(zs) / GRID_STEP) + GRID_MARGIN) * GRID_STEP
    return fy, gx0, gx1, gz0, gz1


def _grid_corner_sx_sy(fy: float, gx0: float, gx1: float,
                       gz0: float, gz1: float) -> list[tuple[float, float]]:
    """Screen-space (sx_px, sy_px) for the four corners of the grid."""
    return [(project_ldraw(np.array([x, fy, z]))[0] * PX_PER_MM,
             project_ldraw(np.array([x, fy, z]))[1] * PX_PER_MM)
            for x, z in [(gx0, gz0), (gx0, gz1), (gx1, gz0), (gx1, gz1)]]


def _draw_isometric_grid(dwg: svgwrite.Drawing, fy: float,
                         gx0: float, gx1: float, gz0: float, gz1: float,
                         cx, cy) -> None:
    """Draw two families of isometric grid lines into dwg."""
    kw  = {"stroke": GRID_COLOR, "stroke_width": GRID_WIDTH}
    grp = dwg.g(id="grid")

    def proj(x: float, z: float) -> tuple[str, str]:
        sx, sy, _ = project_ldraw(np.array([x, fy, z]))
        return f"{cx(sx * PX_PER_MM):.1f}", f"{cy(sy * PX_PER_MM):.1f}"

    zs = range(int(gz0), int(gz1) + 1, GRID_STEP)
    xs = range(int(gx0), int(gx1) + 1, GRID_STEP)

    grp.elements.extend([
        *[dwg.line(proj(gx0, z), proj(gx1, z), **kw) for z in zs],
        *[dwg.line(proj(x, gz0), proj(x, gz1), **kw) for x in xs],
    ])

    dwg.add(grp)
