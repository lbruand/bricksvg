#!/usr/bin/env python3
"""ldr2svg - Render LDraw/LeoCAD scenes to clean LEGO illustration SVGs."""

import sys
import argparse
from pathlib import Path

import numpy as np
import svgwrite

# ---------------------------------------------------------------------------
# LDraw library paths
# ---------------------------------------------------------------------------
LDRAW_ROOTS = [Path("/usr/share/ldraw")]

SEARCH_DIRS: list[Path] = []
for _root in LDRAW_ROOTS:
    SEARCH_DIRS += [
        _root / "parts",
        _root / "p",
        _root / "parts" / "s",
        _root / "p" / "48",
        _root / "p" / "8",
        _root,
    ]

# ---------------------------------------------------------------------------
# LDraw official colour table (id -> #RRGGBB)
# ---------------------------------------------------------------------------
LDRAW_COLORS: dict[int, str] = {
    0: "#05131D",   1: "#0055BF",   2: "#257A3E",   3: "#008555",
    4: "#C91A09",   5: "#C870A0",   6: "#583A20",   7: "#9B9B9B",
    8: "#646464",   9: "#B4D2E3",  10: "#4B974B",  11: "#55A5AF",
   12: "#FF7700",  13: "#F7BCD5",  14: "#F2CD37",  15: "#FFFFFF",
   17: "#BEE90F",  18: "#E6B843",  19: "#F2CDA1",  20: "#C9E0E0",
   22: "#81096E",  25: "#FE8A18",  26: "#B8005E",  27: "#EEFFB3",
   28: "#AD7235",  70: "#5C1D0D",  71: "#969696",  72: "#6C6E68",
   85: "#542071",  86: "#643300",  92: "#B56A37", 288: "#004721",
  320: "#720E0E", 321: "#36ACC8", 322: "#5CC3C3", 323: "#F3F3EE",
  484: "#A05230",
}

EDGE_COLOR_ID = 24   # LDraw "complement" edge colour


def ldraw_hex(color_id: int) -> str:
    return LDRAW_COLORS.get(color_id, "#888888")


def shade_hex(hex_color: str, factor: float) -> str:
    """Scale RGB channels by *factor* (clamp to 0-255)."""
    r = max(0, min(255, int(int(hex_color[1:3], 16) * factor)))
    g = max(0, min(255, int(int(hex_color[3:5], 16) * factor)))
    b = max(0, min(255, int(int(hex_color[5:7], 16) * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Geometry containers
# ---------------------------------------------------------------------------
class Face:
    __slots__ = ("verts", "color_id")

    def __init__(self, verts: np.ndarray, color_id: int):
        self.verts = verts      # (N, 3) float64
        self.color_id = color_id


class Edge:
    __slots__ = ("verts", "color_id")

    def __init__(self, verts: np.ndarray, color_id: int):
        self.verts = verts      # (2, 3) float64
        self.color_id = color_id


# ---------------------------------------------------------------------------
# LDraw file loader
# ---------------------------------------------------------------------------
_file_cache: dict[str, list] = {}


def find_part(name: str) -> Path | None:
    name_fwd = name.replace("\\", "/")
    for d in SEARCH_DIRS:
        for candidate in (d / name_fwd, d / name_fwd.lower()):
            if candidate.exists():
                return candidate
    return None


def load_file(
    path: Path,
    xform: np.ndarray,   # 4×4 cumulative transform
    color: int,           # inherited colour
    invert: bool,         # winding inversion flag
    faces: list[Face],
    edges: list[Edge],
    collect_edges: bool = False,
    depth: int = 0,
) -> None:
    """Recursively load geometry from *path* into *faces* and *edges*."""
    if depth > 20:
        return  # safety guard against circular references

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        print(f"Warning: cannot read {path}: {exc}", file=sys.stderr)
        return

    pending_invert = False

    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue

        t = parts[0]

        # --- Meta / comment ---
        if t == "0":
            rest = " ".join(parts[1:]).upper()
            if "BFC INVERTNEXT" in rest:
                pending_invert = True
            # All other meta lines leave pending_invert unchanged
            continue

        # --- Sub-file reference ---
        if t == "1":
            if len(parts) < 15:
                pending_invert = False
                continue

            sub_color = int(parts[1])
            if sub_color == 16:
                sub_color = color

            pos = np.array([float(p) for p in parts[2:5]])
            rot = np.array([float(p) for p in parts[5:14]], dtype=float).reshape(3, 3)

            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = pos

            sub_name = " ".join(parts[14:])
            sub_path = find_part(sub_name)
            if sub_path is None:
                print(f"Warning: part not found: {sub_name}", file=sys.stderr)
                pending_invert = False
                continue

            # Winding flips when: INVERTNEXT is set OR the transform has negative determinant
            det = np.linalg.det(rot)
            child_invert = invert ^ pending_invert ^ (det < 0)

            load_file(sub_path, xform @ mat, sub_color, child_invert,
                      faces, edges, collect_edges, depth + 1)
            pending_invert = False

        # --- Edge line (type 2) ---
        elif t == "2":
            if collect_edges and len(parts) >= 8:
                edge_color = int(parts[1])
                if edge_color == 16:
                    edge_color = color
                elif edge_color == EDGE_COLOR_ID:
                    edge_color = -1  # use derived edge colour at render time

                coords = [float(p) for p in parts[2:8]]
                verts = np.array([coords[0:3], coords[3:6]], dtype=float)
                verts_h = np.hstack([verts, np.ones((2, 1))])
                verts_w = (xform @ verts_h.T).T[:, :3]

                edges.append(Edge(verts_w, edge_color))
            pending_invert = False

        # --- Triangle (type 3) or Quad (type 4) ---
        elif t in ("3", "4"):
            n = 3 if t == "3" else 4
            expected = 2 + n * 3
            if len(parts) < expected:
                pending_invert = False
                continue

            face_color = int(parts[1])
            if face_color == 16:
                face_color = color

            coords = [float(p) for p in parts[2:2 + n * 3]]
            verts = np.array(coords, dtype=float).reshape(n, 3)
            verts_h = np.hstack([verts, np.ones((n, 1))])
            verts_w = (xform @ verts_h.T).T[:, :3]

            if invert:
                verts_w = verts_w[::-1]

            faces.append(Face(verts_w, face_color))
            pending_invert = False

        else:
            # Types 5 (optional lines) — skip
            pending_invert = False


# ---------------------------------------------------------------------------
# Camera / projection
# ---------------------------------------------------------------------------
def make_view_matrix(h_deg: float = 45.0, v_deg: float = 30.0) -> np.ndarray:
    """
    Build a 3×3 rotation matrix for orthographic isometric-style projection.

    h_deg: horizontal rotation (azimuth around LDraw Y-axis)
    v_deg: vertical tilt (elevation above horizontal)

    LDraw coordinate system: X right, Y down, Z toward viewer.
    We negate Y so that the camera sees the model right-side-up.
    """
    h = np.radians(h_deg)
    v = np.radians(v_deg)

    # Rotate around Y (horizontal spin)
    Ry = np.array([
        [ np.cos(h), 0, np.sin(h)],
        [         0, 1,         0],
        [-np.sin(h), 0, np.cos(h)],
    ])

    # Tilt up: rotate around X (but negate because LDraw Y is down)
    Rx = np.array([
        [1,          0,           0],
        [0,  np.cos(v), np.sin(v)],   # flip sign so "up" goes up on screen
        [0, -np.sin(v), np.cos(v)],
    ])

    return Rx @ Ry


def project_pts(verts: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project (N,3) world coords to (N,2) screen coords."""
    rot = verts @ R.T
    return rot[:, :2]   # drop Z (orthographic)


def avg_depth(verts: np.ndarray, R: np.ndarray) -> float:
    """Painter's algorithm depth: mean Z in view space."""
    return float((verts @ R.T)[:, 2].mean())


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------
# Light direction in LDraw world space (right, up-in-world=-Y, front)
_LIGHT_DIR = np.array([1.0, -2.0, 1.0])
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)


def face_shade(verts: np.ndarray) -> float:
    """Diffuse shading factor [0.35 … 1.0] for a face."""
    if len(verts) < 3:
        return 0.7
    n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    nl = np.linalg.norm(n)
    if nl < 1e-12:
        return 0.7
    n /= nl
    return float(np.clip(np.dot(n, _LIGHT_DIR) * 0.5 + 0.65, 0.35, 1.0))


def screen_normal_z(verts: np.ndarray, R: np.ndarray) -> float:
    """Normalised Z of the face normal in view space.
    Returns a cosine in [-1, 1]; degenerate faces return 0.
    """
    rv = verts @ R.T
    if len(rv) < 3:
        return 1.0
    n = np.cross(rv[1] - rv[0], rv[2] - rv[0])
    nl = float(np.linalg.norm(n))
    if nl < 1e-12:
        return 0.0   # degenerate — will be culled
    return float(n[2]) / nl


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------
def render(
    faces: list[Face],
    edges: list[Edge],
    output: str,
    h_deg: float,
    v_deg: float,
    scale: float,
    padding: int,
    bg: str,
    edge_width: float,
    stroke_width: float,
    draw_edges: bool = False,
) -> None:
    R = make_view_matrix(h_deg, v_deg)

    # ---- Project and cull faces ----
    # Cull threshold > 0: strict back-face cull + removes degenerate/edge-on faces
    CULL_THRESHOLD = 0.01
    draw_faces: list[tuple[float, np.ndarray, int, float]] = []
    for face in faces:
        if screen_normal_z(face.verts, R) <= CULL_THRESHOLD:
            continue   # back-face / edge-on cull
        depth = avg_depth(face.verts, R)
        pts2d = project_pts(face.verts, R) * scale
        shade = face_shade(face.verts)
        draw_faces.append((depth, pts2d, face.color_id, shade))

    # ---- Project edges (only if requested) ----
    proj_edges: list[tuple[float, np.ndarray, int]] = []
    if draw_edges:
        for edge in edges:
            depth = avg_depth(edge.verts, R)
            pts2d = project_pts(edge.verts, R) * scale
            proj_edges.append((depth, pts2d, edge.color_id))

    if not draw_faces:
        print("No visible geometry.", file=sys.stderr)
        return

    # ---- Compute canvas size from faces (+ edges if any) ----
    all_pts = np.vstack(
        [p for _, p, *_ in draw_faces] +
        ([p for _, p, *_ in proj_edges] if proj_edges else [])
    )
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    canvas_w = int(max_xy[0] - min_xy[0] + 2 * padding)
    canvas_h = int(max_xy[1] - min_xy[1] + 2 * padding)

    def offset(pts: np.ndarray) -> list[tuple[float, float]]:
        shifted = pts - min_xy + padding
        return [(float(x), float(y)) for x, y in shifted]

    # ---- Sort faces back-to-front (painter's algorithm) ----
    draw_faces.sort(key=lambda x: x[0])
    proj_edges.sort(key=lambda x: x[0])

    # ---- Build SVG ----
    dwg = svgwrite.Drawing(output, size=(f"{canvas_w}px", f"{canvas_h}px"))
    dwg.add(dwg.rect((0, 0), ("100%", "100%"), fill=bg))

    g_faces = dwg.g(id="faces")
    for _, pts2d, color_id, shade in draw_faces:
        base = ldraw_hex(color_id)
        fill = shade_hex(base, shade)
        stroke = shade_hex(base, shade * 0.5)
        g_faces.add(dwg.polygon(
            offset(pts2d),
            fill=fill,
            stroke=stroke,
            stroke_width=stroke_width,
            stroke_linejoin="round",
        ))
    dwg.add(g_faces)

    if proj_edges:
        g_edges = dwg.g(id="edges")
        for _, pts2d, color_id in proj_edges:
            stroke = "#1a1a1a" if color_id < 0 else shade_hex(ldraw_hex(color_id), 0.6)
            pts = offset(pts2d)
            g_edges.add(dwg.line(
                start=pts[0], end=pts[1],
                stroke=stroke,
                stroke_width=edge_width,
                stroke_linecap="round",
            ))
        dwg.add(g_edges)

    dwg.save(pretty=True)
    print(f"Saved: {output}  ({canvas_w}×{canvas_h} px, {len(draw_faces)} faces)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render an LDraw (.ldr) scene to a clean SVG LEGO illustration."
    )
    parser.add_argument("input", help="Input .ldr file")
    parser.add_argument("-o", "--output", help="Output SVG file (default: <input>.svg)")
    parser.add_argument("--h-angle", type=float, default=45.0,
                        help="Horizontal camera angle in degrees (default 45)")
    parser.add_argument("--v-angle", type=float, default=30.0,
                        help="Vertical camera tilt in degrees (default 30)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor (default 1.0)")
    parser.add_argument("--padding", type=int, default=40,
                        help="Canvas padding in pixels (default 40)")
    parser.add_argument("--bg", default="#f8f8f0",
                        help="Background colour (default #f8f8f0)")
    parser.add_argument("--edges", action="store_true", default=False,
                        help="Draw LDraw type-2 edge lines (off by default)")
    parser.add_argument("--edge-width", type=float, default=0.8,
                        help="Width of edge lines when --edges is set (default 0.8)")
    parser.add_argument("--stroke-width", type=float, default=1.0,
                        help="Width of polygon outlines (default 1.0)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = args.output or str(input_path.with_suffix(".svg"))

    faces: list[Face] = []
    edges: list[Edge] = []

    print(f"Loading {input_path} …")
    load_file(input_path, np.eye(4), color=7, invert=False,
              faces=faces, edges=edges, collect_edges=args.edges)
    print(f"  {len(faces)} faces, {len(edges)} edges loaded")

    render(
        faces, edges, output_path,
        h_deg=args.h_angle,
        v_deg=args.v_angle,
        scale=args.scale,
        padding=args.padding,
        bg=args.bg,
        edge_width=args.edge_width,
        stroke_width=args.stroke_width,
        draw_edges=args.edges,
    )


if __name__ == "__main__":
    main()
