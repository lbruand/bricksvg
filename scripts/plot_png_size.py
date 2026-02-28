#!/usr/bin/env python3
"""Plot compressed PNG size vs image resolution for a single 3062a brick."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ldr2svg.parts import PART_MAP
from ldr2svg.scad import make_scad
from ldr2svg.projection import CAMERA_RX, CAMERA_RZ, CAMERA_D

RESOLUTIONS = [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000]


def render_at_size(scad_src: str, imgsize: int, out_path: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".scad", mode="w", delete=False) as f:
        f.write(scad_src)
        scad_path = Path(f.name)
    try:
        camera = f"0,0,0,{CAMERA_RX},0,{CAMERA_RZ},{CAMERA_D}"
        result = subprocess.run(
            ["openscad",
             "--camera", camera,
             "--projection", "ortho",
             "--imgsize", f"{imgsize},{imgsize}",
             "-o", str(out_path),
             str(scad_path)],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0 and out_path.exists()
    finally:
        scad_path.unlink(missing_ok=True)


def _poly_label(coeffs: np.ndarray) -> str:
    deg = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = deg - i
        if power == 0:
            terms.append(f"{c:+.3f}")
        elif power == 1:
            terms.append(f"{c:+.2e}·x")
        else:
            terms.append(f"{c:+.2e}·x²")
    return "fit: " + " ".join(terms)


def main() -> None:
    scad_src = make_scad(PART_MAP["3062a"], (201, 26, 9), np.eye(3))

    sizes: list[int] = []
    file_sizes: list[int] = []

    with tempfile.TemporaryDirectory(prefix="ldr_density_") as tmpdir:
        for res in RESOLUTIONS:
            out_path = Path(tmpdir) / f"brick_{res}.png"
            print(f"  Rendering {res}×{res} …", end=" ", flush=True)
            ok = render_at_size(scad_src, res, out_path)
            if ok:
                sz = out_path.stat().st_size
                sizes.append(res)
                file_sizes.append(sz)
                print(f"{sz:,} bytes")
            else:
                print("FAILED")

    xs = np.array(sizes)
    ys = np.array(file_sizes) / 1024

    deg = 2
    coeffs = np.polyfit(xs, ys, deg)
    poly = np.poly1d(coeffs)
    xs_fit = np.linspace(xs[0], xs[-1], 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, s=60, zorder=3, label="measured")
    ax.plot(xs_fit, poly(xs_fit), linewidth=1.5, label=_poly_label(coeffs))
    ax.set_xlabel("Image resolution (px per side)")
    ax.set_ylabel("Compressed PNG size (KiB)")
    ax.set_title("3062a brick · compressed PNG size vs render resolution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_png = "png_size_vs_resolution.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()