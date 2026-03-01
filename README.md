# ldr2svg

Render LDraw/LeoCAD (`.ldr`) scenes to clean isometric SVG brick illustrations.

[![CI](https://github.com/lbruand/bricksvg/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand/bricksvg/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type--check-ty-261230)](https://github.com/astral-sh/ty)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

![Example output](docs/img/test_bricks.svg)

## Usage

```bash
uv run ldr2svg scene.ldr
```

Outputs `scene.svg` alongside the input file.

## Requirements

### System dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| [OpenSCAD](https://openscad.org/) | Render brick piece images | `sudo apt install openscad` |
| [Graphviz](https://graphviz.org/) | Lay out diagrams-library graphs (`dot`) | `sudo apt install graphviz` |
| [Xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) | Virtual display for headless OpenSCAD | `sudo apt install xvfb` |

On macOS: `brew install openscad graphviz` (Xvfb not needed).

### Python

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv sync
```

## Development

```bash
uv run pytest test/ -m "not slow"   # fast unit tests (no OpenSCAD needed)
uv run ruff check ldr2svg/ test/    # lint
uv run ty check ldr2svg/            # type check
```

Slow integration tests (require OpenSCAD; use `xvfb-run` on headless systems):

```bash
xvfb-run uv run pytest test/ -v
```
