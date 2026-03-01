# ldr2svg

Render LDraw/LeoCAD (`.ldr`) scenes to clean isometric SVG brick illustrations.

[![CI](https://github.com/lbruand/bricksvg/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand/bricksvg/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE.md)
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

## How to import your svg into google slides

 1. Load the svg into a LibreOffice presentation ( tested on version : Version: 24.2.7.2 (X86_64) / LibreOffice Community
Build ID: 420(Build:2))
 2. Click on the image. Contextual menu : Break the SVG
 3. Save the image in format pptx ( Microsoft PowerPoint 2007+)
 4. Upload the pptx into google slides 

## Credits and licences

- **`ldr2svg/brick.scad`** — derived from [Thingiverse thing:5699](http://www.thingiverse.com/thing:5699),
  © 2015 Christopher Finke, MIT licence.
  LEGO, the LEGO logo, the Brick, DUPLO, and MINDSTORMS are trademarks of the LEGO Group.

- **LDraw** — part geometry and colour definitions follow the
  [LDraw](https://www.ldraw.org/) open standard for LEGO CAD programs.
  LDraw™ is a trademark of the Estate of James Jessiman.
