# ldr2svg

Render LDraw/LeoCAD (`.ldr`) scenes to clean isometric SVG LEGO illustrations.

[![CI](https://github.com/lbruand/bricksvg/actions/workflows/ci.yml/badge.svg)](https://github.com/lbruand/bricksvg/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type--check-ty-261230)](https://github.com/astral-sh/ty)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

![Example output](docs/img/test_lego.svg)

## Usage

```bash
uv run ldr2svg scene.ldr
```

Outputs `scene.svg` alongside the input file.

## Requirements

- Python 3.10+
- [OpenSCAD](https://openscad.org/) (for rendering piece images)
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv sync
```

## Development

```bash
uv run pytest test/ -m "not slow"   # fast unit tests
uv run ruff check ldr2svg/ test/    # lint
uv run ty check ldr2svg/            # type check
```

Slow integration tests (require OpenSCAD):

```bash
uv run pytest test/ -v
```
