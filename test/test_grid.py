"""Unit tests for ldr2svg.grid."""

import numpy as np
import pytest
import svgwrite
from PIL import Image

from ldr2svg.parts import Piece
from ldr2svg.grid import (
    _floor_y_ldu,
    _grid_params,
    _grid_corner_sx_sy,
    _draw_isometric_grid,
)


def _make_renders(pos_y: float = -24.0) -> dict:
    """One-piece renders dict suitable for grid testing.

    Uses part "3003" (2×2 brick, h_mm=9.6, so h_mm/LDU_TO_MM=24 LDU).
    With pos_y=-24 the brick sits on the floor (bottom at Y=0).
    """
    piece = Piece(part="3003", color=15, pos=np.array([0.0, pos_y, 0.0]), rot=np.eye(3))
    return {"3003 color=15 rot=...": (Image.new("RGBA", (100, 80)), 50.0, 40.0, [piece])}


# ---------------------------------------------------------------------------
# _floor_y_ldu
# ---------------------------------------------------------------------------

class TestFloorYLdu:
    def test_piece_on_floor(self):
        # top at Y=-24, height=24 LDU → bottom = 0
        assert _floor_y_ldu(_make_renders(-24.0)) == pytest.approx(0.0)

    def test_elevated_piece(self):
        # top at Y=-32, height=24 LDU → bottom = -8
        assert _floor_y_ldu(_make_renders(-32.0)) == pytest.approx(-8.0)

    def test_empty_renders_default_zero(self):
        assert _floor_y_ldu({}) == pytest.approx(0.0)

    def test_multiple_pieces_returns_max(self):
        piece_low  = Piece(part="3003", color=1, pos=np.array([0.0, -24.0, 0.0]), rot=np.eye(3))
        piece_high = Piece(part="3003", color=2, pos=np.array([40.0, -32.0, 0.0]), rot=np.eye(3))
        renders = {
            "a": (Image.new("RGBA", (10, 10)), 0.0, 0.0, [piece_low]),
            "b": (Image.new("RGBA", (10, 10)), 0.0, 0.0, [piece_high]),
        }
        # max(0.0, -8.0) = 0.0
        assert _floor_y_ldu(renders) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _grid_params
# ---------------------------------------------------------------------------

class TestGridParams:
    def test_empty_renders_returns_none(self):
        assert _grid_params({}) is None

    def test_returns_five_tuple(self):
        result = _grid_params(_make_renders())
        assert result is not None
        assert len(result) == 5

    def test_bounds_include_piece_x_z(self):
        fy, gx0, gx1, gz0, gz1 = _grid_params(_make_renders())
        assert gx0 <= 0 <= gx1
        assert gz0 <= 0 <= gz1

    def test_margin_extends_beyond_piece(self):
        # GRID_MARGIN=2 × GRID_STEP=20 → at least 40 LDU margin each side
        fy, gx0, gx1, gz0, gz1 = _grid_params(_make_renders())
        assert gx0 < 0
        assert gx1 > 0

    def test_floor_y_matches_piece_bottom(self):
        fy, *_ = _grid_params(_make_renders(-24.0))
        assert fy == pytest.approx(0.0)

    def test_snapped_to_grid_step(self):
        from ldr2svg.grid import GRID_STEP
        fy, gx0, gx1, gz0, gz1 = _grid_params(_make_renders())
        assert gx0 % GRID_STEP == pytest.approx(0, abs=1e-9)
        assert gx1 % GRID_STEP == pytest.approx(0, abs=1e-9)


# ---------------------------------------------------------------------------
# _grid_corner_sx_sy
# ---------------------------------------------------------------------------

class TestGridCornerSxSy:
    def test_returns_four_corners(self):
        result = _grid_corner_sx_sy(0.0, -40.0, 40.0, -40.0, 40.0)
        assert len(result) == 4

    def test_each_corner_is_float_pair(self):
        for sx, sy in _grid_corner_sx_sy(0.0, -40.0, 40.0, -40.0, 40.0):
            assert isinstance(sx, float)
            assert isinstance(sy, float)

    def test_corners_differ(self):
        corners = _grid_corner_sx_sy(0.0, -40.0, 40.0, -40.0, 40.0)
        # All four corners should be distinct screen positions
        assert len({c for c in corners}) == 4


# ---------------------------------------------------------------------------
# _draw_isometric_grid
# ---------------------------------------------------------------------------

class TestDrawIsometricGrid:
    def _draw(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        _draw_isometric_grid(dwg, 0.0, -40.0, 40.0, -40.0, 40.0,
                             lambda v: v, lambda v: v)
        return dwg

    def test_adds_grid_group(self, tmp_path):
        dwg = self._draw(tmp_path)
        ids = [el.attribs.get("id") for el in dwg.elements]
        assert "grid" in ids

    def test_grid_group_contains_lines(self, tmp_path):
        dwg = self._draw(tmp_path)
        grp = next(el for el in dwg.elements if el.attribs.get("id") == "grid")
        assert len(grp.elements) > 0

    def test_line_count_matches_grid_size(self, tmp_path):
        from ldr2svg.grid import GRID_STEP
        dwg = self._draw(tmp_path)
        grp = next(el for el in dwg.elements if el.attribs.get("id") == "grid")
        # range(-40, 41, 20) = 5 Z-lines + 5 X-lines = 10
        expected = len(range(-40, 41, GRID_STEP)) * 2
        assert len(grp.elements) == expected
