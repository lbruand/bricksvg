"""Fast unit tests — no OpenSCAD required."""

import numpy as np
import pytest
from pathlib import Path

from ldr2svg.ldr2png_svg import (
    parse_ldr, project_ldraw, ldraw_rgb, PART_MAP,
)

LDR_PATH = Path(__file__).parent.parent / "test.ldr"


class TestParseLdr:
    def test_piece_count(self):
        assert len(parse_ldr(LDR_PATH)) == 5

    def test_part_names(self):
        parts = [p.part for p in parse_ldr(LDR_PATH)]
        assert parts == ["3666", "60474", "3062a", "3062a", "3062a"]

    def test_first_piece_position(self):
        np.testing.assert_array_equal(parse_ldr(LDR_PATH)[0].pos, [50, -8, 60])

    def test_identity_rotation(self):
        # 60474 (index 1) has identity rotation matrix
        np.testing.assert_array_almost_equal(parse_ldr(LDR_PATH)[1].rot, np.eye(3))

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.ldr"
        f.write_text("")
        assert parse_ldr(f) == []

    def test_skips_non_type1_lines(self, tmp_path):
        f = tmp_path / "t.ldr"
        f.write_text("0 comment\n2 7 0 0 0 10 0 0\n")
        assert parse_ldr(f) == []


class TestLdrawRgb:
    def test_white(self):
        assert ldraw_rgb(15) == (255, 255, 255)

    def test_grey(self):
        assert ldraw_rgb(7) == (155, 155, 155)

    def test_unknown_falls_back_to_grey(self):
        assert ldraw_rgb(9999) == (136, 136, 136)


class TestProjectLdraw:
    def test_origin_maps_to_zero(self):
        sx, sy, depth = project_ldraw(np.zeros(3))
        assert sx == pytest.approx(0.0)
        assert sy == pytest.approx(0.0)
        assert depth == pytest.approx(0.0)

    def test_x_symmetry(self):
        # Negating LDraw X should negate screen_x
        sx1, _, _ = project_ldraw(np.array([10.0, 0.0, 0.0]))
        sx2, _, _ = project_ldraw(np.array([-10.0, 0.0, 0.0]))
        assert sx1 == pytest.approx(-sx2)

    def test_elevation_raises_on_screen(self):
        # Higher in scene (more negative LDraw Y) → smaller screen_y (higher on screen)
        _, sy_low, _ = project_ldraw(np.array([0.0, -8.0, 0.0]))
        _, sy_high, _ = project_ldraw(np.array([0.0, -32.0, 0.0]))
        assert sy_high < sy_low


class TestPartMap:
    def test_all_dimensions_positive(self):
        for name, part in PART_MAP.items():
            assert part.w_mm > 0, f"{name}: w_mm <= 0"
            assert part.l_mm > 0, f"{name}: l_mm <= 0"
            assert part.h_mm > 0, f"{name}: h_mm <= 0"

    def test_plate_thinner_than_brick(self):
        assert PART_MAP["3666"].h_mm < PART_MAP["3062a"].h_mm


class TestZOrderSort:
    def test_elevated_pieces_painted_last(self):
        """3062a bricks (ldY=-32) must sort after the plates (ldY=-8)."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        ordered = sorted(pieces, key=lambda p: (-p.pos[1], project_ldraw(p.pos)[2]))
        names = [p.part for p in ordered]
        assert set(names[:2]) == {"3666", "60474"}
        assert names[2:] == ["3062a", "3062a", "3062a"]