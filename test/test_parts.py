"""Unit tests for ldr2svg.parts."""

import numpy as np
from pathlib import Path

from ldr2svg.parts import parse_ldr, _parse_ldr_line, ldraw_rgb, PART_MAP

LDR_PATH = Path(__file__).parent.parent / "test.ldr"

_SAMPLE_LINE = "1 4 50 -8 60 1 0 0 0 1 0 0 0 1 3666.dat".split()


class TestParseLdr:
    def test_piece_count(self):
        assert len(parse_ldr(LDR_PATH)) == 5

    def test_part_names(self):
        parts = [p.part for p in parse_ldr(LDR_PATH)]
        assert parts == ["3666", "60474", "3062a", "3062a", "3062a"]

    def test_first_piece_position(self):
        np.testing.assert_array_equal(parse_ldr(LDR_PATH)[0].pos, [50, -8, 60])

    def test_identity_rotation(self):
        np.testing.assert_array_almost_equal(parse_ldr(LDR_PATH)[1].rot, np.eye(3))

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.ldr"
        f.write_text("")
        assert parse_ldr(f) == []

    def test_skips_non_type1_lines(self, tmp_path):
        f = tmp_path / "t.ldr"
        f.write_text("0 comment\n2 7 0 0 0 10 0 0\n")
        assert parse_ldr(f) == []


class TestParseLdrLine:
    def test_part_name(self):
        assert _parse_ldr_line(_SAMPLE_LINE).part == "3666"

    def test_color(self):
        assert _parse_ldr_line(_SAMPLE_LINE).color == 4

    def test_position(self):
        np.testing.assert_array_equal(_parse_ldr_line(_SAMPLE_LINE).pos, [50, -8, 60])

    def test_rotation_identity(self):
        np.testing.assert_array_almost_equal(_parse_ldr_line(_SAMPLE_LINE).rot, np.eye(3))

    def test_part_with_spaces_in_filename(self):
        parts = "1 15 0 0 0 1 0 0 0 1 0 0 0 1 parts/s/some part.dat".split()
        assert _parse_ldr_line(parts).part == "some part"

    def test_part_stem_lowercased(self):
        parts = "1 1 0 0 0 1 0 0 0 1 0 0 0 1 3062A.DAT".split()
        assert _parse_ldr_line(parts).part == "3062a"


class TestLdrawRgb:
    def test_white(self):
        assert ldraw_rgb(15) == (255, 255, 255)

    def test_grey(self):
        assert ldraw_rgb(7) == (155, 155, 155)

    def test_unknown_falls_back_to_grey(self):
        assert ldraw_rgb(9999) == (136, 136, 136)


class TestPartMap:
    def test_all_dimensions_positive(self):
        for name, part in PART_MAP.items():
            assert part.w_mm > 0, f"{name}: w_mm <= 0"
            assert part.l_mm > 0, f"{name}: l_mm <= 0"
            assert part.h_mm > 0, f"{name}: h_mm <= 0"

    def test_plate_thinner_than_brick(self):
        assert PART_MAP["3666"].h_mm < PART_MAP["3062a"].h_mm
