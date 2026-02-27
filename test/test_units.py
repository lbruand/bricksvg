"""Fast unit tests — no OpenSCAD required."""

import numpy as np
import pytest
import svgwrite
from pathlib import Path
from PIL import Image

from ldr2svg.ldr2png_svg import (
    parse_ldr, project_ldraw, ldraw_rgb, PART_MAP,
    _project_pieces, _canvas_bounds, _build_defs, _inject_def_comments,
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


def _make_pngs():
    """Minimal pngs list from test.ldr with synthetic images."""
    pieces = parse_ldr(LDR_PATH)
    return [
        (p, Image.new("RGBA", (100, 80)), 50.0, 40.0)
        for p in pieces if p.part in PART_MAP
    ]


def _projected_row(sx=0.0, sy=0.0, ax=0.0, ay=0.0, iw=100, ih=80, label="test"):
    """Minimal projected tuple for _canvas_bounds testing."""
    return (0.0, 0.0, sx, sy, None, ax, ay, iw, ih, label)


class TestProjectPieces:
    def test_returns_correct_count(self):
        assert len(_project_pieces(_make_pngs())) == 5

    def test_tuple_has_ten_fields(self):
        assert all(len(row) == 10 for row in _project_pieces(_make_pngs()))

    def test_elevated_pieces_last(self):
        """Pieces with ldY=-32 (elevated) must be last in the sorted output."""
        result = _project_pieces(_make_pngs())
        ldys = [row[1] for row in result]
        assert ldys[-1] < ldys[0]   # last (on top) is more negative = higher in scene


class TestCanvasBounds:
    def test_single_piece_size(self):
        row = _projected_row(sx=0, sy=0, ax=0, ay=0, iw=100, ih=80)
        W, H, _, _ = _canvas_bounds([row], padding=10)
        assert W == 120   # 100 + 2×10
        assert H == 100   # 80  + 2×10

    def test_no_padding(self):
        row = _projected_row(sx=0, sy=0, ax=0, ay=0, iw=50, ih=30)
        W, H, _, _ = _canvas_bounds([row], padding=0)
        assert W == 50
        assert H == 30

    def test_min_xy_offset(self):
        row = _projected_row(sx=200, sy=100, ax=0, ay=0, iw=50, ih=30)
        _, _, min_x, min_y = _canvas_bounds([row], padding=0)
        assert min_x == pytest.approx(200.0)
        assert min_y == pytest.approx(100.0)


class TestBuildDefs:
    def test_deduplicates_identical_labels(self, tmp_path):
        """Three identical 3062a pieces → only one entry in defs."""
        projected = _project_pieces(_make_pngs())
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, projected)
        assert len(defs) == 3   # 3666, 60474, 3062a

    def test_def_id_format(self, tmp_path):
        """def_id must be '{part}-{8 hex chars}'."""
        projected = _project_pieces(_make_pngs())
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, projected)
        for label, (def_id, _, _) in defs.items():
            part = label.split()[0]
            assert def_id.startswith(f"{part}-")
            suffix = def_id[len(part) + 1:]
            assert len(suffix) == 8
            assert all(c in "0123456789abcdef" for c in suffix)

    def test_anchors_stored(self, tmp_path):
        projected = _project_pieces(_make_pngs())
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, projected)
        for _, (_, ax, ay) in defs.items():
            assert isinstance(ax, float)
            assert isinstance(ay, float)


class TestInjectDefComments:
    def test_comments_inserted(self, tmp_path):
        svg = tmp_path / "t.svg"
        svg.write_text('<svg><defs><image id="a"/><image id="b"/></defs></svg>')
        _inject_def_comments(str(svg), {"label-a": ("a", 0.0, 0.0), "label-b": ("b", 0.0, 0.0)})
        result = svg.read_text()
        assert "<!-- label-a -->" in result
        assert "<!-- label-b -->" in result

    def test_comments_in_order(self, tmp_path):
        svg = tmp_path / "t.svg"
        svg.write_text('<svg><image id="x"/><image id="y"/></svg>')
        _inject_def_comments(str(svg), {"first": ("x", 0.0, 0.0), "second": ("y", 0.0, 0.0)})
        result = svg.read_text()
        assert result.index("<!-- first -->") < result.index("<!-- second -->")


class TestZOrderSort:
    def test_elevated_pieces_painted_last(self):
        """3062a bricks (ldY=-32) must sort after the plates (ldY=-8)."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        ordered = sorted(pieces, key=lambda p: (-p.pos[1], project_ldraw(p.pos)[2]))
        names = [p.part for p in ordered]
        assert set(names[:2]) == {"3666", "60474"}
        assert names[2:] == ["3062a", "3062a", "3062a"]