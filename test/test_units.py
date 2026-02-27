"""Fast unit tests — no OpenSCAD required."""

import numpy as np
import pytest
import svgwrite
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from ldr2svg.ldr2png_svg import (
    parse_ldr, _parse_ldr_line, project_ldraw, ldraw_rgb, PART_MAP,
    _project_piece, _project_pieces, _canvas_bounds, _build_defs, _inject_def_comments,
    _piece_label, build_pngs,
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


_SAMPLE_LINE = "1 4 50 -8 60 1 0 0 0 1 0 0 0 1 3666.dat".split()

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


def _make_renders():
    """Unique renders dict from test.ldr with synthetic images (3 entries)."""
    renders: dict = {}
    for p in parse_ldr(LDR_PATH):
        if p.part in PART_MAP:
            label = _piece_label(p)
            if label not in renders:
                renders[label] = (Image.new("RGBA", (100, 80)), 50.0, 40.0)
    return renders


def _projected_row(sx=0.0, sy=0.0, ax=0.0, ay=0.0, iw=100, ih=80, label="test"):
    """Minimal projected tuple for _canvas_bounds testing."""
    return (sx, sy, ax, ay, iw, ih, label)


class TestProjectPiece:
    def _piece_at(self, x=0.0, y=0.0, z=0.0):
        from ldr2svg.ldr2png_svg import Piece
        return Piece(part="3666", color=4,
                     pos=np.array([x, y, z]), rot=np.eye(3))

    def test_tuple_length(self):
        piece = self._piece_at()
        img = Image.new("RGBA", (100, 80))
        row = _project_piece(piece, img, 50.0, 40.0)
        assert len(row) == 10

    def test_image_size_fields(self):
        piece = self._piece_at()
        img = Image.new("RGBA", (120, 90))
        row = _project_piece(piece, img, 0.0, 0.0)
        iw, ih = row[7], row[8]
        assert iw == 120
        assert ih == 90

    def test_ldy_is_ldraw_y(self):
        piece = self._piece_at(y=-32.0)
        img = Image.new("RGBA", (100, 80))
        row = _project_piece(piece, img, 0.0, 0.0)
        assert row[1] == pytest.approx(-32.0)

    def test_label_field(self):
        piece = self._piece_at()
        img = Image.new("RGBA", (100, 80))
        row = _project_piece(piece, img, 0.0, 0.0)
        assert row[9] == _piece_label(piece)

    def test_anchor_passthrough(self):
        piece = self._piece_at()
        img = Image.new("RGBA", (100, 80))
        row = _project_piece(piece, img, 12.5, 7.3)
        assert row[5] == pytest.approx(12.5)
        assert row[6] == pytest.approx(7.3)


class TestProjectPieces:
    def test_returns_correct_count(self):
        assert len(_project_pieces(_make_pngs())) == 5

    def test_tuple_has_seven_fields(self):
        assert all(len(row) == 7 for row in _project_pieces(_make_pngs()))

    def test_elevated_pieces_last(self):
        """Pieces with ldY=-32 (elevated) must be last in the sorted output."""
        result = _project_pieces(_make_pngs())
        sy_pxs = [row[1] for row in result]
        assert sy_pxs[-1] < sy_pxs[0]   # last (on top) has smaller screen_y = higher on screen


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
    def test_one_def_per_unique_render(self, tmp_path):
        """Three unique renders (3666, 60474, 3062a) → three defs entries."""
        renders = _make_renders()
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, renders)
        assert len(defs) == 3   # 3666, 60474, 3062a

    def test_def_id_format(self, tmp_path):
        """def_id must be '{part}-{8 hex chars}'."""
        renders = _make_renders()
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, renders)
        for label, (def_id, _, _) in defs.items():
            part = label.split()[0]
            assert def_id.startswith(f"{part}-")
            suffix = def_id[len(part) + 1:]
            assert len(suffix) == 8
            assert all(c in "0123456789abcdef" for c in suffix)

    def test_anchors_stored(self, tmp_path):
        renders = _make_renders()
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, renders)
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


_FAKE_IMG = Image.new("RGBA", (100, 80))
_FAKE_RENDER = (True, (_FAKE_IMG, 50.0, 40.0))   # render_piece → True, remove_and_crop → tuple


def _fake_remove_and_crop(_path):
    return _FAKE_IMG, 50.0, 40.0


class TestBuildPngs:
    def _run(self, pieces, tmp_path, keep_pngs=False):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            pngs, _renders = build_pngs(pieces, tmp_path, keep_pngs=keep_pngs)
            return pngs

    def test_known_parts_returned(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        result = self._run(pieces, tmp_path)
        assert len(result) == len(pieces)

    def test_unknown_part_skipped(self, tmp_path):
        pieces = parse_ldr(LDR_PATH)
        # All parts in test.ldr are known, so inject one unknown
        from ldr2svg.ldr2png_svg import Piece
        unknown = Piece(part="unknown_part_xyz", color=1,
                        pos=np.zeros(3), rot=np.eye(3))
        result = self._run([unknown], tmp_path)
        assert result == []

    def test_identical_pieces_cached(self, tmp_path):
        """Two identical pieces should only call render_piece once."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part == "3062a"]
        assert len(pieces) == 3
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True) as mock_render, \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            pngs, renders = build_pngs(pieces, tmp_path)
        assert len(pngs) == 3
        mock_render.assert_called_once()   # cached after first render

    def test_renders_has_unique_entries(self, tmp_path):
        """5-piece scene with 3 unique (part, color, rot) combos → renders has 3 entries."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            pngs, renders = build_pngs(pieces, tmp_path)
        assert len(pngs) == 5
        assert len(renders) == 3   # 3666, 60474, 3062a (3 repeated pieces share one entry)

    def test_failed_render_skipped(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=False), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            pngs, renders = build_pngs(pieces, tmp_path)
        assert pngs == []

    def test_keep_pngs_false_removes_files(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            build_pngs(pieces, tmp_path, keep_pngs=False)
        assert list(tmp_path.glob("*.png")) == []

    def test_keep_pngs_true_retains_files(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            # render_piece writes nothing, so create the file manually to test keep logic
            def fake_render(scad_src, png_path):
                png_path.write_bytes(b"")
                return True
            with patch("ldr2svg.ldr2png_svg.render_piece", side_effect=fake_render):
                build_pngs(pieces, tmp_path, keep_pngs=True)
        assert len(list(tmp_path.glob("*.png"))) == 1


class TestZOrderSort:
    def test_elevated_pieces_painted_last(self):
        """3062a bricks (ldY=-32) must sort after the plates (ldY=-8)."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        ordered = sorted(pieces, key=lambda p: (-p.pos[1], project_ldraw(p.pos)[2]))
        names = [p.part for p in ordered]
        assert set(names[:2]) == {"3666", "60474"}
        assert names[2:] == ["3062a", "3062a", "3062a"]