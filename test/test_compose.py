"""Unit tests for ldr2svg.compose."""

import numpy as np
import pytest
import svgwrite
from pathlib import Path
from PIL import Image

from ldr2svg.parts import parse_ldr, PART_MAP, Piece
from ldr2svg.projection import project_ldraw
from ldr2svg.compose import (
    _piece_label, _project_piece, _project_pieces,
    _canvas_bounds, _build_defs, _inject_def_comments,
)

LDR_PATH = Path(__file__).parent.parent / "test.ldr"


def _make_pngs():
    """Minimal pngs list from test.ldr with synthetic image sizes."""
    pieces = parse_ldr(LDR_PATH)
    return [(p, 100, 80, 50.0, 40.0) for p in pieces if p.part in PART_MAP]


def _make_renders():
    """Unique renders dict from test.ldr with synthetic images (3 entries)."""
    pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
    return {
        label: (Image.new("RGBA", (100, 80)), 50.0, 40.0,
                [p for p in pieces if _piece_label(p) == label])
        for label in dict.fromkeys(_piece_label(p) for p in pieces)
    }


def _projected_row(sx=0.0, sy=0.0, ax=0.0, ay=0.0, iw=100, ih=80, label="test"):
    return (sx, sy, ax, ay, iw, ih, label)


class TestProjectPiece:
    def _piece_at(self, x=0.0, y=0.0, z=0.0):
        return Piece(part="3666", color=4,
                     pos=np.array([x, y, z]), rot=np.eye(3))

    def test_tuple_length(self):
        row = _project_piece(self._piece_at(), 100, 80, 50.0, 40.0)
        assert len(row) == 9

    def test_image_size_fields(self):
        row = _project_piece(self._piece_at(), 120, 90, 0.0, 0.0)
        assert row[6] == 120
        assert row[7] == 90

    def test_ldy_is_ldraw_y(self):
        row = _project_piece(self._piece_at(y=-32.0), 100, 80, 0.0, 0.0)
        assert row[1] == pytest.approx(-32.0)

    def test_label_field(self):
        piece = self._piece_at()
        row = _project_piece(piece, 100, 80, 0.0, 0.0)
        assert row[8] == _piece_label(piece)

    def test_anchor_passthrough(self):
        row = _project_piece(self._piece_at(), 100, 80, 12.5, 7.3)
        assert row[4] == pytest.approx(12.5)
        assert row[5] == pytest.approx(7.3)


class TestProjectPieces:
    def test_returns_correct_count(self):
        assert len(_project_pieces(_make_pngs())) == 5

    def test_tuple_has_seven_fields(self):
        assert all(len(row) == 7 for row in _project_pieces(_make_pngs()))

    def test_elevated_pieces_last(self):
        """Pieces with ldY=-32 (elevated) must be last in the sorted output."""
        result = _project_pieces(_make_pngs())
        sy_pxs = [row[1] for row in result]
        assert sy_pxs[-1] < sy_pxs[0]


class TestCanvasBounds:
    def test_single_piece_size(self):
        W, H, _, _ = _canvas_bounds([_projected_row()], padding=10)
        assert W == 120   # 100 + 2×10
        assert H == 100   # 80  + 2×10

    def test_no_padding(self):
        W, H, _, _ = _canvas_bounds([_projected_row(iw=50, ih=30)], padding=0)
        assert W == 50
        assert H == 30

    def test_min_xy_offset(self):
        _, _, min_x, min_y = _canvas_bounds(
            [_projected_row(sx=200, sy=100)], padding=0
        )
        assert min_x == pytest.approx(200.0)
        assert min_y == pytest.approx(100.0)


class TestBuildDefs:
    def test_one_def_per_unique_render(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, _make_renders())
        assert len(defs) == 3

    def test_def_id_format(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, _make_renders())
        for label, (def_id, _, _) in defs.items():
            part = label.split()[0]
            assert def_id.startswith(f"{part}-")
            suffix = def_id[len(part) + 1:]
            assert len(suffix) == 8
            assert all(c in "0123456789abcdef" for c in suffix)

    def test_anchors_stored(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        defs = _build_defs(dwg, _make_renders())
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
