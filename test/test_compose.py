"""Unit tests for ldr2svg.compose."""

import numpy as np
import pytest
import svgwrite
from pathlib import Path
from PIL import Image

from ldr2svg.parts import parse_ldr, PART_MAP, Piece
from ldr2svg.projection import project_ldraw
from ldr2svg.compose import (
    _piece_label_no_color,
    _fmt_rot_rows, _hash_label, _img_to_data_uri,
    _project_piece, _project_pieces,
    _canvas_bounds, _build_defs, _build_duotone_filters,
    compose_svg,
)

LDR_PATH = Path(__file__).parent.parent / "test.ldr"


def _make_white_renders():
    """White renders: keyed by _piece_label_no_color (as build_pngs_white produces)."""
    pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
    return {
        label: (Image.new("RGBA", (100, 80)), 50.0, 40.0,
                [p for p in pieces if _piece_label_no_color(p) == label])
        for label in dict.fromkeys(_piece_label_no_color(p) for p in pieces)
    }


def _make_pngs():
    """Minimal pngs list from test.ldr with synthetic image sizes."""
    pieces = parse_ldr(LDR_PATH)
    return [(p, 100, 80, 50.0, 40.0) for p in pieces if p.part in PART_MAP]


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
        assert row[8] == _piece_label_no_color(piece)

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
    def _make_renders(self):
        pieces = [
            Piece(part="3003", color=1, pos=np.zeros(3), rot=np.eye(3)),
            Piece(part="3666", color=4, pos=np.zeros(3), rot=np.eye(3)),
        ]
        return {
            _piece_label_no_color(p): (Image.new("RGBA", (100, 80)), 50.0, 40.0, [p])
            for p in pieces
        }

    def test_one_entry_per_label(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        renders = self._make_renders()
        result = _build_defs(dwg, renders)
        assert len(result) == len(renders)

    def test_img_id_prefix(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        result = _build_defs(dwg, self._make_renders())
        for img_id, *_ in result.values():
            assert img_id.startswith("grayscale-")

    def test_tuple_length(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        result = _build_defs(dwg, self._make_renders())
        for entry in result.values():
            assert len(entry) == 5  # img_id, ax, ay, iw, ih

    def test_image_size_stored(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        result = _build_defs(dwg, self._make_renders())
        for _, _, _, iw, ih in result.values():
            assert iw == 100
            assert ih == 80


class TestBuildDuotoneFilters:
    def test_one_filter_per_color(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        colors = {"#ff0000", "#00ff00", "#0000ff"}
        result = _build_duotone_filters(dwg, colors)
        assert len(result) == 3

    def test_filter_id_format(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        result = _build_duotone_filters(dwg, {"#aabbcc"})
        assert result["#aabbcc"] == "duotone-aabbcc"

    def test_same_color_same_id(self, tmp_path):
        dwg = svgwrite.Drawing(str(tmp_path / "t.svg"))
        result = _build_duotone_filters(dwg, {"#112233", "#112233"})
        assert len(result) == 1


class TestComposeSvg:
    def test_creates_svg_file(self, tmp_path):
        out = str(tmp_path / "out.svg")
        compose_svg(_make_white_renders(), out)
        assert Path(out).exists()

    def test_svg_contains_use_elements(self, tmp_path):
        out = str(tmp_path / "out.svg")
        compose_svg(_make_white_renders(), out)
        assert "<use" in Path(out).read_text()

    def test_svg_has_defs(self, tmp_path):
        out = str(tmp_path / "out.svg")
        compose_svg(_make_white_renders(), out)
        assert "<defs>" in Path(out).read_text()

    def test_svg_has_filter_elements(self, tmp_path):
        out = str(tmp_path / "out.svg")
        compose_svg(_make_white_renders(), out)
        assert "feColorMatrix" in Path(out).read_text()

    def test_svg_has_duotone_filters(self, tmp_path):
        out = str(tmp_path / "out.svg")
        compose_svg(_make_white_renders(), out)
        assert 'id="duotone-' in Path(out).read_text()

    def test_svg_dimensions_positive(self, tmp_path):
        import xml.etree.ElementTree as ET
        out = str(tmp_path / "out.svg")
        compose_svg(_make_white_renders(), out)
        root = ET.parse(out).getroot()
        w = int(root.attrib["width"].replace("px", ""))
        h = int(root.attrib["height"].replace("px", ""))
        assert w > 0 and h > 0


class TestFmtRotRows:
    def test_identity_format(self):
        piece = Piece(part="3003", color=1, pos=np.zeros(3), rot=np.eye(3))
        assert _fmt_rot_rows(piece) == "[[1,0,0],[0,1,0],[0,0,1]]"

    def test_non_integer_formatted_to_three_decimals(self):
        piece = Piece(part="3003", color=1, pos=np.zeros(3),
                      rot=np.array([[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        result = _fmt_rot_rows(piece)
        assert "0.500" in result

    def test_integer_values_not_formatted_as_float(self):
        piece = Piece(part="3003", color=1, pos=np.zeros(3), rot=np.eye(3))
        assert "." not in _fmt_rot_rows(piece)


class TestHashLabel:
    def test_returns_eight_char_hex(self):
        h = _hash_label("3003 color=1 rot=[[1,0,0]]")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_output(self):
        assert _hash_label("abc") == _hash_label("abc")

    def test_different_inputs_different_outputs(self):
        assert _hash_label("abc") != _hash_label("xyz")


class TestImgToDataUri:
    def test_starts_with_data_uri_prefix(self):
        img = Image.new("RGBA", (10, 10))
        uri = _img_to_data_uri(img)
        assert uri.startswith("data:image/png;base64,")

    def test_decodable_back_to_image(self):
        import base64
        import io
        img = Image.new("RGBA", (10, 10), (128, 64, 32, 255))
        uri = _img_to_data_uri(img)
        b64 = uri[len("data:image/png;base64,"):]
        decoded = Image.open(io.BytesIO(base64.b64decode(b64)))
        assert decoded.size == (10, 10)

    def test_different_images_produce_different_uris(self):
        a = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
        b = Image.new("RGBA", (10, 10), (0, 0, 255, 255))
        assert _img_to_data_uri(a) != _img_to_data_uri(b)


class TestPieceLabelNoColor:
    def _piece(self, color=1):
        return Piece(part="3666", color=color, pos=np.zeros(3), rot=np.eye(3))

    def test_contains_part(self):
        assert "3666" in _piece_label_no_color(self._piece())

    def test_does_not_contain_color(self):
        assert "color=" not in _piece_label_no_color(self._piece())

    def test_same_for_different_colors(self):
        assert _piece_label_no_color(self._piece(1)) == _piece_label_no_color(self._piece(4))


class TestZOrderSort:
    def test_elevated_pieces_painted_last(self):
        """3062a bricks (ldY=-32) must sort after the plates (ldY=-8)."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        ordered = sorted(pieces, key=lambda p: (-p.pos[1], project_ldraw(p.pos)[2]))
        names = [p.part for p in ordered]
        assert set(names[:2]) == {"3666", "60474"}
        assert names[2:] == ["3062a", "3062a", "3062a"]