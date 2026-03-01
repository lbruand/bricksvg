"""Unit tests for ldr2svg.ldr2png_svg (build_pngs_white, _render_one_white)."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from ldr2svg.parts import parse_ldr, PART_MAP, Piece, ldraw_rgb
from ldr2svg.ldr2png_svg import _render_one_white, build_pngs_white, _colorize, colorize_renders
from ldr2svg.compose import _piece_label_no_color

LDR_PATH = Path(__file__).parent.parent / "test.ldr"

_FAKE_IMG = Image.new("RGBA", (100, 80))


def _fake_remove_and_crop(_path):
    return _FAKE_IMG, 50.0, 40.0


class TestRenderOneWhite:
    def _piece(self):
        return Piece(part="3666", color=4, pos=np.zeros(3), rot=np.eye(3))

    def test_success_returns_img_and_anchors(self, tmp_path):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            result = _render_one_white(0, self._piece(), 1, tmp_path, keep_pngs=False)
        assert result is not None
        img, ax, ay = result
        assert img is _FAKE_IMG
        assert ax == pytest.approx(50.0)
        assert ay == pytest.approx(40.0)

    def test_failed_render_returns_none(self, tmp_path):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=False):
            result = _render_one_white(0, self._piece(), 1, tmp_path, keep_pngs=False)
        assert result is None

    def test_keep_pngs_false_deletes_file(self, tmp_path):
        def fake_render(scad_src, png_path):
            png_path.write_bytes(b"")
            return True
        with patch("ldr2svg.ldr2png_svg.render_piece", side_effect=fake_render), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            _render_one_white(0, self._piece(), 1, tmp_path, keep_pngs=False)
        assert list(tmp_path.glob("*.png")) == []

    def test_keep_pngs_true_retains_file(self, tmp_path):
        def fake_render(scad_src, png_path):
            png_path.write_bytes(b"")
            return True
        with patch("ldr2svg.ldr2png_svg.render_piece", side_effect=fake_render), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            _render_one_white(0, self._piece(), 1, tmp_path, keep_pngs=True)
        assert len(list(tmp_path.glob("*.png"))) == 1

    def test_renders_in_white(self, tmp_path):
        """make_scad must be called with (255, 255, 255) regardless of piece.color."""
        with patch("ldr2svg.ldr2png_svg.make_scad", return_value="") as mock_scad, \
             patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            _render_one_white(0, self._piece(), 1, tmp_path, keep_pngs=False)
        _part, color, _rot = mock_scad.call_args[0]
        assert color == (255, 255, 255)


class TestBuildPngsWhite:
    def _run(self, pieces, tmp_path, keep_pngs=False):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            return build_pngs_white(pieces, tmp_path, keep_pngs=keep_pngs)

    def test_known_parts_rendered(self, tmp_path):
        """test.ldr has 3 unique (part, rotation) combos → 3 white renders."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        assert len(self._run(pieces, tmp_path)) == 3

    def test_unknown_part_skipped(self, tmp_path):
        unknown = Piece(part="unknown_part_xyz", color=1,
                        pos=np.zeros(3), rot=np.eye(3))
        assert self._run([unknown], tmp_path) == {}

    def test_identical_pieces_deduplicated(self, tmp_path):
        """Three identical 3062a pieces → render_piece called once → one renders entry."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part == "3062a"]
        assert len(pieces) == 3
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True) as mock_render, \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            renders = build_pngs_white(pieces, tmp_path)
        assert len(renders) == 1
        mock_render.assert_called_once()

    def test_different_colors_same_part_deduplicated(self, tmp_path):
        """Two pieces with same part+rotation but different colours → one white render."""
        p1 = Piece(part="3666", color=1, pos=np.zeros(3), rot=np.eye(3))
        p2 = Piece(part="3666", color=4, pos=np.zeros(3), rot=np.eye(3))
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True) as mock_render, \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            renders = build_pngs_white([p1, p2], tmp_path)
        assert len(renders) == 1
        mock_render.assert_called_once()
        # both pieces must appear in the pieces list for that label
        _img, _ax, _ay, grouped = next(iter(renders.values()))
        assert len(grouped) == 2

    def test_failed_render_skipped(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=False), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            assert build_pngs_white(pieces, tmp_path) == {}

    def test_keep_pngs_false_removes_files(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]
        self._run(pieces, tmp_path, keep_pngs=False)
        assert list(tmp_path.glob("*.png")) == []

    def test_keep_pngs_true_retains_files(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]

        def fake_render(scad_src, png_path):
            png_path.write_bytes(b"")
            return True

        with patch("ldr2svg.ldr2png_svg.render_piece", side_effect=fake_render), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            build_pngs_white(pieces, tmp_path, keep_pngs=True)
        assert len(list(tmp_path.glob("*.png"))) == 1


class TestColorize:
    def _white_img(self, size=(20, 20)):
        return Image.new("RGBA", size, (255, 255, 255, 255))

    def _black_img(self, size=(20, 20)):
        return Image.new("RGBA", size, (0, 0, 0, 255))

    def test_returns_rgba_image(self):
        result = _colorize(self._white_img(), color_id=15)  # white
        assert result.mode == "RGBA"

    def test_same_size_as_input(self):
        result = _colorize(self._white_img((30, 40)), color_id=1)
        assert result.size == (30, 40)

    def test_white_pixels_become_target_color(self):
        color_id = 4  # red in LDraw
        r, g, b = ldraw_rgb(color_id)
        result = _colorize(self._white_img(), color_id=color_id)
        pixel = result.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert pixel[0] == pytest.approx(r, abs=1)
        assert pixel[1] == pytest.approx(g, abs=1)
        assert pixel[2] == pytest.approx(b, abs=1)

    def test_black_pixels_stay_black(self):
        result = _colorize(self._black_img(), color_id=4)
        pixel = result.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert pixel[0] == 0
        assert pixel[1] == 0
        assert pixel[2] == 0

    def test_alpha_preserved(self):
        img = Image.new("RGBA", (10, 10), (255, 255, 255, 128))
        result = _colorize(img, color_id=4)
        pixel = result.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        assert pixel[3] == 128


class TestColorizeRenders:
    def _white_renders(self):
        """Two parts, each with two colors."""
        p1a = Piece(part="3003", color=1, pos=np.zeros(3), rot=np.eye(3))
        p1b = Piece(part="3003", color=4, pos=np.zeros(3), rot=np.eye(3))
        p2a = Piece(part="3666", color=1, pos=np.zeros(3), rot=np.eye(3))
        img = Image.new("RGBA", (100, 80), (255, 255, 255, 255))
        return {
            _piece_label_no_color(p1a): (img, 50.0, 40.0, [p1a, p1b]),
            _piece_label_no_color(p2a): (img, 30.0, 20.0, [p2a]),
        }

    def test_one_entry_per_color_variant(self):
        result = colorize_renders(self._white_renders())
        # 3003 has 2 colors → 2 entries; 3666 has 1 color → 1 entry
        assert len(result) == 3

    def test_keys_include_color(self):
        result = colorize_renders(self._white_renders())
        assert all("color=" in label for label in result)

    def test_output_images_are_rgba(self):
        result = colorize_renders(self._white_renders())
        for img, *_ in result.values():
            assert img.mode == "RGBA"

    def test_anchors_preserved(self):
        result = colorize_renders(self._white_renders())
        for label, (_, ax, ay, _pieces) in result.items():
            if "3003" in label:
                assert ax == pytest.approx(50.0)
                assert ay == pytest.approx(40.0)

    def test_pieces_grouped_by_color(self):
        result = colorize_renders(self._white_renders())
        for _, (_, _, _, pieces) in result.items():
            colors = {p.color for p in pieces}
            assert len(colors) == 1  # each entry has exactly one color
