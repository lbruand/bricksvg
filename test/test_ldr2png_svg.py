"""Unit tests for ldr2svg.ldr2png_svg (build_pngs, _render_one)."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from ldr2svg.parts import parse_ldr, PART_MAP, Piece
from ldr2svg.ldr2png_svg import _render_one, build_pngs

LDR_PATH = Path(__file__).parent.parent / "test.ldr"

_FAKE_IMG = Image.new("RGBA", (100, 80))


def _fake_remove_and_crop(_path):
    return _FAKE_IMG, 50.0, 40.0


class TestRenderOne:
    def _piece(self):
        return Piece(part="3666", color=4, pos=np.zeros(3), rot=np.eye(3))

    def test_success_returns_img_and_anchors(self, tmp_path):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            result = _render_one(0, self._piece(), 1, tmp_path, keep_pngs=False)
        assert result is not None
        img, ax, ay = result
        assert img is _FAKE_IMG
        assert ax == pytest.approx(50.0)
        assert ay == pytest.approx(40.0)

    def test_failed_render_returns_none(self, tmp_path):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=False):
            result = _render_one(0, self._piece(), 1, tmp_path, keep_pngs=False)
        assert result is None

    def test_keep_pngs_false_deletes_file(self, tmp_path):
        def fake_render(scad_src, png_path):
            png_path.write_bytes(b"")
            return True
        with patch("ldr2svg.ldr2png_svg.render_piece", side_effect=fake_render), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            _render_one(0, self._piece(), 1, tmp_path, keep_pngs=False)
        assert list(tmp_path.glob("*.png")) == []

    def test_keep_pngs_true_retains_file(self, tmp_path):
        def fake_render(scad_src, png_path):
            png_path.write_bytes(b"")
            return True
        with patch("ldr2svg.ldr2png_svg.render_piece", side_effect=fake_render), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            _render_one(0, self._piece(), 1, tmp_path, keep_pngs=True)
        assert len(list(tmp_path.glob("*.png"))) == 1


class TestBuildPngs:
    def _run(self, pieces, tmp_path, keep_pngs=False):
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            return build_pngs(pieces, tmp_path, keep_pngs=keep_pngs)

    def test_known_parts_rendered(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        assert len(self._run(pieces, tmp_path)) == 3

    def test_unknown_part_skipped(self, tmp_path):
        unknown = Piece(part="unknown_part_xyz", color=1,
                        pos=np.zeros(3), rot=np.eye(3))
        assert self._run([unknown], tmp_path) == {}

    def test_identical_pieces_cached(self, tmp_path):
        """Three identical 3062a pieces → render_piece called once → one renders entry."""
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part == "3062a"]
        assert len(pieces) == 3
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=True) as mock_render, \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            renders = build_pngs(pieces, tmp_path)
        assert len(renders) == 1
        mock_render.assert_called_once()

    def test_renders_has_unique_entries(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP]
        assert len(self._run(pieces, tmp_path)) == 3

    def test_failed_render_skipped(self, tmp_path):
        pieces = [p for p in parse_ldr(LDR_PATH) if p.part in PART_MAP][:1]
        with patch("ldr2svg.ldr2png_svg.render_piece", return_value=False), \
             patch("ldr2svg.ldr2png_svg.remove_and_crop", side_effect=_fake_remove_and_crop):
            assert build_pngs(pieces, tmp_path) == {}

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
            build_pngs(pieces, tmp_path, keep_pngs=True)
        assert len(list(tmp_path.glob("*.png"))) == 1
