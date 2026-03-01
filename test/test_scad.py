"""Unit tests for ldr2svg.scad."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ldr2svg.parts import PART_MAP
from ldr2svg.scad import make_scad, render_piece, remove_and_crop

SAMPLE_PART = PART_MAP["3003"]   # 2×2 brick, height=1


# ---------------------------------------------------------------------------
# make_scad  (pure string generation — no OpenSCAD required)
# ---------------------------------------------------------------------------

class TestMakeScad:
    def test_returns_string(self):
        assert isinstance(make_scad(SAMPLE_PART, (255, 0, 0), np.eye(3)), str)

    def test_includes_bricklib(self):
        assert "use <" in make_scad(SAMPLE_PART, (255, 0, 0), np.eye(3))

    def test_contains_block_call(self):
        assert "block(" in make_scad(SAMPLE_PART, (255, 0, 0), np.eye(3))

    def test_contains_translate(self):
        assert "translate(" in make_scad(SAMPLE_PART, (255, 0, 0), np.eye(3))

    def test_contains_multmatrix(self):
        assert "multmatrix(" in make_scad(SAMPLE_PART, (255, 0, 0), np.eye(3))

    def test_color_encoded_as_floats(self):
        src = make_scad(SAMPLE_PART, (255, 0, 128), np.eye(3))
        assert "1.000" in src          # 255/255
        assert "0.000" in src          # 0/255
        assert "0.502" in src          # round(128/255, 3)

    def test_part_width_and_length_in_output(self):
        src = make_scad(SAMPLE_PART, (0, 0, 0), np.eye(3))
        assert f"width={SAMPLE_PART.width}" in src
        assert f"length={SAMPLE_PART.length}" in src

    def test_translate_z_is_negative_height(self):
        src = make_scad(SAMPLE_PART, (0, 0, 0), np.eye(3))
        assert f"translate([0, 0, {-SAMPLE_PART.h_mm}])" in src

    def test_non_identity_rotation_produces_matrix(self):
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        src = make_scad(SAMPLE_PART, (0, 0, 0), rot)
        assert "multmatrix(" in src


# ---------------------------------------------------------------------------
# remove_and_crop  (PIL only — no OpenSCAD required)
# ---------------------------------------------------------------------------

def _make_png(tmp_path: Path, bg=(200, 200, 200), draw_box=True) -> Path:
    """Synthetic 100×100 PNG: solid grey background, optional blue box at centre."""
    img = Image.new("RGB", (100, 100), color=bg)
    if draw_box:
        for x in range(40, 60):
            for y in range(40, 60):
                img.putpixel((x, y), (0, 0, 255))
    path = tmp_path / "piece.png"
    img.save(path)
    return path


class TestRemoveAndCrop:
    def test_returns_tuple_of_three(self, tmp_path):
        assert len(remove_and_crop(_make_png(tmp_path))) == 3

    def test_returns_pil_image(self, tmp_path):
        img, _, _ = remove_and_crop(_make_png(tmp_path))
        assert isinstance(img, Image.Image)

    def test_anchors_are_floats(self, tmp_path):
        _, ax, ay = remove_and_crop(_make_png(tmp_path))
        assert isinstance(ax, float)
        assert isinstance(ay, float)

    def test_background_made_transparent(self, tmp_path):
        img, _, _ = remove_and_crop(_make_png(tmp_path))
        arr = np.array(img.convert("RGBA"))
        # Top-left corner is background → must be fully transparent
        assert arr[0, 0, 3] == 0

    def test_foreground_remains_opaque(self, tmp_path):
        img, _, _ = remove_and_crop(_make_png(tmp_path))
        arr = np.array(img.convert("RGBA"))
        assert np.any(arr[:, :, 3] > 0)

    def test_image_cropped_tighter_than_original(self, tmp_path):
        img, _, _ = remove_and_crop(_make_png(tmp_path))
        # Content is a 20×20 box centred in 100×100 → crop much smaller
        assert img.width < 100
        assert img.height < 100

    def test_solid_background_only_handled(self, tmp_path):
        # No foreground: function must not crash; returns full image
        img, ax, ay = remove_and_crop(_make_png(tmp_path, draw_box=False))
        assert isinstance(img, Image.Image)
        assert isinstance(ax, float)
        assert isinstance(ay, float)

    def test_anchor_shifts_with_crop(self, tmp_path):
        # Anchor encodes where IMG_PX/2 falls inside the cropped image.
        # With a box centred at (50,50) in a 100px image, ax > 0 since the
        # crop starts before the original image centre.
        _, ax, ay = remove_and_crop(_make_png(tmp_path))
        assert ax > 0
        assert ay > 0


# ---------------------------------------------------------------------------
# render_piece  (requires OpenSCAD)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRenderPiece:
    def test_valid_scad_creates_png(self, tmp_path):
        src = make_scad(SAMPLE_PART, (255, 0, 0), np.eye(3))
        out = tmp_path / "out.png"
        assert render_piece(src, out)
        assert out.exists()

    def test_invalid_scad_returns_false(self, tmp_path):
        out = tmp_path / "out.png"
        assert not render_piece("this is not valid OpenSCAD !!!", out)
