"""Unit tests for ldr2svg.diagram_compose."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import svgwrite
from PIL import Image

from ldr2svg.compose import _piece_label
from ldr2svg.diagram_compose import (
    _proj_canvas,
    _add_arrow_defs,
    _draw_floor_arrows,
    _draw_icons,
    _draw_labels,
    compose_diagram_svg,
)
from ldr2svg.parts import Piece

NS = "http://www.w3.org/2000/svg"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_dwg(tmp_path, name="t.svg"):
    return svgwrite.Drawing(str(tmp_path / name))


def _identity(val):
    return val


def _floor_pos(x=0.0, z=0.0):
    return np.array([x, 0.0, z])


def _node_pos(x=0.0, z=0.0):
    return np.array([x, -24.0, z])


def _parse_svg(path):
    return ET.parse(path).getroot()


def _minimal_renders():
    """Two synthetic renders (in-memory only) at two different LDraw positions."""
    pieces = [
        Piece(part="3003", color=1,  pos=np.array([ 0.0, -24.0,  0.0]), rot=np.eye(3)),
        Piece(part="3003", color=4,  pos=np.array([80.0, -24.0,  0.0]), rot=np.eye(3)),
    ]
    img = Image.new("RGBA", (100, 80))
    return {
        _piece_label(p): (img, 50.0, 40.0, [p])
        for p in pieces
    }


def _minimal_node_data(icon_path=None, label="test-node"):
    return [{
        "pos":       _node_pos(0.0, 0.0),
        "icon_path": icon_path,
        "label":     label,
        "half_w":    20,
    }]


def _make_icon(tmp_path, name="icon.png", size=(32, 32)):
    path = tmp_path / name
    Image.new("RGBA", size, (200, 100, 50, 255)).save(path, "PNG")
    return str(path)


# ---------------------------------------------------------------------------
# _proj_canvas
# ---------------------------------------------------------------------------

class TestProjCanvas:
    def test_returns_two_values(self):
        result = _proj_canvas(np.zeros(3), _identity, _identity)
        assert len(result) == 2

    def test_both_values_are_floats(self):
        x, y = _proj_canvas(np.zeros(3), _identity, _identity)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_origin_is_finite(self):
        x, y = _proj_canvas(np.zeros(3), _identity, _identity)
        assert abs(x) < 1e6
        assert abs(y) < 1e6

    def test_positive_x_increases_screen_x(self):
        x0, _ = _proj_canvas(_floor_pos(  0), _identity, _identity)
        x1, _ = _proj_canvas(_floor_pos(100), _identity, _identity)
        assert x1 > x0

    def test_cx_cy_offsets_applied(self):
        offset = 50.0

        def cx(v): return v + offset
        def cy(v): return v + offset

        x_plain, y_plain = _proj_canvas(np.zeros(3), _identity, _identity)
        x_off,   y_off   = _proj_canvas(np.zeros(3), cx, cy)
        assert x_off == pytest.approx(x_plain + offset)
        assert y_off == pytest.approx(y_plain + offset)


# ---------------------------------------------------------------------------
# _add_arrow_defs
# ---------------------------------------------------------------------------

class TestAddArrowDefs:
    def _get_root(self, tmp_path):
        dwg = _make_dwg(tmp_path)
        _add_arrow_defs(dwg)
        dwg.save()
        return _parse_svg(tmp_path / "t.svg")

    def test_marker_added_to_defs(self, tmp_path):
        root = self._get_root(tmp_path)
        defs = root.find(f"{{{NS}}}defs")
        markers = defs.findall(f"{{{NS}}}marker")
        assert len(markers) == 1

    def test_marker_id_is_arrow(self, tmp_path):
        root = self._get_root(tmp_path)
        marker = root.find(f".//{{{NS}}}marker")
        assert marker.attrib["id"] == "arrow"

    def test_marker_orient_auto(self, tmp_path):
        root = self._get_root(tmp_path)
        marker = root.find(f".//{{{NS}}}marker")
        assert marker.attrib.get("orient") == "auto"

    def test_marker_has_polygon(self, tmp_path):
        root = self._get_root(tmp_path)
        polygon = root.find(f".//{{{NS}}}polygon")
        assert polygon is not None

    def test_polygon_has_three_points(self, tmp_path):
        root = self._get_root(tmp_path)
        polygon = root.find(f".//{{{NS}}}polygon")
        # SVG polygon "points" attribute: 3 pairs = 6 numbers
        points_str = polygon.attrib.get("points", "")
        coords = points_str.replace(",", " ").split()
        assert len(coords) == 6


# ---------------------------------------------------------------------------
# _draw_floor_arrows
# ---------------------------------------------------------------------------

class TestDrawFloorArrows:
    def _run(self, tmp_path, arrows, name="arrows.svg"):
        dwg = _make_dwg(tmp_path, name)
        _add_arrow_defs(dwg)
        _draw_floor_arrows(dwg, arrows, _identity, _identity)
        dwg.save()
        return _parse_svg(tmp_path / name)

    def _arrow_lines(self, root):
        return root.findall(f".//{{{NS}}}line[@stroke='#888']")

    def test_empty_input_produces_no_lines(self, tmp_path):
        root = self._run(tmp_path, [])
        assert len(self._arrow_lines(root)) == 0

    def test_one_arrow_produces_one_line(self, tmp_path):
        arrows = [(_floor_pos(0, 0), _floor_pos(80, 0))]
        root = self._run(tmp_path, arrows)
        assert len(self._arrow_lines(root)) == 1

    def test_arrow_count_matches_input(self, tmp_path):
        arrows = [
            (_floor_pos(0,   0), _floor_pos(80,  0)),
            (_floor_pos(80,  0), _floor_pos(160, 0)),
            (_floor_pos(160, 0), _floor_pos(240, 0)),
        ]
        root = self._run(tmp_path, arrows)
        assert len(self._arrow_lines(root)) == 3

    def test_lines_have_marker_end_attribute(self, tmp_path):
        arrows = [(_floor_pos(0, 0), _floor_pos(80, 0))]
        root = self._run(tmp_path, arrows)
        line = self._arrow_lines(root)[0]
        assert "marker-end" in line.attrib
        assert "arrow" in line.attrib["marker-end"]

    def test_coincident_points_skipped(self, tmp_path):
        arrows = [(_floor_pos(0, 0), _floor_pos(0, 0))]
        root = self._run(tmp_path, arrows, name="skip.svg")
        assert len(self._arrow_lines(root)) == 0

    def test_arrows_group_present(self, tmp_path):
        root = self._run(tmp_path, [])
        grp = root.find(f".//{{{NS}}}g[@id='arrows']")
        assert grp is not None


# ---------------------------------------------------------------------------
# _draw_icons
# ---------------------------------------------------------------------------

class TestDrawIcons:
    def _run(self, tmp_path, node_data, name="icons.svg"):
        dwg = _make_dwg(tmp_path, name)
        _draw_icons(dwg, node_data, _identity, _identity)
        dwg.save()
        return _parse_svg(tmp_path / name)

    def _images(self, root):
        return root.findall(f".//{{{NS}}}image")

    def test_none_icon_path_produces_no_image(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data(icon_path=None))
        assert len(self._images(root)) == 0

    def test_invalid_path_skipped_gracefully(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data(icon_path="/no/such/file.png"),
                         name="inv.svg")
        assert len(self._images(root)) == 0

    def test_valid_icon_produces_one_image(self, tmp_path):
        icon = _make_icon(tmp_path)
        root = self._run(tmp_path, _minimal_node_data(icon_path=icon))
        assert len(self._images(root)) == 1

    def test_image_has_matrix_transform(self, tmp_path):
        icon = _make_icon(tmp_path)
        root = self._run(tmp_path, _minimal_node_data(icon_path=icon))
        image = self._images(root)[0]
        assert "transform" in image.attrib
        assert image.attrib["transform"].startswith("matrix(")

    def test_matrix_has_six_parameters(self, tmp_path):
        icon = _make_icon(tmp_path)
        root = self._run(tmp_path, _minimal_node_data(icon_path=icon))
        transform = self._images(root)[0].attrib["transform"]
        inner = transform[len("matrix("):-1]
        values = [v.strip() for v in inner.split(",")]
        assert len(values) == 6

    def test_image_is_base64_png(self, tmp_path):
        icon = _make_icon(tmp_path)
        root = self._run(tmp_path, _minimal_node_data(icon_path=icon))
        image = self._images(root)[0]
        href = (image.attrib.get("href")
                or image.attrib.get("{http://www.w3.org/1999/xlink}href", ""))
        assert href.startswith("data:image/png;base64,")

    def test_multiple_icons(self, tmp_path):
        icon = _make_icon(tmp_path)
        node_data = [
            {"pos": _node_pos(  0), "icon_path": icon,  "label": "a", "half_w": 20},
            {"pos": _node_pos( 80), "icon_path": None,  "label": "b", "half_w": 20},
            {"pos": _node_pos(160), "icon_path": icon,  "label": "c", "half_w": 20},
        ]
        root = self._run(tmp_path, node_data, name="multi.svg")
        assert len(self._images(root)) == 2

    def test_icons_group_present(self, tmp_path):
        root = self._run(tmp_path, [])
        grp = root.find(f".//{{{NS}}}g[@id='icons']")
        assert grp is not None


# ---------------------------------------------------------------------------
# _draw_labels
# ---------------------------------------------------------------------------

class TestDrawLabels:
    def _run(self, tmp_path, node_data, name="labels.svg"):
        dwg = _make_dwg(tmp_path, name)
        _draw_labels(dwg, node_data, _identity, _identity)
        dwg.save()
        return _parse_svg(tmp_path / name)

    def _texts(self, root):
        return root.findall(f".//{{{NS}}}text")

    def test_nonempty_label_produces_text(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data())
        assert len(self._texts(root)) == 1

    def test_empty_label_skipped(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data(label=""), name="empty.svg")
        assert len(self._texts(root)) == 0

    def test_text_content_matches_label(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data(label="my-service"))
        text = self._texts(root)[0]
        assert text.text == "my-service"

    def test_text_has_translate_scale_rotate_transform(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data())
        transform = self._texts(root)[0].attrib.get("transform", "")
        assert "translate" in transform
        assert "scale" in transform
        assert "rotate" in transform

    def test_rotate_minus_30(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data())
        transform = self._texts(root)[0].attrib.get("transform", "")
        assert "rotate(-30)" in transform

    def test_mixed_empty_and_nonempty_labels(self, tmp_path):
        node_data = [
            {"pos": _node_pos(  0), "label": "A", "half_w": 20},
            {"pos": _node_pos( 80), "label": "",  "half_w": 20},
            {"pos": _node_pos(160), "label": "C", "half_w": 20},
        ]
        root = self._run(tmp_path, node_data, name="mixed.svg")
        assert len(self._texts(root)) == 2

    def test_labels_group_present(self, tmp_path):
        root = self._run(tmp_path, [])
        grp = root.find(f".//{{{NS}}}g[@id='labels']")
        assert grp is not None


# ---------------------------------------------------------------------------
# compose_diagram_svg — integration (synthetic renders, no OpenSCAD)
# ---------------------------------------------------------------------------

class TestComposeDiagramSvg:
    def _run(self, tmp_path, arrows=None, node_data=None, name="out.svg"):
        output = str(tmp_path / name)
        compose_diagram_svg(
            _minimal_renders(),
            output,
            arrows=arrows or [],
            node_data=node_data or [],
        )
        return output, _parse_svg(output)

    def test_creates_svg_file(self, tmp_path):
        output, _ = self._run(tmp_path)
        assert Path(output).exists()

    def test_svg_dimensions_positive(self, tmp_path):
        _, root = self._run(tmp_path)
        w = int(root.attrib["width"].replace("px", ""))
        h = int(root.attrib["height"].replace("px", ""))
        assert w > 0
        assert h > 0

    def test_has_background_rect(self, tmp_path):
        _, root = self._run(tmp_path)
        rect = root.find(f"{{{NS}}}rect")
        assert rect is not None
        assert rect.attrib.get("fill") == "#f8f8f0"

    def test_has_grid_group(self, tmp_path):
        _, root = self._run(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='grid']")
        assert grp is not None

    def test_has_arrows_group(self, tmp_path):
        _, root = self._run(tmp_path, arrows=[(_floor_pos(0), _floor_pos(80))])
        grp = root.find(f".//{{{NS}}}g[@id='arrows']")
        assert grp is not None

    def test_has_icons_group(self, tmp_path):
        _, root = self._run(tmp_path, node_data=_minimal_node_data())
        grp = root.find(f".//{{{NS}}}g[@id='icons']")
        assert grp is not None

    def test_has_labels_group(self, tmp_path):
        _, root = self._run(tmp_path, node_data=_minimal_node_data())
        grp = root.find(f".//{{{NS}}}g[@id='labels']")
        assert grp is not None

    def test_arrow_count_in_svg(self, tmp_path):
        arrows = [(_floor_pos(0), _floor_pos(80)), (_floor_pos(80), _floor_pos(160))]
        _, root = self._run(tmp_path, arrows=arrows, name="arrows.svg")
        lines = root.findall(f".//{{{NS}}}line[@stroke='#888']")
        assert len(lines) == 2

    def test_label_text_content(self, tmp_path):
        _, root = self._run(tmp_path, node_data=_minimal_node_data(label="my-node"),
                            name="label.svg")
        texts = root.findall(f".//{{{NS}}}text")
        assert any(t.text == "my-node" for t in texts)

    def test_brick_use_elements_present(self, tmp_path):
        _, root = self._run(tmp_path)
        uses = root.findall(f".//{{{NS}}}use")
        # Two pieces in _minimal_renders → two <use> elements
        assert len(uses) == 2

    def test_layer_order_arrows_after_bricks(self, tmp_path):
        """Arrow group must appear after brick <use> elements in SVG."""
        arrows = [(_floor_pos(0), _floor_pos(80))]
        output, _ = self._run(tmp_path, arrows=arrows, name="order.svg")
        svg_text = Path(output).read_text()
        assert svg_text.index('id="arrows"') > svg_text.rindex("<use")

    def test_layer_order_icons_after_bricks(self, tmp_path):
        """Icon group must appear after the last brick <use> element."""
        icon = _make_icon(tmp_path)
        output, _ = self._run(tmp_path,
                               node_data=_minimal_node_data(icon_path=icon),
                               name="iconorder.svg")
        svg_text = Path(output).read_text()
        last_use = svg_text.rindex("<use")
        icons_pos = svg_text.index('id="icons"')
        assert icons_pos > last_use


# ---------------------------------------------------------------------------
# compose_diagram_svg — piece_groups / <g> cluster grouping
# ---------------------------------------------------------------------------

def _make_piece_groups(renders):
    """Build minimal piece_groups from _minimal_renders() pieces."""
    all_pieces = [p for _, _, _, pl in renders.values() for p in pl]
    return [("cluster_test", all_pieces[:1]), ("lone", all_pieces[1:])]


class TestComposeDiagramSvgGroups:
    def _run_grouped(self, tmp_path, name="grouped.svg"):
        renders = _minimal_renders()
        output = str(tmp_path / name)
        piece_groups = _make_piece_groups(renders)
        compose_diagram_svg(renders, output, arrows=[], node_data=[],
                            piece_groups=piece_groups)
        return output, ET.parse(output).getroot()

    def test_cluster_g_present(self, tmp_path):
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='cluster_test']")
        assert grp is not None

    def test_lone_g_present(self, tmp_path):
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='lone']")
        assert grp is not None

    def test_uses_inside_cluster_g(self, tmp_path):
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='cluster_test']")
        uses = grp.findall(f"{{{NS}}}use")
        assert len(uses) == 1

    def test_uses_inside_lone_g(self, tmp_path):
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='lone']")
        uses = grp.findall(f"{{{NS}}}use")
        assert len(uses) == 1

    def test_total_use_count_unchanged(self, tmp_path):
        _, root = self._run_grouped(tmp_path)
        uses = root.findall(f".//{{{NS}}}use")
        assert len(uses) == 2

    def test_cluster_g_before_lone_g(self, tmp_path):
        output, _ = self._run_grouped(tmp_path)
        svg_text = Path(output).read_text()
        assert svg_text.index('id="cluster_test"') < svg_text.index('id="lone"')
