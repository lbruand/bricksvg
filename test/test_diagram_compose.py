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
    _arrow_polygon_3d,
    _make_iso_arrow,
    _draw_floor_arrows,
    _draw_icons,
    _draw_labels,
    _draw_cluster_labels,
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
        "half_h":    16,
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
# _arrow_polygon_3d
# ---------------------------------------------------------------------------

class TestArrowPolygon3d:
    def _arrow(self, x0=0.0, z0=0.0, x1=80.0, z1=0.0, y=-32.0):
        return _arrow_polygon_3d(
            np.array([x0, y, z0]),
            np.array([x1, y, z1]),
        )

    def test_returns_nonempty_list(self):
        assert len(self._arrow()) > 0

    def test_zero_length_returns_empty(self):
        result = _arrow_polygon_3d(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        assert result == []

    def test_all_vertices_share_y(self):
        y = -32.0
        for v in self._arrow(y=y):
            assert v[1] == pytest.approx(y)

    def test_tip_is_to_pos(self):
        verts = self._arrow(x1=80.0, z1=0.0, y=-32.0)
        xs = [v[0] for v in verts]
        zs = [v[2] for v in verts]
        # tip (to_pos) must be one of the vertices
        assert any(abs(x - 80.0) < 1e-6 and abs(z - 0.0) < 1e-6 for x, z in zip(xs, zs))

    def test_polygon_has_enough_vertices(self):
        # n_body=16 → 17 left + 17 right + 3 head = 37 vertices minimum
        assert len(self._arrow()) >= 37


# ---------------------------------------------------------------------------
# _draw_floor_arrows
# ---------------------------------------------------------------------------

class TestDrawFloorArrows:
    def _run(self, tmp_path, arrows, name="arrows.svg"):
        dwg = _make_dwg(tmp_path, name)
        _draw_floor_arrows(dwg, arrows, _identity, _identity)
        dwg.save()
        return _parse_svg(tmp_path / name)

    def _arrow_polygons(self, root):
        return root.findall(f".//{{{NS}}}polygon[@fill='#888']")

    def test_empty_input_produces_no_polygons(self, tmp_path):
        root = self._run(tmp_path, [])
        assert len(self._arrow_polygons(root)) == 0

    def test_one_arrow_produces_one_polygon(self, tmp_path):
        arrows = [(_floor_pos(0, 0), _floor_pos(80, 0))]
        root = self._run(tmp_path, arrows)
        assert len(self._arrow_polygons(root)) == 1

    def test_arrow_count_matches_input(self, tmp_path):
        arrows = [
            (_floor_pos(0,   0), _floor_pos(80,  0)),
            (_floor_pos(80,  0), _floor_pos(160, 0)),
            (_floor_pos(160, 0), _floor_pos(240, 0)),
        ]
        root = self._run(tmp_path, arrows)
        assert len(self._arrow_polygons(root)) == 3

    def test_polygons_have_fill(self, tmp_path):
        arrows = [(_floor_pos(0, 0), _floor_pos(80, 0))]
        root = self._run(tmp_path, arrows)
        poly = self._arrow_polygons(root)[0]
        assert poly.attrib.get("fill") == "#888"

    def test_coincident_points_skipped(self, tmp_path):
        arrows = [(_floor_pos(0, 0), _floor_pos(0, 0))]
        root = self._run(tmp_path, arrows, name="skip.svg")
        assert len(self._arrow_polygons(root)) == 0

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
            {"pos": _node_pos(  0), "icon_path": icon,  "label": "a", "half_w": 20, "half_h": 16},
            {"pos": _node_pos( 80), "icon_path": None,  "label": "b", "half_w": 20, "half_h": 16},
            {"pos": _node_pos(160), "icon_path": icon,  "label": "c", "half_w": 20, "half_h": 16},
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

    def test_text_has_matrix_transform(self, tmp_path):
        root = self._run(tmp_path, _minimal_node_data())
        transform = self._texts(root)[0].attrib.get("transform", "")
        assert transform.startswith("matrix(")

    def test_matrix_encodes_front_face_isometric(self, tmp_path):
        """a=0.866025, b=0.5, c=0, d=1 — left/front face projection."""
        root = self._run(tmp_path, _minimal_node_data())
        transform = self._texts(root)[0].attrib.get("transform", "")
        inner = transform[len("matrix("):-1]
        a, b, c, d, *_ = [v.strip() for v in inner.split(",")]
        assert float(a) == pytest.approx(0.866025, abs=1e-4)
        assert float(b) == pytest.approx(0.5,      abs=1e-4)
        assert float(c) == pytest.approx(0.0,      abs=1e-4)
        assert float(d) == pytest.approx(1.0,      abs=1e-4)

    def test_mixed_empty_and_nonempty_labels(self, tmp_path):
        node_data = [
            {"pos": _node_pos(  0), "label": "A", "half_w": 20, "half_h": 16},
            {"pos": _node_pos( 80), "label": "",  "half_w": 20, "half_h": 16},
            {"pos": _node_pos(160), "label": "C", "half_w": 20, "half_h": 16},
        ]
        root = self._run(tmp_path, node_data, name="mixed.svg")
        assert len(self._texts(root)) == 2

    def test_labels_group_present(self, tmp_path):
        root = self._run(tmp_path, [])
        grp = root.find(f".//{{{NS}}}g[@id='labels']")
        assert grp is not None


# ---------------------------------------------------------------------------
# _draw_cluster_labels
# ---------------------------------------------------------------------------

class TestDrawClusterLabels:
    def _cluster_data(self, label="Cluster A"):
        return [{"pos": np.array([0.0, -8.0, 40.0]), "label": label}]

    def _run(self, tmp_path, cluster_data, name="clabels.svg"):
        dwg = _make_dwg(tmp_path, name)
        _draw_cluster_labels(dwg, cluster_data, _identity, _identity)
        dwg.save()
        return _parse_svg(tmp_path / name)

    def _texts(self, root):
        return root.findall(f".//{{{NS}}}text")

    def test_nonempty_label_produces_text(self, tmp_path):
        root = self._run(tmp_path, self._cluster_data())
        assert len(self._texts(root)) == 1

    def test_empty_label_skipped(self, tmp_path):
        root = self._run(tmp_path, self._cluster_data(label=""), name="empty.svg")
        assert len(self._texts(root)) == 0

    def test_text_content_matches_label(self, tmp_path):
        root = self._run(tmp_path, self._cluster_data("MyCluster"))
        assert self._texts(root)[0].text == "MyCluster"

    def test_text_has_matrix_transform(self, tmp_path):
        root = self._run(tmp_path, self._cluster_data())
        transform = self._texts(root)[0].attrib.get("transform", "")
        assert transform.startswith("matrix(")

    def test_matrix_encodes_isometric_axes(self, tmp_path):
        """The matrix must use the standard isometric top-face coefficients."""
        root = self._run(tmp_path, self._cluster_data())
        transform = self._texts(root)[0].attrib.get("transform", "")
        assert "0.866025" in transform
        assert "0.5" in transform
        assert "-0.866025" in transform

    def test_cluster_labels_group_present(self, tmp_path):
        root = self._run(tmp_path, [])
        grp = root.find(f".//{{{NS}}}g[@id='cluster_labels']")
        assert grp is not None

    def test_multiple_clusters(self, tmp_path):
        cluster_data = [
            {"pos": np.array([  0.0, -8.0,  40.0]), "label": "A"},
            {"pos": np.array([200.0, -8.0, 140.0]), "label": "B"},
        ]
        root = self._run(tmp_path, cluster_data, name="multi.svg")
        assert len(self._texts(root)) == 2


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
        polys = root.findall(f".//{{{NS}}}polygon[@fill='#888']")
        assert len(polys) == 2

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
# compose_diagram_svg — piece_groups (platform <g> + globally-sorted nodes)
# ---------------------------------------------------------------------------

def _make_piece_groups_with_platform(renders):
    """Build piece_groups that include a 3024 platform piece and a 3003 node brick."""
    all_pieces = [p for _, _, _, pl in renders.values() for p in pl]
    # Wrap the first piece as a 3024 platform tile so the grouping path is exercised.
    platform_piece = Piece(part="3024", color=15,
                           pos=np.array([0.0, 0.0, 0.0]), rot=np.eye(3))
    # Re-use the same image as the 3003 piece (same size) so _build_defs finds it.
    img = Image.new("RGBA", (100, 80))
    from ldr2svg.compose import _piece_label
    platform_label = _piece_label(platform_piece)
    renders[platform_label] = (img, 50.0, 40.0, [platform_piece])
    return [("cluster_test", [platform_piece] + all_pieces[:1]), ("lone", all_pieces[1:])]


class TestComposeDiagramSvgGroups:
    def _run_grouped(self, tmp_path, name="grouped.svg"):
        renders = _minimal_renders()
        output = str(tmp_path / name)
        piece_groups = _make_piece_groups_with_platform(renders)
        compose_diagram_svg(renders, output, arrows=[], node_data=[],
                            piece_groups=piece_groups)
        return output, ET.parse(output).getroot()

    def test_total_use_count_with_piece_groups(self, tmp_path):
        """All pieces from piece_groups must appear as <use> elements."""
        _, root = self._run_grouped(tmp_path)
        uses = root.findall(f".//{{{NS}}}use")
        # 1 platform (3024) + 2 node bricks (3003) = 3
        assert len(uses) == 3

    def test_platform_group_present(self, tmp_path):
        """3024 tiles must be wrapped in a <g id='platform_cluster_test'>."""
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='platform_cluster_test']")
        assert grp is not None

    def test_platform_group_contains_use(self, tmp_path):
        """The platform <g> must contain at least one <use> element."""
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='platform_cluster_test']")
        assert grp is not None
        uses = grp.findall(f"{{{NS}}}use")
        assert len(uses) >= 1

    def test_no_platform_group_for_lone(self, tmp_path):
        """The 'lone' group has no 3024 pieces, so no platform <g id='platform_lone'>."""
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='platform_lone']")
        assert grp is None

    def test_node_group_present_for_cluster(self, tmp_path):
        """Node bricks must be wrapped in <g id='nodes_cluster_test'>."""
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='nodes_cluster_test']")
        assert grp is not None

    def test_node_group_present_for_lone(self, tmp_path):
        """Lone node bricks must be wrapped in <g id='nodes_lone'>."""
        _, root = self._run_grouped(tmp_path)
        grp = root.find(f".//{{{NS}}}g[@id='nodes_lone']")
        assert grp is not None

    def test_node_group_contains_use(self, tmp_path):
        """Each nodes <g> must contain at least one <use> element."""
        _, root = self._run_grouped(tmp_path)
        for gid in ("nodes_cluster_test", "nodes_lone"):
            grp = root.find(f".//{{{NS}}}g[@id='{gid}']")
            assert grp is not None
            assert len(grp.findall(f"{{{NS}}}use")) >= 1
