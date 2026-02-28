"""Unit tests for ldr2svg.diagram_bridge."""

from pathlib import Path

import numpy as np
import pytest

from ldr2svg.diagram_bridge import (
    extract_graph,
    build_ldr_scene,
    _provider_color,
    _median_nn_dist,
    _PLATE_H_LDU,
    _BRICK_H_LDU,
)

EXAMPLE_DIAGRAM = Path(__file__).parent / "example_diagram.py"

# ---------------------------------------------------------------------------
# Minimal synthetic graph fixtures (no graphviz required)
# ---------------------------------------------------------------------------

def _lone_graph():
    """Two unconnected lone nodes (no clusters)."""
    return {
        "objects": [
            {"_gvid": 0, "pos": "0,0",   "image": "/res/k8s/pod.png", "label": "alpha"},
            {"_gvid": 1, "pos": "155,0", "image": "/res/gcp/lb.png",  "label": "beta"},
        ],
        "edges": [{"tail": 0, "head": 1}],
    }


def _cluster_graph():
    """One cluster containing one node; one lone node outside."""
    return {
        "objects": [
            {"_gvid": 0, "nodes": [1], "name": "cluster_A"},
            {"_gvid": 1, "pos": "0,0",   "image": "/res/k8s/pod.png", "label": "inside"},
            {"_gvid": 2, "pos": "155,0", "image": "/res/gcp/lb.png",  "label": "outside"},
        ],
        "edges": [{"tail": 1, "head": 2}],
    }


def _nested_cluster_graph():
    """Outer cluster A contains inner cluster B which contains node 2."""
    return {
        "objects": [
            {"_gvid": 0, "nodes": [1, 2], "name": "cluster_A"},
            {"_gvid": 1, "nodes": [2],    "name": "cluster_B"},
            {"_gvid": 2, "pos": "0,0",   "image": "/res/k8s/pod.png", "label": "deep"},
            {"_gvid": 3, "pos": "155,0", "image": "/res/gcp/lb.png",  "label": "shallow"},
        ],
        "edges": [],
    }


# ---------------------------------------------------------------------------
# extract_graph
# ---------------------------------------------------------------------------

class TestExtractGraph:
    def test_returns_dict(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        assert isinstance(result, dict)

    def test_has_objects_key(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        assert "objects" in result

    def test_has_edges_key(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        assert "edges" in result

    def test_node_objects_count(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        node_objs = [o for o in result["objects"] if "pos" in o]
        assert len(node_objs) == 5

    def test_cluster_objects_count(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        cluster_objs = [o for o in result["objects"] if "nodes" in o]
        assert len(cluster_objs) == 5

    def test_edges_count(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        assert len(result["edges"]) == 4

    def test_node_objects_have_pos_field(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        for obj in (o for o in result["objects"] if "pos" in o):
            parts = obj["pos"].split(",")
            assert len(parts) == 2
            float(parts[0]), float(parts[1])  # must be parseable as floats

    def test_node_objects_have_image_field(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        for obj in (o for o in result["objects"] if "pos" in o):
            assert "image" in obj

    def test_cluster_objects_have_nodes_list(self):
        result = extract_graph(str(EXAMPLE_DIAGRAM))
        for obj in (o for o in result["objects"] if "nodes" in o):
            assert isinstance(obj["nodes"], list)

    def test_does_not_produce_output_file(self, tmp_path, monkeypatch):
        """Running the diagram script must not write any .svg/.png file."""
        monkeypatch.chdir(tmp_path)
        extract_graph(str(EXAMPLE_DIAGRAM))
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# _provider_color
# ---------------------------------------------------------------------------

class TestProviderColor:
    def test_k8s_is_blue(self):
        assert _provider_color("/resources/k8s/compute/pod.png") == 1

    def test_gcp_is_red(self):
        assert _provider_color("/resources/gcp/network/lb.png") == 4

    def test_aws_is_orange(self):
        assert _provider_color("/resources/aws/compute/ec2.png") == 25

    def test_azure_is_light_blue(self):
        assert _provider_color("/resources/azure/compute/vm.png") == 41

    def test_onprem_is_green(self):
        assert _provider_color("/resources/onprem/network/nginx.png") == 2

    def test_unknown_is_gray(self):
        assert _provider_color("/resources/generic/icon.png") == 7

    def test_empty_string_is_gray(self):
        assert _provider_color("") == 7

    def test_matching_is_case_sensitive(self):
        # "K8S" (uppercase) should not match "k8s"
        assert _provider_color("/resources/K8S/pod.png") == 7


# ---------------------------------------------------------------------------
# _median_nn_dist
# ---------------------------------------------------------------------------

class TestMedianNnDist:
    def test_two_equidistant_points(self):
        assert _median_nn_dist([0.0, 10.0], [0.0, 0.0]) == pytest.approx(10.0)

    def test_collinear_equal_spacing(self):
        assert _median_nn_dist([0.0, 10.0, 20.0], [0.0, 0.0, 0.0]) == pytest.approx(10.0)

    def test_single_point_returns_fallback(self):
        assert _median_nn_dist([5.0], [5.0]) == pytest.approx(1.0)

    def test_result_is_positive(self):
        assert _median_nn_dist([0.0, 3.0, 8.0, 15.0], [0.0, 4.0, 0.0, 4.0]) > 0

    def test_diagonal_distance(self):
        # Two points at (0,0) and (3,4): distance = 5
        assert _median_nn_dist([0.0, 3.0], [0.0, 4.0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# build_ldr_scene — return structure
# ---------------------------------------------------------------------------

class TestBuildLdrSceneStructure:
    def test_returns_three_values(self):
        assert len(build_ldr_scene(_lone_graph())) == 3

    def test_pieces_is_list(self):
        pieces, _, _ = build_ldr_scene(_lone_graph())
        assert isinstance(pieces, list)

    def test_arrows_is_list(self):
        _, arrows, _ = build_ldr_scene(_lone_graph())
        assert isinstance(arrows, list)

    def test_node_data_is_list(self):
        _, _, node_data = build_ldr_scene(_lone_graph())
        assert isinstance(node_data, list)

    def test_node_data_count_matches_nodes(self):
        _, _, node_data = build_ldr_scene(_lone_graph())
        assert len(node_data) == 2

    def test_empty_graph_returns_empty(self):
        pieces, arrows, node_data = build_ldr_scene({"objects": [], "edges": []})
        assert pieces == []
        assert arrows == []
        assert node_data == []


# ---------------------------------------------------------------------------
# build_ldr_scene — lone nodes (no cluster)
# ---------------------------------------------------------------------------

class TestBuildLdrSceneLoneNodes:
    def setup_method(self):
        self.pieces, self.arrows, self.node_data = build_ldr_scene(_lone_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]

    def test_two_bricks_created(self):
        assert len(self.bricks) == 2

    def test_no_platform_plates(self):
        assert all(p.part != "3022" for p in self.pieces)

    def test_lone_node_y_equals_negative_brick_height(self):
        for b in self.bricks:
            assert b.pos[1] == pytest.approx(-_BRICK_H_LDU)

    def test_brick_colors_from_provider(self):
        colors = {p.color for p in self.bricks}
        assert 1 in colors   # k8s → blue
        assert 4 in colors   # gcp → red

    def test_positions_on_20ldu_grid(self):
        for b in self.bricks:
            assert b.pos[0] % 20 == pytest.approx(0, abs=1e-6)
            assert b.pos[2] % 20 == pytest.approx(0, abs=1e-6)

    def test_two_bricks_at_different_positions(self):
        assert not np.allclose(self.bricks[0].pos, self.bricks[1].pos)


# ---------------------------------------------------------------------------
# build_ldr_scene — clustered nodes
# ---------------------------------------------------------------------------

class TestBuildLdrSceneCluster:
    def setup_method(self):
        self.pieces, self.arrows, self.node_data = build_ldr_scene(_cluster_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]
        self.plates = [p for p in self.pieces if p.part == "3022"]

    def test_platform_plates_exist(self):
        assert len(self.plates) > 0

    def test_platform_plate_y(self):
        for plate in self.plates:
            assert plate.pos[1] == pytest.approx(-_PLATE_H_LDU)

    def test_in_cluster_node_elevated(self):
        inside_nd = next(nd for nd in self.node_data if nd["label"] == "inside")
        inside_brick = next(p for p in self.bricks if np.allclose(p.pos, inside_nd["pos"]))
        assert inside_brick.pos[1] == pytest.approx(-(_PLATE_H_LDU + _BRICK_H_LDU))

    def test_lone_node_at_floor_level(self):
        outside_nd = next(nd for nd in self.node_data if nd["label"] == "outside")
        outside_brick = next(p for p in self.bricks if np.allclose(p.pos, outside_nd["pos"]))
        assert outside_brick.pos[1] == pytest.approx(-_BRICK_H_LDU)

    def test_in_cluster_node_uses_cluster_color(self):
        inside_nd = next(nd for nd in self.node_data if nd["label"] == "inside")
        inside_brick = next(p for p in self.bricks if np.allclose(p.pos, inside_nd["pos"]))
        # cluster_A is first unique cluster → palette index 0 → color 1 (blue)
        assert inside_brick.color == 1

    def test_plates_part_number(self):
        for plate in self.plates:
            assert plate.part == "3022"


# ---------------------------------------------------------------------------
# build_ldr_scene — nested clusters (innermost wins)
# ---------------------------------------------------------------------------

class TestBuildLdrSceneNestedCluster:
    def setup_method(self):
        self.pieces, _, self.node_data = build_ldr_scene(_nested_cluster_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]

    def test_deep_node_uses_innermost_cluster_color(self):
        deep_nd = next(nd for nd in self.node_data if nd["label"] == "deep")
        deep_brick = next(p for p in self.bricks if np.allclose(p.pos, deep_nd["pos"]))
        # cluster_B (size=1) wins over cluster_A (size=2)
        # cluster_B is the first unique innermost cluster → palette[0] = 1 (blue)
        assert deep_brick.color == 1

    def test_deep_node_is_elevated(self):
        # cluster_B (depth=1) sits inside cluster_A (depth=0).
        # Node brick Y = -(depth+1)*PLATE_H - BRICK_H = -(2*8 + 24) = -40
        deep_nd = next(nd for nd in self.node_data if nd["label"] == "deep")
        deep_brick = next(p for p in self.bricks if np.allclose(p.pos, deep_nd["pos"]))
        assert deep_brick.pos[1] == pytest.approx(-(2 * _PLATE_H_LDU + _BRICK_H_LDU))


# ---------------------------------------------------------------------------
# build_ldr_scene — edges / floor arrows
# ---------------------------------------------------------------------------

class TestBuildLdrSceneEdges:
    def test_edge_count_matches_graph(self):
        _, arrows, _ = build_ldr_scene(_lone_graph())
        assert len(arrows) == 1

    def test_arrow_positions_at_floor_y(self):
        _, arrows, _ = build_ldr_scene(_lone_graph())
        for from_pos, to_pos in arrows:
            assert from_pos[1] == pytest.approx(0.0)
            assert to_pos[1] == pytest.approx(0.0)

    def test_arrow_endpoints_differ(self):
        _, arrows, _ = build_ldr_scene(_lone_graph())
        from_pos, to_pos = arrows[0]
        assert not np.allclose(from_pos, to_pos)

    def test_arrow_positions_are_arrays(self):
        _, arrows, _ = build_ldr_scene(_lone_graph())
        for from_pos, to_pos in arrows:
            assert isinstance(from_pos, np.ndarray)
            assert isinstance(to_pos, np.ndarray)

    def test_no_edges_means_no_arrows(self):
        graph = {**_lone_graph(), "edges": []}
        _, arrows, _ = build_ldr_scene(graph)
        assert arrows == []

    def test_edge_to_unknown_gvid_skipped(self):
        graph = {**_lone_graph(), "edges": [{"tail": 0, "head": 99}]}
        _, arrows, _ = build_ldr_scene(graph)
        assert arrows == []


# ---------------------------------------------------------------------------
# build_ldr_scene — node_data fields
# ---------------------------------------------------------------------------

class TestBuildLdrSceneNodeData:
    def setup_method(self):
        _, _, self.node_data = build_ldr_scene(_lone_graph())

    def test_half_w_is_20(self):
        for nd in self.node_data:
            assert nd["half_w"] == 20

    def test_labels_present(self):
        labels = {nd["label"] for nd in self.node_data}
        assert "alpha" in labels
        assert "beta" in labels

    def test_icon_path_key_present(self):
        for nd in self.node_data:
            assert "icon_path" in nd

    def test_pos_is_3d_array(self):
        for nd in self.node_data:
            assert isinstance(nd["pos"], np.ndarray)
            assert nd["pos"].shape == (3,)

    def test_node_without_label_gets_empty_string(self):
        graph = {
            "objects": [{"_gvid": 0, "pos": "0,0", "image": "", "label": ""}],
            "edges": [],
        }
        _, _, node_data = build_ldr_scene(graph)
        assert node_data[0]["label"] == ""

    def test_node_without_image_gets_none(self):
        graph = {
            "objects": [{"_gvid": 0, "pos": "0,0", "image": "", "label": "x"}],
            "edges": [],
        }
        _, _, node_data = build_ldr_scene(graph)
        assert node_data[0]["icon_path"] is None
