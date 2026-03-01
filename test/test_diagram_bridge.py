"""Unit tests for ldr2svg.diagram_bridge."""

from pathlib import Path

import numpy as np
import pytest

from ldr2svg.diagram_bridge import (
    extract_graph,
    build_ldr_scene,
    TileExtent,
    _provider_color,
    _median_nn_dist,
    _PLATE_H_LDU,
    _BRICK_H_LDU,
    _TILE_LDU,
    _NODE_TILE_PART,
    _PLATFORM_TILE_PART,
    _parse_objects,
    _compute_cluster_metadata,
    _layout_positions,
    _compute_platform_extents,
    _first_overlapping_extent,
    _displace_lone_nodes,
    _build_node_pieces,
    _build_platform_pieces,
    _build_cluster_label_data,
    _assemble_piece_groups,
    _build_edge_positions,
)
from ldr2svg.parts import Piece

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


def _overlapping_graph():
    """Lone node placed between two cluster nodes, initially inside the platform.

    cluster_A has nodes at graphviz (0,0) and (160,0).
    After scaling (nn_dist=80 → scale=1.0) they snap to LDraw X=0 and X=160.
    Lone node at (80,0) snaps to X=80 — right in the middle of the platform
    which extends from X=-40 to X=200 with pad=40.
    """
    return {
        "objects": [
            {"_gvid": 0, "nodes": [1, 2], "name": "cluster_A"},
            {"_gvid": 1, "pos": "0,0",   "image": "/res/k8s/pod.png", "label": "left"},
            {"_gvid": 2, "pos": "160,0", "image": "/res/k8s/pod.png", "label": "right"},
            {"_gvid": 3, "pos": "80,0",  "image": "/res/gcp/lb.png",  "label": "lone"},
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
    def test_returns_five_values(self):
        assert len(build_ldr_scene(_lone_graph())) == 5

    def test_pieces_is_list(self):
        pieces, _, _, _, _ = build_ldr_scene(_lone_graph())
        assert isinstance(pieces, list)

    def test_arrows_is_list(self):
        _, arrows, _, _, _ = build_ldr_scene(_lone_graph())
        assert isinstance(arrows, list)

    def test_node_data_is_list(self):
        _, _, node_data, _, _ = build_ldr_scene(_lone_graph())
        assert isinstance(node_data, list)

    def test_node_data_count_matches_nodes(self):
        _, _, node_data, _, _ = build_ldr_scene(_lone_graph())
        assert len(node_data) == 2

    def test_piece_groups_is_list(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_lone_graph())
        assert isinstance(piece_groups, list)

    def test_piece_groups_entries_are_tuples(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_lone_graph())
        for name, grp in piece_groups:
            assert isinstance(name, str)
            assert isinstance(grp, list)

    def test_empty_graph_returns_empty(self):
        pieces, arrows, node_data, piece_groups, cluster_data = build_ldr_scene({"objects": [], "edges": []})
        assert pieces == []
        assert arrows == []
        assert node_data == []
        assert piece_groups == []
        assert cluster_data == []


# ---------------------------------------------------------------------------
# build_ldr_scene — lone nodes (no cluster)
# ---------------------------------------------------------------------------

class TestBuildLdrSceneLoneNodes:
    def setup_method(self):
        self.pieces, self.arrows, self.node_data, self.piece_groups, _ = build_ldr_scene(_lone_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]

    def test_two_bricks_created(self):
        assert len(self.bricks) == 2

    def test_no_platform_plates(self):
        assert all(p.part != _PLATFORM_TILE_PART for p in self.pieces)

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
        self.pieces, self.arrows, self.node_data, self.piece_groups, _ = build_ldr_scene(_cluster_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]
        self.plates = [p for p in self.pieces if p.part == _PLATFORM_TILE_PART]

    def test_platform_plates_exist(self):
        assert len(self.plates) > 0

    def test_platform_plate_y(self):
        for plate in self.plates:
            assert plate.pos[1] == pytest.approx(-_PLATE_H_LDU)

    def test_in_cluster_node_elevated(self):
        inside_nd = next(nd for nd in self.node_data if nd["label"] == "inside")
        inside_brick = next(p for p in self.bricks if np.isclose(p.pos[0], inside_nd["pos"][0]) and np.isclose(p.pos[2], inside_nd["pos"][2]))
        assert inside_brick.pos[1] == pytest.approx(-(_PLATE_H_LDU + _BRICK_H_LDU))

    def test_lone_node_at_floor_level(self):
        outside_nd = next(nd for nd in self.node_data if nd["label"] == "outside")
        outside_brick = next(p for p in self.bricks if np.isclose(p.pos[0], outside_nd["pos"][0]) and np.isclose(p.pos[2], outside_nd["pos"][2]))
        assert outside_brick.pos[1] == pytest.approx(-_BRICK_H_LDU)

    def test_in_cluster_node_uses_cluster_color(self):
        inside_nd = next(nd for nd in self.node_data if nd["label"] == "inside")
        inside_brick = next(p for p in self.bricks if np.isclose(p.pos[0], inside_nd["pos"][0]) and np.isclose(p.pos[2], inside_nd["pos"][2]))
        # cluster_A is first unique cluster → palette index 0 → color 1 (blue)
        assert inside_brick.color == 1

    def test_plates_part_number(self):
        for plate in self.plates:
            assert plate.part == _PLATFORM_TILE_PART


# ---------------------------------------------------------------------------
# build_ldr_scene — nested clusters (innermost wins)
# ---------------------------------------------------------------------------

class TestBuildLdrSceneNestedCluster:
    def setup_method(self):
        self.pieces, _, self.node_data, _, _ = build_ldr_scene(_nested_cluster_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]

    def test_deep_node_uses_innermost_cluster_color(self):
        deep_nd = next(nd for nd in self.node_data if nd["label"] == "deep")
        deep_brick = next(p for p in self.bricks if np.isclose(p.pos[0], deep_nd["pos"][0]) and np.isclose(p.pos[2], deep_nd["pos"][2]))
        # cluster_B (size=1) wins over cluster_A (size=2)
        # cluster_B is the first unique innermost cluster → palette[0] = 1 (blue)
        assert deep_brick.color == 1

    def test_deep_node_is_elevated(self):
        # cluster_B (depth=1) sits inside cluster_A (depth=0).
        # Node brick Y = -(depth+1)*PLATE_H - BRICK_H = -(2*8 + 24) = -40
        deep_nd = next(nd for nd in self.node_data if nd["label"] == "deep")
        deep_brick = next(p for p in self.bricks if np.isclose(p.pos[0], deep_nd["pos"][0]) and np.isclose(p.pos[2], deep_nd["pos"][2]))
        assert deep_brick.pos[1] == pytest.approx(-(2 * _PLATE_H_LDU + _BRICK_H_LDU))


# ---------------------------------------------------------------------------
# build_ldr_scene — lone node displacement
# ---------------------------------------------------------------------------

class TestLoneNodeDisplacement:
    """Lone nodes that overlap a cluster platform must be pushed outside."""

    def setup_method(self):
        self.pieces, _, self.node_data, _, _ = build_ldr_scene(_overlapping_graph())
        self.bricks = [p for p in self.pieces if p.part == "3003"]

    def _platform_box(self):
        """Cluster_A platform extent: X=[-40,200], Z=[-40,40] (pad=40)."""
        return (-40, 200, -40, 40)

    def test_lone_node_not_inside_platform(self):
        lone_nd = next(nd for nd in self.node_data if nd["label"] == "lone")
        lx = lone_nd["pos"][0]
        lz = lone_nd["pos"][2]
        px0, px1, pz0, pz1 = self._platform_box()
        overlaps_x = (lx - 20) < px1 and (lx + 20) > px0
        overlaps_z = (lz - 20) < pz1 and (lz + 20) > pz0
        assert not (overlaps_x and overlaps_z)

    def test_lone_node_on_20ldu_grid(self):
        lone_nd = next(nd for nd in self.node_data if nd["label"] == "lone")
        assert lone_nd["pos"][0] % 20 == pytest.approx(0, abs=1e-6)
        assert lone_nd["pos"][2] % 20 == pytest.approx(0, abs=1e-6)

    def test_cluster_nodes_unchanged(self):
        """Cluster node positions should not be affected by the displacement step."""
        left_nd  = next(nd for nd in self.node_data if nd["label"] == "left")
        right_nd = next(nd for nd in self.node_data if nd["label"] == "right")
        assert left_nd["pos"][0]  == pytest.approx(0.0)
        assert right_nd["pos"][0] == pytest.approx(160.0)

    def test_lone_node_brick_exists(self):
        lone_nd = next(nd for nd in self.node_data if nd["label"] == "lone")
        # node_data pos is the tile top face; match brick by XZ only
        match = [p for p in self.bricks
                 if np.isclose(p.pos[0], lone_nd["pos"][0])
                 and np.isclose(p.pos[2], lone_nd["pos"][2])]
        assert len(match) == 1


# ---------------------------------------------------------------------------
# build_ldr_scene — edges / floor arrows
# ---------------------------------------------------------------------------

class TestBuildLdrSceneEdges:
    def test_edge_count_matches_graph(self):
        _, arrows, _, _, _ = build_ldr_scene(_lone_graph())
        assert len(arrows) == 1

    def test_arrow_positions_at_mid_y(self):
        _, arrows, _, _, _ = build_ldr_scene(_lone_graph())
        # Lone nodes → node_y = -BRICK_H_LDU; mid = node_y + BRICK_H_LDU/2
        expected_mid_y = -_BRICK_H_LDU + _BRICK_H_LDU / 2
        for from_pos, to_pos in arrows:
            assert from_pos[1] == pytest.approx(expected_mid_y)
            assert to_pos[1] == pytest.approx(expected_mid_y)

    def test_arrow_endpoints_differ(self):
        _, arrows, _, _, _ = build_ldr_scene(_lone_graph())
        from_pos, to_pos = arrows[0]
        assert not np.allclose(from_pos, to_pos)

    def test_arrow_positions_are_arrays(self):
        _, arrows, _, _, _ = build_ldr_scene(_lone_graph())
        for from_pos, to_pos in arrows:
            assert isinstance(from_pos, np.ndarray)
            assert isinstance(to_pos, np.ndarray)

    def test_no_edges_means_no_arrows(self):
        graph = {**_lone_graph(), "edges": []}
        _, arrows, _, _, _ = build_ldr_scene(graph)
        assert arrows == []

    def test_edge_to_unknown_gvid_skipped(self):
        graph = {**_lone_graph(), "edges": [{"tail": 0, "head": 99}]}
        _, arrows, _, _, _ = build_ldr_scene(graph)
        assert arrows == []


# ---------------------------------------------------------------------------
# build_ldr_scene — node_data fields
# ---------------------------------------------------------------------------

class TestBuildLdrSceneNodeData:
    def setup_method(self):
        _, _, self.node_data, _, _ = build_ldr_scene(_lone_graph())

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
        _, _, node_data, _, _ = build_ldr_scene(graph)
        assert node_data[0]["label"] == ""

    def test_node_without_image_gets_none(self):
        graph = {
            "objects": [{"_gvid": 0, "pos": "0,0", "image": "", "label": "x"}],
            "edges": [],
        }
        _, _, node_data, _, _ = build_ldr_scene(graph)
        assert node_data[0]["icon_path"] is None


# ---------------------------------------------------------------------------
# build_ldr_scene — piece_groups structure
# ---------------------------------------------------------------------------

class TestPieceGroups:
    def test_lone_graph_has_lone_group(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_lone_graph())
        names = [name for name, _ in piece_groups]
        assert "lone" in names

    def test_lone_group_contains_brick_and_tile_per_node(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_lone_graph())
        lone = next(grp for name, grp in piece_groups if name == "lone")
        # 2 nodes × (1 brick + 1 tile) = 4 pieces
        assert len(lone) == 4

    def test_cluster_graph_has_cluster_group(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_cluster_graph())
        names = [name for name, _ in piece_groups]
        assert "cluster_A" in names

    def test_cluster_group_contains_platform_tiles(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_cluster_graph())
        cluster_grp = next(grp for name, grp in piece_groups if name == "cluster_A")
        tiles = [p for p in cluster_grp if p.part == _PLATFORM_TILE_PART]
        assert len(tiles) > 0

    def test_cluster_group_contains_node_brick_and_tile(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_cluster_graph())
        cluster_grp = next(grp for name, grp in piece_groups if name == "cluster_A")
        assert sum(1 for p in cluster_grp if p.part == "3003") == 1
        assert sum(1 for p in cluster_grp if p.part == _NODE_TILE_PART) == 1

    def test_cluster_group_before_lone_group(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_cluster_graph())
        names = [name for name, _ in piece_groups]
        assert names.index("cluster_A") < names.index("lone")

    def test_outer_cluster_before_inner_cluster(self):
        _, _, _, piece_groups, _ = build_ldr_scene(_nested_cluster_graph())
        names = [name for name, _ in piece_groups]
        assert names.index("cluster_A") < names.index("cluster_B")

    def test_all_pieces_covered_by_groups(self):
        pieces, _, _, piece_groups, _ = build_ldr_scene(_cluster_graph())
        grouped = {id(p) for _, grp in piece_groups for p in grp}
        assert all(id(p) in grouped for p in pieces)


# ---------------------------------------------------------------------------
# Step: _parse_objects
# ---------------------------------------------------------------------------

class TestParseObjects:
    def test_splits_nodes_and_clusters(self):
        graph = {"objects": [
            {"_gvid": 0, "nodes": [1], "name": "cluster_A"},
            {"_gvid": 1, "pos": "0,0", "label": "x"},
        ]}
        node_objs, cluster_objs = _parse_objects(graph)
        assert len(node_objs) == 1
        assert len(cluster_objs) == 1

    def test_empty_objects(self):
        node_objs, cluster_objs = _parse_objects({"objects": []})
        assert node_objs == []
        assert cluster_objs == []

    def test_only_nodes(self):
        node_objs, cluster_objs = _parse_objects({"objects": [{"_gvid": 0, "pos": "0,0"}]})
        assert len(node_objs) == 1
        assert cluster_objs == []

    def test_only_clusters(self):
        node_objs, cluster_objs = _parse_objects({"objects": [{"_gvid": 0, "nodes": [1], "name": "c"}]})
        assert node_objs == []
        assert len(cluster_objs) == 1

    def test_missing_objects_key(self):
        node_objs, cluster_objs = _parse_objects({})
        assert node_objs == []
        assert cluster_objs == []


# ---------------------------------------------------------------------------
# Step: _compute_cluster_metadata
# ---------------------------------------------------------------------------

class TestComputeClusterMetadata:
    def _lone_inputs(self):
        return [{"_gvid": 0, "pos": "0,0"}, {"_gvid": 1, "pos": "155,0"}], []

    def _cluster_inputs(self):
        node_objs = [{"_gvid": 1, "pos": "0,0"}, {"_gvid": 2, "pos": "155,0"}]
        cluster_objs = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        return node_objs, cluster_objs

    def _nested_inputs(self):
        node_objs = [{"_gvid": 2, "pos": "0,0"}, {"_gvid": 3, "pos": "155,0"}]
        cluster_objs = [
            {"_gvid": 0, "nodes": [2, 3], "name": "cluster_A"},
            {"_gvid": 1, "nodes": [2],    "name": "cluster_B"},
        ]
        return node_objs, cluster_objs

    def test_lone_nodes_not_in_node_cluster(self):
        node_cluster, _, _, _ = _compute_cluster_metadata(*self._lone_inputs())
        assert node_cluster == {}

    def test_node_mapped_to_cluster(self):
        node_cluster, _, _, _ = _compute_cluster_metadata(*self._cluster_inputs())
        assert node_cluster[1] == "cluster_A"

    def test_lone_node_not_mapped(self):
        node_cluster, _, _, _ = _compute_cluster_metadata(*self._cluster_inputs())
        assert 2 not in node_cluster

    def test_node_mapped_to_innermost_cluster(self):
        # gvid 2 is in cluster_A and cluster_B; cluster_B (size=1) wins
        node_cluster, _, _, _ = _compute_cluster_metadata(*self._nested_inputs())
        assert node_cluster[2] == "cluster_B"

    def test_cluster_color_assigned(self):
        _, cluster_color, _, _ = _compute_cluster_metadata(*self._cluster_inputs())
        assert "cluster_A" in cluster_color

    def test_two_clusters_get_different_colors(self):
        _, cluster_color, _, _ = _compute_cluster_metadata(*self._nested_inputs())
        assert cluster_color["cluster_A"] != cluster_color["cluster_B"]

    def test_outermost_cluster_depth_zero(self):
        _, _, cluster_depth, _ = _compute_cluster_metadata(*self._nested_inputs())
        assert cluster_depth["cluster_A"] == 0

    def test_inner_cluster_depth_one(self):
        _, _, cluster_depth, _ = _compute_cluster_metadata(*self._nested_inputs())
        assert cluster_depth["cluster_B"] == 1

    def test_root_cluster_parent_is_none(self):
        _, _, _, cluster_parent = _compute_cluster_metadata(*self._cluster_inputs())
        assert cluster_parent["cluster_A"] is None

    def test_child_cluster_parent_is_outer(self):
        _, _, _, cluster_parent = _compute_cluster_metadata(*self._nested_inputs())
        assert cluster_parent["cluster_B"] == "cluster_A"


# ---------------------------------------------------------------------------
# Step: _layout_positions
# ---------------------------------------------------------------------------

class TestLayoutPositions:
    def test_returns_gvid_keys(self):
        node_objs = [{"_gvid": 5, "pos": "0,0"}, {"_gvid": 7, "pos": "80,0"}]
        result = _layout_positions(node_objs)
        assert set(result.keys()) == {5, 7}

    def test_positions_snapped_to_tile_grid(self):
        node_objs = [{"_gvid": 0, "pos": "0,0"}, {"_gvid": 1, "pos": "80,0"}]
        result = _layout_positions(node_objs)
        for ldx, ldz in result.values():
            assert ldx % _TILE_LDU == 0
            assert ldz % _TILE_LDU == 0

    def test_origin_stays_at_zero(self):
        node_objs = [{"_gvid": 0, "pos": "0,0"}, {"_gvid": 1, "pos": "155,0"}]
        result = _layout_positions(node_objs)
        assert result[0] == (0, 0)

    def test_scale_separates_nodes_by_80ldu(self):
        # Two nodes 155 gv-points apart → scale=80/155 → separation = 80 LDU
        node_objs = [{"_gvid": 0, "pos": "0,0"}, {"_gvid": 1, "pos": "155,0"}]
        result = _layout_positions(node_objs)
        assert result[1][0] == 80

    def test_graphviz_y_negated_to_ldraw_z(self):
        # Positive graphviz Y (up) → negative LDraw Z (away from viewer)
        node_objs = [{"_gvid": 0, "pos": "0,0"}, {"_gvid": 1, "pos": "0,155"}]
        result = _layout_positions(node_objs)
        assert result[1][1] < 0

    def test_single_node_snapped_to_grid(self):
        # With one node, nn_dist falls back to 1.0 → scale=80; still snaps to tile grid
        node_objs = [{"_gvid": 0, "pos": "42,17"}]
        result = _layout_positions(node_objs)
        ldx, ldz = result[0]
        assert ldx % _TILE_LDU == 0
        assert ldz % _TILE_LDU == 0


# ---------------------------------------------------------------------------
# Step: _compute_platform_extents
# ---------------------------------------------------------------------------

class TestComputePlatformExtents:
    def _single(self):
        cluster_objs  = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        gvid_to_ld    = {1: (0, 0)}
        cluster_depth  = {"cluster_A": 0}
        cluster_parent = {"cluster_A": None}
        return cluster_objs, gvid_to_ld, cluster_depth, cluster_parent

    def test_cluster_in_result(self):
        assert "cluster_A" in _compute_platform_extents(*self._single())

    def test_extent_covers_member_node(self):
        ext = _compute_platform_extents(*self._single())["cluster_A"]
        assert ext.x0 <= 0 <= ext.x1
        assert ext.z0 <= 0 <= ext.z1

    def test_extent_has_padding(self):
        # depth=0, max_depth=0 → pad = tile*(0−0+2) = 40 LDU
        ext = _compute_platform_extents(*self._single())["cluster_A"]
        assert ext.x0 <= -40
        assert ext.x1 >= 40

    def test_extent_snapped_to_tile_grid(self):
        ext = _compute_platform_extents(*self._single())["cluster_A"]
        for v in (ext.x0, ext.x1, ext.z0, ext.z1):
            assert v % _TILE_LDU == 0

    def test_cluster_with_no_positioned_members_omitted(self):
        cluster_objs  = [{"_gvid": 0, "nodes": [99], "name": "cluster_A"}]
        result = _compute_platform_extents(
            cluster_objs, {}, {"cluster_A": 0}, {"cluster_A": None}
        )
        assert "cluster_A" not in result

    def test_sibling_extents_do_not_overlap(self):
        # Two sibling clusters, nodes far apart; their platforms must not overlap
        cluster_objs = [
            {"_gvid": 0, "nodes": [2], "name": "cluster_A"},
            {"_gvid": 1, "nodes": [3], "name": "cluster_B"},
        ]
        gvid_to_ld    = {2: (0, 0), 3: (160, 0)}
        cluster_depth  = {"cluster_A": 0, "cluster_B": 0}
        cluster_parent = {"cluster_A": None, "cluster_B": None}
        result = _compute_platform_extents(cluster_objs, gvid_to_ld, cluster_depth, cluster_parent)
        # A is left of B: A's right edge must not exceed B's left edge
        assert result["cluster_A"].x1 <= result["cluster_B"].x0

    def test_child_extent_within_parent(self):
        cluster_objs = [
            {"_gvid": 0, "nodes": [1, 2], "name": "cluster_A"},
            {"_gvid": 1, "nodes": [2],    "name": "cluster_B"},
        ]
        gvid_to_ld    = {1: (0, 0), 2: (80, 0)}
        cluster_depth  = {"cluster_A": 0, "cluster_B": 1}
        cluster_parent = {"cluster_A": None, "cluster_B": "cluster_A"}
        result = _compute_platform_extents(cluster_objs, gvid_to_ld, cluster_depth, cluster_parent)
        a = result["cluster_A"]
        b = result["cluster_B"]
        assert b.x0 >= a.x0 and b.x1 <= a.x1
        assert b.z0 >= a.z0 and b.z1 <= a.z1


# ---------------------------------------------------------------------------
# Step: _displace_lone_nodes
# ---------------------------------------------------------------------------

class TestDisplaceLoneNodes:
    def test_lone_node_inside_platform_is_moved(self):
        gvid_to_ld = {0: (0, 0)}
        _displace_lone_nodes(gvid_to_ld, {}, {"c": TileExtent(-40, 40, -40, 40)})
        assert gvid_to_ld[0] != (0, 0)

    def test_displaced_node_no_longer_overlaps(self):
        gvid_to_ld = {0: (0, 0)}
        _displace_lone_nodes(gvid_to_ld, {}, {"c": TileExtent(-40, 40, -40, 40)})
        ldx, ldz = gvid_to_ld[0]
        bx0, bx1 = ldx - _TILE_LDU, ldx + _TILE_LDU
        bz0, bz1 = ldz - _TILE_LDU, ldz + _TILE_LDU
        assert not (bx0 < 40 and bx1 > -40 and bz0 < 40 and bz1 > -40)

    def test_clustered_node_not_displaced(self):
        gvid_to_ld = {0: (0, 0)}
        _displace_lone_nodes(gvid_to_ld, {0: "cluster_A"}, {"cluster_A": TileExtent(-40, 40, -40, 40)})
        assert gvid_to_ld[0] == (0, 0)

    def test_non_overlapping_lone_node_unchanged(self):
        gvid_to_ld = {0: (200, 0)}
        _displace_lone_nodes(gvid_to_ld, {}, {"c": TileExtent(-40, 40, -40, 40)})
        assert gvid_to_ld[0] == (200, 0)

    def test_result_snapped_to_tile_grid(self):
        gvid_to_ld = {0: (0, 0)}
        _displace_lone_nodes(gvid_to_ld, {}, {"c": TileExtent(-40, 40, -40, 40)})
        ldx, ldz = gvid_to_ld[0]
        assert ldx % _TILE_LDU == 0
        assert ldz % _TILE_LDU == 0


# ---------------------------------------------------------------------------
# Step: _build_node_pieces
# ---------------------------------------------------------------------------

class TestBuildNodePieces:
    def _inputs(self):
        node_objs = [
            {"_gvid": 0, "pos": "0,0",  "image": "/k8s/pod.png", "label": "lone"},
            {"_gvid": 1, "pos": "80,0", "image": "",             "label": "clustered"},
        ]
        gvid_to_ld    = {0: (0, 0), 1: (80, 0)}
        node_cluster  = {1: "cluster_A"}
        cluster_color = {"cluster_A": 1}
        cluster_depth = {"cluster_A": 0}
        return node_objs, gvid_to_ld, node_cluster, cluster_color, cluster_depth

    def test_pieces_count_equals_two_per_node(self):
        # Each node produces a 3003 brick + a 3068b tile = 2 per node
        pieces, _, _, _, _ = _build_node_pieces(*self._inputs())
        assert len(pieces) == 4

    def test_node_data_count_equals_nodes(self):
        _, node_data, _, _, _ = _build_node_pieces(*self._inputs())
        assert len(node_data) == 2

    def test_each_node_has_brick_and_tile(self):
        pieces, _, _, _, _ = _build_node_pieces(*self._inputs())
        assert sum(1 for p in pieces if p.part == "3003") == 2
        assert sum(1 for p in pieces if p.part == _NODE_TILE_PART) == 2

    def test_lone_node_y(self):
        _, _, _, lone_bricks, _ = _build_node_pieces(*self._inputs())
        brick = next(p for p in lone_bricks if p.part == "3003")
        assert brick.pos[1] == pytest.approx(-_BRICK_H_LDU)

    def test_cluster_node_elevated_by_platform(self):
        # depth=0 → node_y = -(1*PLATE_H + BRICK_H)
        _, _, cluster_bricks, _, _ = _build_node_pieces(*self._inputs())
        brick = next(p for p in cluster_bricks["cluster_A"] if p.part == "3003")
        assert brick.pos[1] == pytest.approx(-_PLATE_H_LDU - _BRICK_H_LDU)

    def test_cluster_node_bricks_keyed_by_cluster(self):
        # 1 brick + 1 tile per node in the cluster
        _, _, cluster_bricks, _, _ = _build_node_pieces(*self._inputs())
        assert "cluster_A" in cluster_bricks
        assert len(cluster_bricks["cluster_A"]) == 2

    def test_lone_node_bricks_list(self):
        # 1 brick + 1 tile per lone node
        _, _, _, lone_bricks, _ = _build_node_pieces(*self._inputs())
        assert len(lone_bricks) == 2

    def test_lone_node_color_from_provider(self):
        _, _, _, lone_bricks, _ = _build_node_pieces(*self._inputs())
        assert lone_bricks[0].color == 1  # k8s → blue

    def test_mid_y_lone_node(self):
        _, _, _, _, gvid_to_mid_y = _build_node_pieces(*self._inputs())
        assert gvid_to_mid_y[0] == pytest.approx(-_BRICK_H_LDU + _BRICK_H_LDU / 2)

    def test_mid_y_cluster_node(self):
        # depth=0 → node_y = -(PLATE_H + BRICK_H); mid = node_y + BRICK_H/2
        _, _, _, _, gvid_to_mid_y = _build_node_pieces(*self._inputs())
        expected = -(_PLATE_H_LDU + _BRICK_H_LDU) + _BRICK_H_LDU / 2
        assert gvid_to_mid_y[1] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Step: _build_platform_pieces
# ---------------------------------------------------------------------------

class TestBuildPlatformPieces:
    def _inputs(self):
        cluster_objs        = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        cluster_tile_extent = {"cluster_A": TileExtent(0, 40, 0, 40)}
        cluster_depth       = {"cluster_A": 0}
        cluster_color       = {"cluster_A": 1}
        return cluster_objs, cluster_tile_extent, cluster_depth, cluster_color

    def test_pieces_are_1x1_plates(self):
        pieces, _ = _build_platform_pieces(*self._inputs())
        assert all(p.part == _PLATFORM_TILE_PART for p in pieces)

    def test_tile_count_covers_extent(self):
        # extent (0,40, 0,40) → 2×2 tiles of 20 LDU each → 4 tiles
        pieces, _ = _build_platform_pieces(*self._inputs())
        assert len(pieces) == 4

    def test_plate_y_depth_0(self):
        pieces, _ = _build_platform_pieces(*self._inputs())
        for p in pieces:
            assert p.pos[1] == pytest.approx(-_PLATE_H_LDU)

    def test_plate_y_depth_1(self):
        cluster_objs        = [{"_gvid": 0, "nodes": [1], "name": "cluster_B"}]
        cluster_tile_extent = {"cluster_B": TileExtent(0, 20, 0, 20)}
        cluster_depth       = {"cluster_B": 1}
        cluster_color       = {"cluster_B": 4}
        pieces, _ = _build_platform_pieces(cluster_objs, cluster_tile_extent, cluster_depth, cluster_color)
        for p in pieces:
            assert p.pos[1] == pytest.approx(-2 * _PLATE_H_LDU)

    def test_cluster_platform_tiles_populated(self):
        _, cluster_platform_tiles = _build_platform_pieces(*self._inputs())
        assert "cluster_A" in cluster_platform_tiles
        assert len(cluster_platform_tiles["cluster_A"]) == 4

    def test_cluster_missing_from_extent_skipped(self):
        cluster_objs = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        pieces, _ = _build_platform_pieces(cluster_objs, {}, {"cluster_A": 0}, {})
        assert pieces == []


# ---------------------------------------------------------------------------
# Step: _build_cluster_label_data
# ---------------------------------------------------------------------------

class TestBuildClusterLabelData:
    def _inputs(self):
        cluster_objs        = [{"_gvid": 0, "nodes": [1], "name": "cluster_A", "label": "MyCluster"}]
        cluster_tile_extent = {"cluster_A": TileExtent(0, 60, 0, 40)}
        cluster_depth       = {"cluster_A": 0}
        return cluster_objs, cluster_tile_extent, cluster_depth

    def test_returns_one_entry_per_cluster(self):
        result = _build_cluster_label_data(*self._inputs())
        assert len(result) == 1

    def test_label_uses_display_name(self):
        result = _build_cluster_label_data(*self._inputs())
        assert result[0]["label"] == "MyCluster"

    def test_label_falls_back_to_name(self):
        cluster_objs = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        result = _build_cluster_label_data(
            cluster_objs, {"cluster_A": TileExtent(0, 40, 0, 40)}, {"cluster_A": 0}
        )
        assert result[0]["label"] == "cluster_A"

    def test_pos_is_3d_array(self):
        result = _build_cluster_label_data(*self._inputs())
        assert isinstance(result[0]["pos"], np.ndarray)
        assert result[0]["pos"].shape == (3,)

    def test_pos_x_is_center_of_extent(self):
        result = _build_cluster_label_data(*self._inputs())
        # TileExtent(0, 60, ...) → center X = 30
        assert result[0]["pos"][0] == pytest.approx(30.0)

    def test_pos_z_is_front_edge(self):
        result = _build_cluster_label_data(*self._inputs())
        # TileExtent(..., 0, 40) → front Z = z1 = 40
        assert result[0]["pos"][2] == pytest.approx(40.0)

    def test_pos_y_is_platform_surface(self):
        result = _build_cluster_label_data(*self._inputs())
        # depth=0 → platform_y = -(0+1)*8 = -8
        assert result[0]["pos"][1] == pytest.approx(-_PLATE_H_LDU)

    def test_cluster_missing_from_extent_skipped(self):
        cluster_objs = [{"_gvid": 0, "nodes": [1], "name": "cluster_A", "label": "A"}]
        result = _build_cluster_label_data(cluster_objs, {}, {"cluster_A": 0})
        assert result == []


# ---------------------------------------------------------------------------
# Step: _assemble_piece_groups
# ---------------------------------------------------------------------------

def _make_piece(part: str = _PLATFORM_TILE_PART) -> Piece:
    return Piece(part=part, color=1, pos=np.zeros(3), rot=np.eye(3))


class TestAssemblePieceGroups:
    def test_cluster_group_present(self):
        cluster_objs  = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        cluster_depth = {"cluster_A": 0}
        groups = _assemble_piece_groups(
            cluster_objs, cluster_depth, {"cluster_A": [_make_piece("3003")]}, {}, []
        )
        assert any(name == "cluster_A" for name, _ in groups)

    def test_lone_group_appended_last(self):
        cluster_objs  = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        cluster_depth = {"cluster_A": 0}
        groups = _assemble_piece_groups(
            cluster_objs, cluster_depth,
            {"cluster_A": [_make_piece("3003")]}, {},
            [_make_piece("3003")],
        )
        assert groups[-1][0] == "lone"

    def test_platform_tiles_before_node_brick_within_group(self):
        cluster_objs  = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        cluster_depth = {"cluster_A": 0}
        tile  = _make_piece(_PLATFORM_TILE_PART)
        brick = _make_piece("3003")
        groups = _assemble_piece_groups(
            cluster_objs, cluster_depth,
            {"cluster_A": [brick]}, {"cluster_A": [tile]}, []
        )
        _, grp = groups[0]
        assert grp[0].part == _PLATFORM_TILE_PART
        assert grp[-1].part == "3003"

    def test_outer_cluster_before_inner(self):
        cluster_objs = [
            {"_gvid": 0, "nodes": [1, 2], "name": "cluster_A"},
            {"_gvid": 1, "nodes": [2],    "name": "cluster_B"},
        ]
        cluster_depth = {"cluster_A": 0, "cluster_B": 1}
        groups = _assemble_piece_groups(
            cluster_objs, cluster_depth,
            {"cluster_A": [_make_piece("3003")], "cluster_B": [_make_piece("3003")]},
            {}, [],
        )
        names = [n for n, _ in groups]
        assert names.index("cluster_A") < names.index("cluster_B")

    def test_empty_cluster_omitted(self):
        cluster_objs  = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        cluster_depth = {"cluster_A": 0}
        groups = _assemble_piece_groups(cluster_objs, cluster_depth, {}, {}, [])
        assert groups == []

    def test_no_lone_group_when_no_lone_bricks(self):

        cluster_objs  = [{"_gvid": 0, "nodes": [1], "name": "cluster_A"}]
        cluster_depth = {"cluster_A": 0}
        groups = _assemble_piece_groups(
            cluster_objs, cluster_depth, {"cluster_A": [_make_piece("3003")]}, {}, []
        )
        assert all(name != "lone" for name, _ in groups)


# ---------------------------------------------------------------------------
# Step: _build_edge_positions
# ---------------------------------------------------------------------------

class TestBuildEdgePositions:
    _mid_y = {0: -12.0, 1: -12.0}

    def test_known_edge_produces_pair(self):
        result = _build_edge_positions([{"tail": 0, "head": 1}], {0: (0, 0), 1: (80, 0)}, self._mid_y)
        assert len(result) == 1

    def test_positions_are_numpy_arrays(self):
        from_pos, to_pos = _build_edge_positions(
            [{"tail": 0, "head": 1}], {0: (0, 0), 1: (80, 0)}, self._mid_y
        )[0]
        assert isinstance(from_pos, np.ndarray)
        assert isinstance(to_pos, np.ndarray)

    def test_positions_at_mid_y(self):
        from_pos, to_pos = _build_edge_positions(
            [{"tail": 0, "head": 1}], {0: (0, 0), 1: (80, 0)}, self._mid_y
        )[0]
        assert from_pos[1] == pytest.approx(-12.0)
        assert to_pos[1] == pytest.approx(-12.0)

    def test_xz_offset_toward_dest(self):
        # Pure X direction: tail=(0,0), head=(80,0) → offset by _TILE_LDU=20
        from_pos, to_pos = _build_edge_positions(
            [{"tail": 0, "head": 1}], {0: (0, 0), 1: (80, 0)}, self._mid_y
        )[0]
        assert from_pos[0] == pytest.approx(_TILE_LDU)
        assert to_pos[0] == pytest.approx(80 - _TILE_LDU)
        assert from_pos[2] == pytest.approx(0.0)
        assert to_pos[2] == pytest.approx(0.0)

    def test_unknown_tail_skipped(self):
        assert _build_edge_positions([{"tail": 99, "head": 1}], {0: (0, 0), 1: (80, 0)}, self._mid_y) == []

    def test_unknown_head_skipped(self):
        assert _build_edge_positions([{"tail": 0, "head": 99}], {0: (0, 0)}, {0: -12.0}) == []

    def test_empty_edges(self):
        assert _build_edge_positions([], {0: (0, 0)}, {0: -12.0}) == []


# ---------------------------------------------------------------------------
# Step: _first_overlapping_extent
# ---------------------------------------------------------------------------

class TestFirstOverlappingExtent:
    def test_no_extents_returns_none(self):
        assert _first_overlapping_extent(0, 0, []) is None

    def test_non_overlapping_returns_none(self):
        # brick at (200,0), platform at (-40,40,-40,40) — far apart
        assert _first_overlapping_extent(200, 0, [TileExtent(-40, 40, -40, 40)]) is None

    def test_overlapping_returns_extent(self):
        ext = TileExtent(-40, 40, -40, 40)
        assert _first_overlapping_extent(0, 0, [ext]) == ext

    def test_just_touching_not_overlapping(self):
        # brick right edge at ldx+TILE=40, platform left edge at 40 → strict inequality fails
        assert _first_overlapping_extent(20, 0, [TileExtent(40, 80, -40, 40)]) is None

    def test_returns_first_of_multiple(self):
        ext1 = TileExtent(-40, 40, -40, 40)
        ext2 = TileExtent(-60, 60, -60, 60)
        assert _first_overlapping_extent(0, 0, [ext1, ext2]) is ext1

    def test_z_axis_overlap_detected(self):
        # brick at (200, 0) overlaps z-wise with platform (180, 220, -40, 40)
        ext = TileExtent(180, 220, -40, 40)
        assert _first_overlapping_extent(200, 0, [ext]) == ext
