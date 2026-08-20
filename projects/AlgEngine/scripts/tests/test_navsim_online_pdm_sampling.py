import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


DATASET_FILE = (
    Path(__file__).resolve().parents[2]
    / "mmdet3d_plugin/datasets/navsim_openscene_nuplan.py"
)
DETECTOR_FILE = (
    Path(__file__).resolve().parents[2]
    / "mmdet3d_plugin/navformer/detectors/navformer.py"
)
REWARD_FILE = (
    Path(__file__).resolve().parents[2]
    / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
)


class _MultiMetricIndex:
    NO_COLLISION = 0
    DRIVABLE_AREA = 1


class _WeightedMetricIndex:
    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2
    DRIVING_DIRECTION = 3


def _load_pairwise_official_scores():
    tree = ast.parse(REWARD_FILE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pairwise_official_scores"
    )
    namespace = {
        "np": np,
        "PDMScorer": object,
        "Tuple": tuple,
        "MultiMetricIndex": _MultiMetricIndex,
        "WeightedMetricIndex": _WeightedMetricIndex,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), REWARD_FILE, "exec"),
        namespace,
    )
    return namespace["pairwise_official_scores"]


def _load_online_scoring_method():
    tree = ast.parse(DATASET_FILE.read_text())
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NavSimOpenSceneE2E"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_online_pdm_scores"
    )
    namespace = {
        "np": np,
        "logger": SimpleNamespace(info=lambda *_: None, warning=lambda *_: None),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), DATASET_FILE, "exec"), namespace)
    return namespace["_compute_online_pdm_scores"]


def _load_dataset_method(method_name):
    tree = ast.parse(DATASET_FILE.read_text())
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NavSimOpenSceneE2E"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), DATASET_FILE, "exec"), namespace)
    return namespace[method_name]


def _method_source(file_path, class_name, method_name):
    source = file_path.read_text()
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def _install_module(monkeypatch, name, **attributes):
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        module_name = ".".join(parts[:index])
        if module_name not in sys.modules:
            module = ModuleType(module_name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, module_name, module)
    module = sys.modules[name]
    for key, value in attributes.items():
        setattr(module, key, value)


def test_online_pdm_uses_official_evaluation_sampling(monkeypatch):
    calls = {}

    class TrajectorySampling:
        def __init__(self, num_poses, interval_length):
            self.num_poses = num_poses
            self.interval_length = interval_length

    class Trajectory:
        def __init__(self, poses):
            self.poses = poses

    class PDMSimulator:
        def __init__(self, proposal_sampling):
            calls["simulator_sampling"] = proposal_sampling

    class PDMScorer:
        def __init__(self, proposal_sampling):
            calls["scorer_sampling"] = proposal_sampling

    def pdm_score(**kwargs):
        calls["pdm_score"] = kwargs
        return SimpleNamespace(
            no_at_fault_collisions=0.91,
            drivable_area_compliance=0.82,
            ego_progress=0.73,
            time_to_collision_within_bound=0.64,
            comfort=0.55,
            score=0.46,
        )

    _install_module(
        monkeypatch,
        "nuplan.planning.simulation.trajectory.trajectory_sampling",
        TrajectorySampling=TrajectorySampling,
    )
    _install_module(
        monkeypatch,
        "navsim.common.dataclasses",
        Trajectory=Trajectory,
    )
    _install_module(
        monkeypatch,
        "navsim.evaluate.pdm_score",
        pdm_score=pdm_score,
    )
    _install_module(
        monkeypatch,
        "navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator",
        PDMSimulator=PDMSimulator,
    )
    _install_module(
        monkeypatch,
        "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer",
        PDMScorer=PDMScorer,
    )

    result = {
        "token": "scene-token",
        "trajectory": np.arange(120, dtype=np.float32).reshape(40, 3),
        "score": float("nan"),
    }
    dataset = SimpleNamespace(
        metric_cache_dict={"scene-token": "cache-path"},
        get_metric_cache=lambda token: f"cache:{token}",
    )

    _load_online_scoring_method()(dataset, [result])

    model_trajectory = calls["pdm_score"]["model_trajectory"]
    future_sampling = calls["pdm_score"]["future_sampling"]
    assert model_trajectory.poses.shape == (8, 3)
    assert future_sampling.num_poses == 40
    assert future_sampling.interval_length == 0.1
    assert calls["simulator_sampling"] is future_sampling
    assert calls["scorer_sampling"] is future_sampling
    assert result["score"] == 0.46
    assert result["drivable_area_compliance"] == 0.82


def test_generated_trajectory_heads_defer_scores_to_online_evaluation():
    source = _method_source(DETECTOR_FILE, "NAVFormer", "forward_test")
    assert "requires_online_pdm_scoring" in source
    assert "use_online_pdm" in source
    assert "float('nan')" in source
    assert "DiffusionPlanningHead" not in source
    assert "GoalFlowPlanningHead" not in source


def test_generated_results_require_official_rescoring():
    requires_official = _load_dataset_method("_requires_official_pdm_rescoring")
    dataset = SimpleNamespace()

    assert requires_official(dataset, [{"score": float("nan")}]) is True
    assert requires_official(dataset, [{"score": 0.75}]) is False
    assert requires_official(
        dataset,
        [{"score": 0.75}, {"score": float("nan")}],
    ) is True


def test_evaluate_skips_embedded_scoring_for_generated_results():
    source = _method_source(DATASET_FILE, "NavSimOpenSceneE2E", "evaluate")

    assert "_requires_official_pdm_rescoring" in source
    assert "_compute_online_pdm_scores" in source
    assert source.index("_requires_official_pdm_rescoring") < source.index(
        "_compute_online_pdm_scores"
    )
    assert "DiffusionPlanningHead" not in source
    assert "GoalFlowPlanningHead" not in source


def test_evaluate_exports_submission_for_navtest_failures_subset():
    source = _method_source(DATASET_FILE, "NavSimOpenSceneE2E", "evaluate")

    assert '"navtest_failures_filtered.yaml"' in source
    assert "_navsim_submission.pkl" in source


def _reward_scorer(progress, multi=None, threshold=0.1):
    progress = np.asarray(progress, dtype=np.float64)
    count = len(progress)
    if multi is None:
        multi = np.ones((2, count), dtype=np.float64)
    weighted = np.ones((4, count), dtype=np.float64)
    config = SimpleNamespace(
        progress_distance_threshold=threshold,
        weighted_metrics_array=np.asarray([5.0, 5.0, 2.0, 0.0]),
    )
    return SimpleNamespace(
        _multi_metrics=np.asarray(multi, dtype=np.float64),
        _weighted_metrics=weighted,
        _progress_raw=progress,
        _config=config,
    )


def _expected_score(progress_component, multiplicative):
    weighted = (5.0 * progress_component + 5.0 + 2.0) / 12.0
    return multiplicative * weighted


def test_pairwise_progress_is_unchanged_when_multiplicative_is_one():
    pairwise = _load_pairwise_official_scores()
    scores, components = pairwise(_reward_scorer([10.0, 5.0]))
    assert scores.shape == (1,)
    assert components.shape == (1, 6)
    assert components[0, 2] == 0.5
    assert scores[0] == np.float32(_expected_score(0.5, 1.0))


def test_pairwise_progress_uses_raw_reference_before_reference_gate():
    pairwise = _load_pairwise_official_scores()
    multi = np.ones((2, 2), dtype=np.float64)
    multi[0, 0] = 0.0
    scores, components = pairwise(_reward_scorer([10.0, 5.0], multi))
    assert components[0, 2] == 0.5
    assert scores[0] == np.float32(_expected_score(0.5, 1.0))


def test_pairwise_progress_applies_candidate_gate_after_normalization():
    pairwise = _load_pairwise_official_scores()
    multi = np.ones((2, 2), dtype=np.float64)
    multi[0, 1] = 0.5
    scores, components = pairwise(_reward_scorer([10.0, 20.0], multi))
    assert components[0, 2] == 0.5
    assert scores[0] == np.float32(_expected_score(0.5, 0.5))


def test_pairwise_progress_threshold_fallback_precedes_candidate_gate():
    pairwise = _load_pairwise_official_scores()
    multi = np.ones((2, 2), dtype=np.float64)
    multi[0, 1] = 0.5
    scores, components = pairwise(_reward_scorer([0.01, 0.02], multi))
    assert components[0, 2] == 0.5
    assert scores[0] == np.float32(_expected_score(0.5, 0.5))


def test_pairwise_progress_preserves_twenty_candidate_order():
    pairwise = _load_pairwise_official_scores()
    progress = np.concatenate(([10.0], np.arange(1.0, 21.0)))
    multi = np.ones((2, 21), dtype=np.float64)
    multi[0, 1::3] = 0.5
    scores, components = pairwise(_reward_scorer(progress, multi))
    candidate_multi = multi.prod(axis=0)[1:]
    expected_components = np.asarray(
        [value / max(10.0, value) for value in progress[1:]]
    ) * candidate_multi
    expected_scores = np.asarray(
        [
            _expected_score(component, gate)
            for component, gate in zip(expected_components, candidate_multi)
        ],
        dtype=np.float32,
    )
    assert scores.shape == (20,)
    assert components.shape == (20, 6)
    np.testing.assert_allclose(components[:, 2], expected_components)
    np.testing.assert_allclose(scores, expected_scores)
