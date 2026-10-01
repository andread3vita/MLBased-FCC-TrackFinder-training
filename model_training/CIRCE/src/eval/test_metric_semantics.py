import json
import math

import numpy as np
import polars as pl
import pytest

from src.eval.fcc_cache_parallel import (
    candidate_stats,
    cluster_event,
    fake_flags_vec,
    ggtf_stats,
    overall,
)
from src.eval.ggtf_assign import ggtf_assignment
from src.eval.plot_fcc_metrics import write_metrics_table
from src.eval.report_epoch16_cluster_sweep import _pareto, _safe_delta
from src.eval.merge_forward_shards import merge_frames
from src.eval.report_op_sweep import collect
from src.eval.run_epoch16_cluster_sweep import _complete


def test_overall_keeps_def1_def2_and_scope_separate():
    particles = pl.DataFrame({
        "purity_of_match": [0.80, 0.60, 0.90, 0.75],
        "efficiency_per_hit": [0.40, 0.80, 0.80, 0.90],
        # Deliberately wrong: def1 must be derived from purity, not this cache
        # convenience column, so stale assignment semantics cannot leak in.
        "matched": [False, False, False, False],
        "idea": [True, True, False, True],
    })

    all_targets = overall(particles, None)
    idea = overall(particles, "idea")

    assert all_targets["match_rate"] == pytest.approx(0.5)
    assert all_targets["def2"] == pytest.approx(0.75)
    assert idea["match_rate"] == pytest.approx(1 / 3)
    assert idea["def2"] == pytest.approx(2 / 3)


def test_ggtf_rate_is_not_candidate_normalized_and_splits_fake_causes():
    clusters = pl.DataFrame({
        "cluster_size": [11, 12, 13, 14],
        "ggtf_assigned": [True, True, False, False],
        "ggtf_iou": [0.8, 0.7, 0.2, 0.0],
    })

    stats = ggtf_stats(clusters, cuts=(10,))

    assert stats["ggtf_fake_rate_gt10"] == pytest.approx(1.0)
    assert stats["ggtf_candidate_fake_fraction_gt10"] == pytest.approx(0.5)
    assert stats["ggtf_n_fake_clone_gt10"] == 1
    assert stats["ggtf_n_fake_spurious_gt10"] == 1


def test_strict_def1_complement_treats_exactly_075_as_fake():
    flags = fake_flags_vec(
        np.array([0.75]),
        np.array([1]),
        np.array([0]),
        np.array([181]),
        {(1, 0, 181)},
    )
    candidates = pl.DataFrame({
        "purity": [0.75],
        "matched_mc_idx": [1],
        "event_id": [0],
        "seed": [181],
        "is_fake_idea": [True],
    })

    assert flags.tolist() == [True]
    assert candidate_stats(candidates, 1)["ghost_rate"] == pytest.approx(1.0)


def test_ggtf_ratio_is_infinite_when_no_candidate_matches():
    clusters = pl.DataFrame({
        "cluster_size": [11],
        "ggtf_assigned": [False],
        "ggtf_iou": [0.0],
    })

    assert np.isinf(ggtf_stats(clusters, cuts=(10,))["ggtf_fake_rate_gt10"])


def test_ggtf_assignment_is_one_to_one_for_clone_candidates():
    labels = np.array([0, 0, 1, 1])
    mc = np.array([1, 1, 1, 1])

    cluster_labels, assigned, iou = ggtf_assignment(labels, mc)

    assert cluster_labels.tolist() == [0, 1]
    assert assigned.sum() == 1
    assert np.all(iou > 0)


def test_sweep_report_records_metric_scope_and_normalization(tmp_path):
    point = tmp_path / "arm" / "tb0.1_td0.2_n500"
    point.mkdir(parents=True)
    (point / "summary.json").write_text(json.dumps({
        "tbeta": 0.1,
        "td": 0.2,
        "n_events": 500,
        "n_tracks_total": 1000,
        "candidates_per_event": 3.0,
        "no_cuts": {
            "efficiency": 0.6,
            "match_rate": 0.5,
            "def2": 0.4,
        },
        "idea": {
            "efficiency": 0.7,
            "match_rate": 0.65,
            "def2": 0.55,
        },
        "ggtf_fake_rate_gt10": 1.0,
        "ggtf_n_cand_gt10": 4,
        "ggtf_n_matched_gt10": 2,
        "ggtf_n_fake_clone_gt10": 1,
        "ggtf_n_fake_spurious_gt10": 1,
    }))

    row = collect(str(tmp_path), "arm", "500")[0]

    assert row["def2_no_cuts"] == pytest.approx(40)
    assert row["def2_idea"] == pytest.approx(55)
    assert row["ggtf10_unassigned_per_matched"] == pytest.approx(100)
    assert row["candidate_fake10"] == pytest.approx(50)


def test_self_seed_greedy_does_not_spawn_from_an_already_claimed_seed():
    beta = np.array([0.9, 0.8, 0.1])
    coords = np.array([[0.0], [0.1], [0.2]])

    current = cluster_event("greedy", beta, coords, 0.5, 0.15, 4)
    self_seed = cluster_event(
        "self_seed_greedy", beta, coords, 0.5, 0.15, 4
    )

    assert current.tolist() == [0, 0, 1]
    assert self_seed.tolist() == [0, 0, -1]


def test_markdown_does_not_label_candidate_fraction_as_ggtf_ratio(tmp_path):
    particles = pl.DataFrame({
        "purity_of_match": [0.8],
        "efficiency_per_hit": [0.9],
        "is_reconstructable_idea": [True],
        "is_reconstructable_cld": [True],
        "is_reconstructable_displaced": [True],
    })
    clusters = pl.DataFrame({
        "purity": [0.9, 0.9, 0.2],
        "is_fake_idea": [False, False, True],
        "is_fake_cld": [False, False, True],
        "cluster_size": [11, 12, 13],
        "ggtf_assigned": [True, True, False],
    })
    output = tmp_path / "metrics.md"

    write_metrics_table(particles, clusters, str(output), "test", 0.1, 0.2)
    text = output.read_text()

    assert "unassigned / matched" in text
    assert "**50.0%** (1 / 2)" in text
    assert "Candidate fake fraction" in text
    assert "**33.3%**" in text


def test_pareto_excludes_degenerate_and_oracle_points():
    rows = [
        {
            "point": "valid",
            "n_matched_gt10": 10,
            "promotable": True,
            "ggtf10_unassigned_per_matched": 0.1,
            "idea_def2": 0.9,
        },
        {
            "point": "collapsed",
            "n_matched_gt10": 0,
            "promotable": True,
            "ggtf10_unassigned_per_matched": 0.0,
            "idea_def2": 0.0,
        },
        {
            "point": "oracle",
            "n_matched_gt10": 10,
            "promotable": False,
            "ggtf10_unassigned_per_matched": 0.01,
            "idea_def2": 0.99,
        },
    ]

    assert _pareto(rows, "idea_def2") == {"valid"}


def test_sweep_completion_requires_exact_provenance_and_configuration(tmp_path):
    expected = {
        "clusterer": "greedy",
        "tbeta": 0.1,
        "td": 0.2,
        "min_cluster_hits": 4,
        "merge_td": 0.0,
        "attach_td": 0.0,
        "helix_tol": 0.0,
        "min_target_hits": 3,
        "min_signal_mc": 0,
    }
    (tmp_path / "summary.json").write_text(json.dumps({
        **expected,
        "fix_particle_zero": True,
        "sweep_provenance_sha256": "campaign",
        "event_splits": {},
        "ggtf_candidate_fake_fraction_gt10": 0.1,
        "all_targets_no_reconstruction_cuts": {},
    }))

    assert _complete(tmp_path, "campaign", expected)
    assert not _complete(tmp_path, "other", expected)
    assert not _complete(tmp_path, "campaign", {**expected, "td": 0.3})


def test_nonfinite_split_rates_do_not_produce_nan_deltas():
    assert _safe_delta(math.inf, math.inf) is None
    assert _safe_delta(0.1, 0.2) == pytest.approx(0.1)


def test_shard_merge_sorts_events_but_preserves_hit_order():
    first = pl.DataFrame({
        "seed": [181, 181],
        "event_id": [2, 2],
        "hit_order": [20, 21],
    })
    second = pl.DataFrame({
        "seed": [181, 181],
        "event_id": [1, 1],
        "hit_order": [10, 11],
    })

    sorted_events = merge_frames([first, second], preserve_event_order=False)
    shard_order = merge_frames([first, second], preserve_event_order=True)

    assert sorted_events["hit_order"].to_list() == [10, 11, 20, 21]
    assert shard_order["hit_order"].to_list() == [20, 21, 10, 11]
