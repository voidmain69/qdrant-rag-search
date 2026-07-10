"""Unit coverage for the eval harness scoring math and the CI quality gate — the metric
computations had no tests, and the gate decides whether CI blocks a PR."""

import pytest

from scripts.eval_search import bootstrap_ci, gate, metrics_for


class TestMetricsFor:
    def test_perfect_single_relevant(self):
        recall, mrr, ndcg = metrics_for(["a"], {"a": 3}, k=10)
        assert (recall, mrr, ndcg) == (1.0, 1.0, 1.0)

    def test_empty_ranking_scores_zero(self):
        assert metrics_for([], {"a": 3}, k=10) == (0.0, 0.0, 0.0)

    def test_mrr_only_counts_strong_grades(self):
        # grade 1 is "related": it counts for Recall but NOT for MRR (which needs grade >= 2)
        recall, mrr, ndcg = metrics_for(["a"], {"a": 1}, k=10)
        assert recall == 1.0
        assert mrr == 0.0
        assert ndcg == 1.0  # only one graded doc, in its ideal position

    def test_wrong_order_lowers_ndcg(self):
        # ideal is [a(3), b(1)]; ranking b before a is suboptimal
        recall, mrr, ndcg = metrics_for(["b", "a"], {"a": 3, "b": 1}, k=10)
        assert recall == 1.0
        assert mrr == pytest.approx(0.5)  # first strong (a, grade 3) is at rank 2
        assert ndcg == pytest.approx(0.710, abs=0.01)

    def test_k_truncates(self):
        # the relevant doc is below the cutoff
        assert metrics_for(["x", "a"], {"a": 3}, k=1) == (0.0, 0.0, 0.0)


class TestBootstrapCI:
    def test_single_value_is_degenerate(self):
        assert bootstrap_ci([0.5]) == (0.5, 0.5)

    def test_identical_values_collapse(self):
        assert bootstrap_ci([0.8, 0.8, 0.8]) == (0.8, 0.8)

    def test_interval_brackets_the_mean(self):
        values = [0.0, 1.0, 0.5, 0.9, 0.3]
        lo, hi = bootstrap_ci(values)
        m = sum(values) / len(values)
        assert lo <= m <= hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_deterministic(self):
        values = [0.2, 0.9, 0.4, 1.0]
        assert bootstrap_ci(values) == bootstrap_ci(values)


def _results(overall: float, segments: dict | None = None) -> dict:
    return {"overall": {"ndcg": overall}, "segments": segments or {}}


class TestGate:
    def test_floor_pass_and_fail(self):
        assert gate(_results(0.95), None, min_ndcg=0.90, max_regression=None) == []
        assert gate(_results(0.80), None, min_ndcg=0.90, max_regression=None)  # non-empty → fail

    def test_regression_vs_baseline(self):
        base = _results(0.95)
        assert gate(_results(0.93), base, None, max_regression=0.03) == []  # 0.02 drop OK
        assert gate(_results(0.90), base, None, max_regression=0.03)  # 0.05 drop → fail

    def test_segment_regression_flagged(self):
        base = _results(0.95, {"mode:relaxed": {"ndcg": 0.99}})
        cand = _results(0.95, {"mode:relaxed": {"ndcg": 0.90}})
        reasons = gate(cand, base, None, max_regression=0.03)
        assert any("mode:relaxed" in r for r in reasons)

    def test_strict_segment_is_never_gated(self):
        # strict depends on the LLM path — reported but not gated, so it can't fail CI
        base = _results(0.95, {"mode:strict": {"ndcg": 0.99}})
        cand = _results(0.95, {"mode:strict": {"ndcg": 0.40}})
        assert gate(cand, base, None, max_regression=0.03) == []
