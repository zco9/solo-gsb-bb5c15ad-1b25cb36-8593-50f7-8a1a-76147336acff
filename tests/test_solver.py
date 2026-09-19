"""Unit tests for the MILP solver (no HTTP layer)."""
import pytest

from app.solver import SourceData, solve_blend, solve_contingency


def src(id, max_volume, cost, ca=50.0, mg=10.0, alk=100.0, batch=None):
    return SourceData(
        id=id, max_volume=max_volume, cost=cost,
        calcium=ca, magnesium=mg, alkalinity=alk, batch_size=batch,
    )


def test_cheapest_source_wins():
    sources = [src("B", 1000, 2.0), src("A", 1000, 1.0)]
    out = solve_blend(sources, 500, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(500.0, abs=1e-6)
    assert vols["B"] == pytest.approx(0.0, abs=1e-6)
    assert out.total_cost == pytest.approx(500.0, abs=1e-6)


def test_indicator_constraint_forces_mixing():
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (40.0, 60.0)})
    assert out.status == "optimal"
    w1, w2 = out.volumes
    assert w1 == pytest.approx(500.0, abs=1e-6)
    assert w2 == pytest.approx(500.0, abs=1e-6)
    assert out.indicators["calcium"] == pytest.approx(60.0, abs=1e-6)
    assert out.total_cost == pytest.approx(1500.0, abs=1e-6)


def test_lexicographic_tie_break_uses_id_order():
    # Identical cost/concentrations: any split is optimal.  The usage vector
    # ordered by source id must be lexicographically minimal.
    sources = [
        src("C", 60, 1.0),
        src("A", 100, 1.0),
        src("B", 100, 1.0),
    ]
    out = solve_blend(sources, 100, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B", "C"], out.volumes))
    assert vols["A"] == pytest.approx(0.0, abs=1e-6)
    assert vols["B"] == pytest.approx(40.0, abs=1e-6)
    assert vols["C"] == pytest.approx(60.0, abs=1e-6)


def test_infeasible_indicator_reports_achievable_range():
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    out = solve_blend(sources, 1000, {"calcium": (70.0, 90.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(36.0, abs=1e-6)
    assert ca["max"] == pytest.approx(68.0, abs=1e-6)
    assert ca["requested_min"] == 70.0
    assert "calcium" in out.message


def test_infeasible_volume_reports_max_available():
    sources = [src("W1", 600, 1.0), src("W2", 800, 2.0)]
    out = solve_blend(sources, 2000, {})
    assert out.status == "infeasible"
    assert out.achievable_ranges["volume"]["max"] == pytest.approx(1400.0)
    # ranges are computed against the actually attainable volume
    assert out.achievable_ranges["calcium"]["min"] >= 0


def test_jointly_incompatible_ranges():
    # Calcium wants lots of W1, alkalinity wants lots of W2; the windows
    # cannot be met at the same time.
    sources = [
        src("W1", 1000, 1.0, ca=100.0, alk=200.0),
        src("W2", 1000, 1.0, ca=0.0, alk=0.0),
    ]
    out = solve_blend(
        sources, 1000, {"calcium": (90.0, 100.0), "alkalinity": (0.0, 20.0)}
    )
    assert out.status == "infeasible"
    assert "jointly incompatible" in out.message


# --------------------------------------------------------------------------- #
# supply-reduction contingencies
# --------------------------------------------------------------------------- #

def test_contingency_returns_original_when_cap_not_binding():
    sources = [src("A", 100, 1.0), src("B", 100, 2.0)]
    out = solve_blend(sources, 50, {})
    x0 = dict(zip(["A", "B"], out.volumes))
    cont = solve_contingency(sources, 50, {}, x0, {"A": 100.0})
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([50.0, 0.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(0.0, abs=1e-9)
    assert cont.cost_delta == pytest.approx(0.0, abs=1e-9)


def test_contingency_min_change_then_cost():
    # Allocation (40, 60, 0).  B (expensive) is capped at 40, so 20 units
    # must be redistributed.  Minimum change = 40: A alone (60, 40, 0) or
    # with C involved; the cheapest min-change option avoids C.
    sources = [
        src("A", 100, 1.0),
        src("B", 100, 2.0),
        src("C", 60, 3.0),
    ]
    cont = solve_contingency(
        sources, 100, {}, {"A": 40.0, "B": 60.0, "C": 0.0}, {"B": 40.0}
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([60.0, 40.0, 0.0], abs=1e-6)
    assert cont.deltas == pytest.approx([20.0, -20.0, 0.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(40.0, abs=1e-6)
    assert cont.total_cost == pytest.approx(140.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(-20.0, abs=1e-6)


def test_contingency_cost_increase_possible():
    # Cheap source W1 is cut below what the indicator window requires; the
    # missing volume moves to the more expensive W2.
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    limits = {"calcium": (40.0, 60.0)}
    cont = solve_contingency(
        sources, 1000, limits, {"W1": 500.0, "W2": 500.0}, {"W1": 400.0}
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([400.0, 600.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(200.0, abs=1e-6)
    assert cont.total_cost == pytest.approx(1600.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(100.0, abs=1e-6)
    assert cont.indicators["calcium"] == pytest.approx(52.0, abs=1e-6)


def test_contingency_lexicographic_tie_break():
    # Three identical sources, original (40, 60, 0), both A and B capped at
    # 40: every min-change allocation shifts 20 units away from B; the
    # id-sorted usage vector must be lexicographically minimal.
    sources = [
        src("A", 100, 1.0),
        src("B", 100, 1.0),
        src("C", 60, 1.0),
    ]
    cont = solve_contingency(
        sources, 100, {},
        {"A": 40.0, "B": 60.0, "C": 0.0},
        {"A": 40.0, "B": 40.0},
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([40.0, 40.0, 20.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(40.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(0.0, abs=1e-9)


def test_contingency_zero_cap_eliminates_source():
    sources = [src("A", 100, 1.0), src("B", 100, 2.0)]
    cont = solve_contingency(
        sources, 50, {}, {"A": 50.0, "B": 0.0}, {"A": 0.0}
    )
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([0.0, 50.0], abs=1e-6)
    assert cont.total_adjustment == pytest.approx(100.0, abs=1e-6)


def test_contingency_infeasible_volume():
    sources = [src("W1", 600, 1.0), src("W2", 800, 2.0)]
    cont = solve_contingency(
        sources, 2000, {},
        {"W1": 600.0, "W2": 800.0}, {"W2": 100.0},
    )
    assert cont.status == "infeasible"
    assert cont.achievable_ranges["volume"]["max"] == pytest.approx(700.0)
    assert "below the target" in cont.message


def test_contingency_infeasible_indicator():
    # Total supply stays at 1000 but cutting the high-calcium W1 to 200
    # makes the calcium minimum 40 unattainable (max achievable is 36).
    sources = [
        src("W1", 600, 1.0, ca=100.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    cont = solve_contingency(
        sources, 1000, {"calcium": (40.0, 60.0)},
        {"W1": 500.0, "W2": 500.0}, {"W1": 200.0},
    )
    assert cont.status == "infeasible"
    assert "volume" not in cont.achievable_ranges
    assert cont.achievable_ranges["calcium"]["max"] == pytest.approx(36.0, abs=1e-6)
    assert "calcium" in cont.message


# --------------------------------------------------------------------------- #
# batch sizes (mixed-integer model)
# --------------------------------------------------------------------------- #

def test_batch_forces_integer_multiple():
    # Cheap batched source A (multiples of 200) + expensive continuous B.
    # The continuous optimum A=500 is forbidden; the MILP picks A=400, B=100.
    sources = [src("A", 600, 1.0, batch=200.0), src("B", 1000, 2.0)]
    out = solve_blend(sources, 500, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(400.0, abs=1e-6)
    assert vols["B"] == pytest.approx(100.0, abs=1e-6)
    assert vols["A"] % 200.0 == pytest.approx(0.0, abs=1e-6)
    assert out.total_cost == pytest.approx(600.0, abs=1e-6)


def test_batch_non_divisible_cap_truncates():
    # max_volume 550 with batch 200 -> at most 2 batches (400) are usable.
    sources = [src("A", 550, 1.0, batch=200.0), src("B", 1000, 2.0)]
    out = solve_blend(sources, 500, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(400.0, abs=1e-6)
    assert vols["B"] == pytest.approx(100.0, abs=1e-6)
    assert out.total_cost == pytest.approx(600.0, abs=1e-6)


def test_batch_larger_than_cap_disables_source():
    sources = [src("A", 100, 1.0, batch=200.0), src("B", 1000, 2.0)]
    out = solve_blend(sources, 500, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(0.0, abs=1e-6)
    assert vols["B"] == pytest.approx(500.0, abs=1e-6)


def test_batch_lexicographic_tie_break_mixed():
    # Equal costs; A continuous, B batched (multiples of 100).  The
    # id-sorted lexicographic minimum must respect B's discreteness:
    # A=0 would force B=150 (not a multiple of 100), so A=50, B=100.
    # A continuous solve followed by rounding could never produce this.
    sources = [src("A", 200, 1.0), src("B", 200, 1.0, batch=100.0)]
    out = solve_blend(sources, 150, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(50.0, abs=1e-6)
    assert vols["B"] == pytest.approx(100.0, abs=1e-6)
    assert out.total_cost == pytest.approx(150.0, abs=1e-6)


def test_batch_lexicographic_tie_break_all_batched():
    sources = [
        src("A", 200, 1.0, batch=100.0),
        src("B", 200, 1.0, batch=100.0),
    ]
    out = solve_blend(sources, 100, {})
    assert out.status == "optimal"
    vols = dict(zip(["A", "B"], out.volumes))
    assert vols["A"] == pytest.approx(0.0, abs=1e-6)
    assert vols["B"] == pytest.approx(100.0, abs=1e-6)


def test_batch_infeasible_indicator_uses_milp_ranges():
    # W1 (calcium 100) is batched at 300 with a non-divisible cap of 500, so
    # at most 300 m3 are usable; the achievable calcium maximum is 50, not
    # the 83.3 the continuous relaxation would suggest.
    sources = [
        src("W1", 500, 1.0, ca=100.0, batch=300.0),
        src("W2", 1000, 2.0, ca=0.0),
    ]
    out = solve_blend(sources, 600, {"calcium": (60.0, 100.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(0.0, abs=1e-6)
    assert ca["max"] == pytest.approx(50.0, abs=1e-6)
    assert ca["requested_min"] == 60.0
    assert "calcium" in out.message


def test_batch_infeasible_unattainable_volume():
    # Both sources deliver 200 m3 batches only; 500 m3 cannot be assembled.
    sources = [
        src("W1", 600, 1.0, batch=200.0),
        src("W2", 800, 2.0, batch=200.0),
    ]
    out = solve_blend(sources, 500, {})
    assert out.status == "infeasible"
    vol = out.achievable_ranges["volume"]
    assert vol["requested"] == pytest.approx(500.0)
    assert vol["max_below"] == pytest.approx(400.0, abs=1e-6)
    assert vol["min_above"] == pytest.approx(600.0, abs=1e-6)
    assert "not attainable" in out.message
    # indicator ranges are computed at the nearest attainable volume (400)
    assert out.achievable_ranges["calcium"]["min"] == pytest.approx(50.0, abs=1e-6)
    assert out.achievable_ranges["calcium"]["max"] == pytest.approx(50.0, abs=1e-6)


def test_batch_discrete_gap_inside_reported_range():
    # Attainable calcium values are {20, 60, 100}; the window (40, 59) lies
    # inside the reported [min, max] range but contains no attainable point.
    sources = [
        src("W1", 600, 1.0, ca=100.0, batch=300.0),
        src("W2", 800, 2.0, ca=20.0),
    ]
    out = solve_blend(sources, 600, {"calcium": (40.0, 59.0)})
    assert out.status == "infeasible"
    ca = out.achievable_ranges["calcium"]
    assert ca["min"] == pytest.approx(20.0, abs=1e-6)
    assert ca["max"] == pytest.approx(100.0, abs=1e-6)
    assert "batch" in out.message


def test_contingency_respects_batch_sizes():
    # Original plan A=400 (2 batches), B=100.  Capping A at 300 truncates to
    # one batch (200); the minimum-change re-allocation is A=200, B=300.
    sources = [src("A", 600, 1.0, batch=200.0), src("B", 800, 2.0)]
    out = solve_blend(sources, 500, {})
    x0 = dict(zip(["A", "B"], out.volumes))
    assert x0["A"] == pytest.approx(400.0, abs=1e-6)
    cont = solve_contingency(sources, 500, {}, x0, {"A": 300.0})
    assert cont.status == "optimal"
    assert cont.volumes == pytest.approx([200.0, 300.0], abs=1e-6)
    assert cont.volumes[0] % 200.0 == pytest.approx(0.0, abs=1e-6)
    assert cont.total_adjustment == pytest.approx(400.0, abs=1e-6)
    assert cont.total_cost == pytest.approx(800.0, abs=1e-6)
    assert cont.cost_delta == pytest.approx(200.0, abs=1e-6)


def test_contingency_batch_granularity_infeasible():
    # Plan A=200, B=300 (batches 200/300).  Shutting A off leaves only
    # multiples of 300, so the 500 m3 target becomes unattainable.
    sources = [
        src("A", 600, 1.0, batch=200.0),
        src("B", 600, 1.0, batch=300.0),
    ]
    out = solve_blend(sources, 500, {})
    x0 = dict(zip(["A", "B"], out.volumes))
    assert x0["A"] == pytest.approx(200.0, abs=1e-6)
    assert x0["B"] == pytest.approx(300.0, abs=1e-6)
    cont = solve_contingency(sources, 500, {}, x0, {"A": 0.0})
    assert cont.status == "infeasible"
    vol = cont.achievable_ranges["volume"]
    assert vol["max_below"] == pytest.approx(300.0, abs=1e-6)
    assert vol["min_above"] == pytest.approx(600.0, abs=1e-6)
    assert "not attainable" in cont.message
