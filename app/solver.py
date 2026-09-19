"""Mixed-integer linear-programming blend optimizer.

Model
-----
Variables x_i >= 0: volume taken from source i.  A source with a
``batch_size`` may only be used in non-negative integer multiples of it
(``x_i = batch_size_i * k_i`` with integer ``k_i >= 0``); sources without a
batch size stay continuous.

    minimize    sum(cost_i * x_i)
    subject to  sum(x_i)                       = target_volume
                lo_j * target_volume <= sum(conc_ij * x_i) <= hi_j * target_volume
                x_i                            <= max_volume_i
                x_i in batch_size_i * Z_{>=0}  (sources with a batch size)

Batched sources are substituted by their integer batch count
(``x_i = batch_size_i * k_i``), so every solve below is a genuine
mixed-integer program handled by HiGHS -- the continuous relaxation is never
solved and rounded.

When several optima share the same minimal cost, the usage vector ordered by
ascending source id is minimized lexicographically via a sequence of MILPs
(for a batched source, minimizing its batch count minimizes its volume).

If the problem is infeasible, the achievable (min, max) concentration range of
every indicator is reported, projected onto the same mixed-integer
supply/volume feasible set.  When the target volume itself is unattainable
with the given batch sizes, the nearest attainable volumes below and above
the target are reported as well.

Supply-reduction contingency plans (``solve_contingency``) use the same
feasible set with tightened supply caps (batch constraints included), but
optimize a lexicographic objective: first minimize the total absolute volume
change versus the stored allocation, then minimize cost, and finally the
source-id-ordered usage vector lexicographically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

INDICATORS: Tuple[str, ...] = ("calcium", "magnesium", "alkalinity")

# Half-width (relative) of the interval used to pin a variable after its
# lexicographic minimum has been found.  Comfortably above solver noise and
# below the 6-decimal output rounding for typical magnitudes.
_FIX_RTOL = 1e-8
# Fallback cost slack (relative) if the exact cost-equality refinement fails.
_COST_RTOL = 1e-9
# Slack used when truncating a supply cap to whole batches, so that caps
# like 0.3 with batch size 0.1 (whose float ratio is 2.9999999999999996)
# still allow 3 batches.
_BATCH_RTOL = 1e-9
# Distance within which an integer-constrained value is snapped to the
# nearest integer (HiGHS considers values integral within ~1e-6 already).
_INT_TOL = 1e-6


@dataclass
class SourceData:
    id: str
    max_volume: float
    cost: float
    calcium: float
    magnesium: float
    alkalinity: float
    batch_size: Optional[float] = None  # if set, usage must be a multiple of it


@dataclass
class SolveOutput:
    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    total_cost: Optional[float] = None
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


@dataclass
class ContingencyOutput:
    """Result of a supply-reduction contingency re-optimization."""

    status: str  # "optimal" | "infeasible"
    volumes: Optional[np.ndarray] = None       # aligned with the sorted sources
    deltas: Optional[np.ndarray] = None        # new volume - original volume
    total_adjustment: Optional[float] = None   # sum of absolute deltas
    total_cost: Optional[float] = None
    cost_delta: Optional[float] = None         # new cost - original cost
    indicators: Optional[Dict[str, float]] = None
    achievable_ranges: Optional[Dict[str, dict]] = None
    message: Optional[str] = None


@dataclass
class _ZModel:
    """The blend model in z-space, where ``x_i = scale_i * z_i``.

    A batched source uses its integer batch count as ``z_i``
    (``scale_i = batch_size_i``); a continuous source keeps ``z_i = x_i``
    (``scale_i = 1``).  ``bounds``/``integrality`` describe the z variables.
    """

    scale: np.ndarray
    bounds: List[Tuple[float, float]]
    integrality: np.ndarray


def _compile(sources: List[SourceData], caps: Optional[Mapping[str, float]] = None) -> _ZModel:
    """Translate sources (optionally with tightened caps) into z-space."""
    scale: List[float] = []
    bounds: List[Tuple[float, float]] = []
    integrality: List[int] = []
    for s in sources:
        cap = float(s.max_volume if caps is None else caps.get(s.id, s.max_volume))
        if s.batch_size is None:
            scale.append(1.0)
            bounds.append((0.0, cap))
            integrality.append(0)
        else:
            batch = float(s.batch_size)
            scale.append(batch)
            bounds.append((0.0, float(_max_batches(cap, batch))))
            integrality.append(1)
    return _ZModel(
        scale=np.asarray(scale, dtype=float),
        bounds=bounds,
        integrality=np.asarray(integrality, dtype=int),
    )


def _max_batches(cap: float, batch_size: float) -> int:
    """Largest integer k with ``k * batch_size <= cap`` (up to float noise)."""
    return max(0, int(math.floor(cap / batch_size + _BATCH_RTOL)))


def _milp(obj, A_ub, b_ub, A_eq, b_eq, bounds, integrality):
    """Minimize ``obj @ z`` subject to the given rows, bounds and integrality."""
    constraints = []
    if A_eq is not None:
        constraints.append(LinearConstraint(np.asarray(A_eq, dtype=float), b_eq, b_eq))
    if A_ub is not None:
        constraints.append(LinearConstraint(np.asarray(A_ub, dtype=float), -np.inf, b_ub))
    lb = [b[0] for b in bounds]
    ub = [np.inf if b[1] is None else b[1] for b in bounds]
    return milp(
        np.asarray(obj, dtype=float),
        constraints=constraints or None,
        bounds=Bounds(lb, ub),
        integrality=np.asarray(integrality, dtype=int),
        options={"mip_rel_gap": 0.0},
    )


def _snap(z, integrality) -> np.ndarray:
    """Snap integer-constrained coordinates to the nearest integer (they are
    already within solver tolerance of it)."""
    z = np.asarray(z, dtype=float).copy()
    for i, integ in enumerate(integrality):
        if integ:
            r = round(z[i])
            if abs(z[i] - r) <= _INT_TOL:
                z[i] = float(r)
    return z


def solve_blend(
    sources: List[SourceData],
    target_volume: float,
    limits: Dict[str, Tuple[float, float]],
) -> SolveOutput:
    """Solve the blending MILP.  ``limits`` maps indicator name -> (min, max)."""
    sources = sorted(sources, key=lambda s: s.id)
    model = _compile(sources)
    costs_x = np.array([s.cost for s in sources], dtype=float)
    costs_z = costs_x * model.scale

    A_eq = model.scale.reshape(1, -1)
    b_eq = np.array([float(target_volume)])
    A_ub, b_ub = _indicator_constraints(sources, target_volume, limits, model.scale)

    res = _milp(costs_z, A_ub, b_ub, A_eq, b_eq, model.bounds, model.integrality)
    if res.status == 2:  # provably infeasible
        return _infeasible(sources, target_volume, limits, model)
    if res.status != 0:
        raise RuntimeError(f"MILP solver failed (status {res.status}): {res.message}")

    z = _lexicographic_min(
        costs_z, res.x, A_ub, b_ub, A_eq, b_eq, model.bounds, model.integrality
    )
    x = model.scale * _snap(z, model.integrality)
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return SolveOutput(
        status="optimal",
        volumes=x,
        total_cost=float(costs_x @ x),
        indicators=indicators,
    )


def _indicator_constraints(
    sources: List[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    scale: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Indicator-window rows in z-space:
    lo_j*V <= sum(conc_ij * scale_i * z_i) <= hi_j*V."""
    rows, rhs = [], []
    for name in INDICATORS:
        rng = limits.get(name)
        if rng is None:
            continue
        lo, hi = rng
        conc = np.array([getattr(s, name) for s in sources], dtype=float) * scale
        rows.append(-conc)
        rhs.append(-lo * target_volume)
        rows.append(conc)
        rhs.append(hi * target_volume)
    if not rows:
        return None, None
    return np.asarray(rows), np.asarray(rhs)


def solve_contingency(
    sources: List[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
    original_volumes: Mapping[str, float],
    new_caps: Mapping[str, float],
) -> ContingencyOutput:
    """Re-optimize a saved optimal plan under tightened supply caps.

    The lexicographic objective is:

      1. minimize the total absolute volume change
         ``sum_i |x_i - x_i^0|`` versus the stored allocation;
      2. among minimum-change allocations, minimize total cost;
      3. among those, minimize the usage vector (source ids ascending)
         lexicographically.

    Target volume, indicator windows, unit costs and batch sizes are
    unchanged.  ``new_caps`` only needs to list sources whose cap is
    reduced; every other source keeps its original ``max_volume``.  A cap on
    a batched source is truncated to whole batches.
    """
    sources = sorted(sources, key=lambda s: s.id)
    n = len(sources)
    model = _compile(sources, caps=new_caps)
    x0 = np.array([float(original_volumes[s.id]) for s in sources], dtype=float)
    costs_x = np.array([s.cost for s in sources], dtype=float)
    costs_z = costs_x * model.scale

    A_eq = model.scale.reshape(1, -1)
    b_eq = np.array([float(target_volume)])
    A_ub, b_ub = _indicator_constraints(sources, target_volume, limits, model.scale)

    # Phase 1: minimize sum(d_i) subject to |x_i - x0_i| <= d_i.
    # Variables are ordered (z_1..z_n, d_1..d_n).
    A1, b1, A1_eq, b1_eq, bounds1, integrality1 = _l1_model(
        model, x0, A_ub, b_ub, A_eq, b_eq
    )
    p1 = _milp(
        np.concatenate([np.zeros(n), np.ones(n)]),
        A1, b1, A1_eq, b1_eq, bounds1, integrality1,
    )
    if p1.status == 2:  # provably infeasible under the new caps
        achievable, message = _analyze_infeasible(
            sources, target_volume, limits, model
        )
        return ContingencyOutput(
            status="infeasible",
            achievable_ranges=achievable,
            message=message,
        )
    if p1.status != 0:
        raise RuntimeError(f"MILP solver failed (status {p1.status}): {p1.message}")
    adj_star = float(p1.fun)

    z = _contingency_lexicographic(
        model, costs_z, x0, adj_star, A_ub, b_ub, A_eq, b_eq
    )
    x = model.scale * _snap(z, model.integrality)

    deltas = x - x0
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return ContingencyOutput(
        status="optimal",
        volumes=x,
        deltas=deltas,
        total_adjustment=float(np.abs(deltas).sum()),
        total_cost=float(costs_x @ x),
        cost_delta=float(costs_x @ (x - x0)),
        indicators=indicators,
    )


def _l1_model(model: _ZModel, x0, A_ub, b_ub, A_eq, b_eq):
    """Build the (z, d) model whose feasible set is exactly the allocations
    x = scale*z satisfying the original constraints with |x_i - x0_i| <= d_i."""
    n = len(x0)
    eye = np.eye(n)
    S = np.diag(model.scale)
    rows = [
        np.hstack([S, -eye]),         # scale_i*z_i - d_i <= x0_i
        np.hstack([-S, -eye]),        # -scale_i*z_i - d_i <= -x0_i
    ]
    rhs = [x0, -x0]
    if A_ub is not None:
        rows.insert(0, np.hstack([A_ub, np.zeros((A_ub.shape[0], n))]))
        rhs.insert(0, b_ub)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    A_eq_ext = np.hstack([A_eq, np.zeros((A_eq.shape[0], n))])
    bounds_ext = list(model.bounds) + [(0.0, None)] * n
    integrality_ext = np.concatenate([model.integrality, np.zeros(n, dtype=int)])
    return A, b, A_eq_ext, b_eq, bounds_ext, integrality_ext


def _contingency_lexicographic(
    model: _ZModel, costs_z, x0, adj_star, A_ub, b_ub, A_eq, b_eq
) -> np.ndarray:
    """Phase 2+3: within the minimum-change allocations, minimize cost and
    then the id-sorted usage vector lexicographically."""
    n = len(x0)
    A_l1, b_l1, A_eq_ext, b_eq_ext, bounds_ext, integrality_ext = _l1_model(
        model, x0, A_ub, b_ub, A_eq, b_eq
    )
    change_row = np.concatenate([np.zeros(n), np.ones(n)])
    cost_row = np.concatenate([costs_z, np.zeros(n)])
    adj_tol = _FIX_RTOL * max(1.0, abs(adj_star))

    # Phase 2: minimum cost with the total change pinned at its minimum.
    A_eq2 = np.vstack([A_eq_ext, change_row])
    b_eq2 = np.append(b_eq_ext, adj_star)
    p2 = _milp(cost_row, A_l1, b_l1, A_eq2, b_eq2, bounds_ext, integrality_ext)
    if p2.status != 0:  # numerical issue: relax the pin to a tight inequality
        p2 = _milp(
            cost_row,
            np.vstack([A_l1, change_row]),
            np.append(b_l1, adj_star + adj_tol),
            A_eq_ext, b_eq_ext, bounds_ext, integrality_ext,
        )
        if p2.status != 0:
            raise RuntimeError(
                f"contingency cost refinement failed (status {p2.status}): {p2.message}"
            )
    cost_star = float(p2.fun)

    # Phase 3: lex-min (z_1, ..., z_n) among min-change, min-cost solutions.
    A_eq3 = np.vstack([A_eq2, cost_row])
    b_eq3 = np.append(b_eq2, cost_star)
    try:
        return _sequential_min(n, A_l1, b_l1, A_eq3, b_eq3, bounds_ext, integrality_ext)
    except RuntimeError:
        cost_tol = _COST_RTOL * max(1.0, abs(cost_star))
        A_ub3 = np.vstack([A_l1, change_row, cost_row])
        b_ub3 = np.append(b_l1, [adj_star + adj_tol, cost_star + cost_tol])
        return _sequential_min(
            n, A_ub3, b_ub3, A_eq_ext, b_eq_ext, bounds_ext, integrality_ext
        )


def _lexicographic_min(
    costs_z, z_star, A_ub, b_ub, A_eq, b_eq, bounds, integrality
) -> np.ndarray:
    """Among all cost-optimal solutions, minimize (z_1, ..., z_n)
    lexicographically (indices follow the id-sorted source order; for a
    batched source this minimizes its volume as well)."""
    n = len(costs_z)
    # Pin the optimal cost as an equality; z_star satisfies it by construction.
    cost_value = float(costs_z @ z_star)
    A_eq2 = np.vstack([A_eq, costs_z.reshape(1, -1)])
    b_eq2 = np.append(b_eq, cost_value)
    try:
        return _sequential_min(n, A_ub, b_ub, A_eq2, b_eq2, bounds, integrality)
    except RuntimeError:
        # Fallback: relax the cost pin to a tight inequality.
        tol = _COST_RTOL * max(1.0, abs(cost_value))
        cost_row = costs_z.reshape(1, -1)
        A_ub2 = cost_row if A_ub is None else np.vstack([A_ub, cost_row])
        b_ub2 = (
            np.array([cost_value + tol])
            if b_ub is None
            else np.append(b_ub, cost_value + tol)
        )
        return _sequential_min(n, A_ub2, b_ub2, A_eq, b_eq, bounds, integrality)


def _sequential_min(n, A_ub, b_ub, A_eq, b_eq, bounds, integrality) -> np.ndarray:
    """Lexicographically minimize (z_0, ..., z_{n-1}) by solving one MILP per
    coordinate; ``bounds`` may contain additional variables (e.g. slack
    variables), which stay free after the first ``n`` coordinates."""
    bnds = list(bounds)
    n_vars = len(bnds)
    z = np.zeros(n_vars)
    for k in range(n):
        obj = np.zeros(n_vars)
        obj[k] = 1.0
        r = _milp(obj, A_ub, b_ub, A_eq, b_eq, bnds, integrality)
        if r.status != 0:
            raise RuntimeError(f"lexicographic refinement failed at index {k}: {r.message}")
        lo0, hi0 = bnds[k]
        v = min(max(float(r.x[k]), lo0), hi0)
        if integrality[k]:
            vr = round(v)
            if abs(v - vr) <= _INT_TOL:
                v = float(vr)
        z[k] = v
        eps = _FIX_RTOL * max(1.0, abs(v))
        bnds[k] = (max(lo0, v - eps), min(hi0, v + eps))
    return z[:n]


def _infeasible(sources, target_volume, limits, model: _ZModel) -> SolveOutput:
    """Explain infeasibility with per-indicator achievable ranges."""
    achievable, message = _analyze_infeasible(sources, target_volume, limits, model)
    return SolveOutput(
        status="infeasible",
        achievable_ranges=achievable,
        message=message,
    )


def _analyze_infeasible(sources, target_volume, limits, model: _ZModel):
    """Per-indicator achievable ranges and a human-readable explanation,
    projected onto the same mixed-integer supply/volume feasible set."""
    scale, bounds, integrality = model.scale, model.bounds, model.integrality
    n = len(sources)
    total_supply = float(scale @ np.array([hi for _, hi in bounds]))
    achievable: Dict[str, dict] = {}
    messages: List[str] = []

    volume = float(target_volume)
    if total_supply < target_volume * (1.0 - 1e-9):
        messages.append(
            f"total available volume {total_supply:g} is below the target {target_volume:g}"
        )
        achievable["volume"] = {"requested": float(target_volume), "max": float(total_supply)}
        volume = float(total_supply)
    else:
        # With batch sizes the target volume itself may be unattainable even
        # though enough water is available in aggregate.
        probe = _milp(
            np.zeros(n), None, None,
            scale.reshape(1, -1), np.array([volume]), bounds, integrality,
        )
        if probe.status == 2:
            below, above = _neighboring_volumes(model, volume)
            messages.append(
                f"target volume {volume:g} is not attainable with the source "
                f"batch sizes; nearest achievable volumes are {below:g} "
                f"(below) and {above:g} (above)"
            )
            achievable["volume"] = {
                "requested": float(target_volume),
                "max_below": float(below),
                "min_above": float(above),
            }
            volume = below
        elif probe.status != 0:
            raise RuntimeError(
                f"MILP solver failed (status {probe.status}): {probe.message}"
            )

    if volume > 0:
        A_eq = scale.reshape(1, -1)
        b_eq = np.array([volume])
        for name in INDICATORS:
            conc = np.array([getattr(s, name) for s in sources], dtype=float) * scale
            lo_res = _milp(conc, None, None, A_eq, b_eq, bounds, integrality)
            hi_res = _milp(-conc, None, None, A_eq, b_eq, bounds, integrality)
            entry: Dict[str, float] = {}
            if lo_res.status == 0:
                entry["min"] = float(lo_res.fun) / volume
            if hi_res.status == 0:
                entry["max"] = float(-hi_res.fun) / volume
            req = limits.get(name)
            if req is not None:
                entry["requested_min"], entry["requested_max"] = float(req[0]), float(req[1])
                if "max" in entry and entry["max"] < req[0]:
                    messages.append(
                        f"{name}: requested minimum {req[0]:g} exceeds the achievable "
                        f"maximum {entry['max']:.6g}"
                    )
                if "min" in entry and entry["min"] > req[1]:
                    messages.append(
                        f"{name}: requested maximum {req[1]:g} is below the achievable "
                        f"minimum {entry['min']:.6g}"
                    )
            achievable[name] = entry
    elif not messages:
        messages.append("no source water is available")

    if not messages:
        if integrality.any():
            messages.append(
                "the indicator windows cannot be satisfied under the supply, "
                "volume and batch constraints; with batch sizes the achievable "
                "set is discrete, so concentrations inside the reported ranges "
                "may be unattainable as well"
            )
        else:
            messages.append(
                "indicator ranges are individually attainable but jointly incompatible "
                "under the supply and volume constraints"
            )
    return achievable, "; ".join(messages)


def _neighboring_volumes(model: _ZModel, target: float) -> Tuple[float, float]:
    """Largest attainable total volume <= target and smallest >= target,
    under the batch/supply constraints only."""
    scale, bounds, integrality = model.scale, model.bounds, model.integrality
    row = scale.reshape(1, -1)
    below_res = _milp(-scale, row, np.array([target]), None, None, bounds, integrality)
    above_res = _milp(scale, -row, np.array([-target]), None, None, bounds, integrality)
    if below_res.status != 0 or above_res.status != 0:
        raise RuntimeError("volume projection onto the batch feasible set failed")
    below = float(scale @ _snap(below_res.x, integrality))
    above = float(scale @ _snap(above_res.x, integrality))
    return below, above
