"""Linear-programming blend optimizer.

Model
-----
Variables x_i >= 0: volume taken from source i.

    minimize    sum(cost_i * x_i)
    subject to  sum(x_i)                       = target_volume
                lo_j * target_volume <= sum(conc_ij * x_i) <= hi_j * target_volume
                x_i                            <= max_volume_i

When several optima share the same minimal cost, the usage vector ordered by
ascending source id is minimized lexicographically via a sequence of LPs.

If the problem is infeasible, the achievable (min, max) concentration range of
every indicator is reported, projected onto the supply/volume feasible set.

Supply-reduction contingency plans (``solve_contingency``) use the same
feasible set with tightened supply caps, but optimize a lexicographic
objective: first minimize the total absolute volume change versus the stored
allocation, then minimize cost, and finally the source-id-ordered usage
vector lexicographically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

INDICATORS: Tuple[str, ...] = ("calcium", "magnesium", "alkalinity")

# Half-width (relative) of the interval used to pin a variable after its
# lexicographic minimum has been found.  Comfortably above solver noise and
# below the 6-decimal output rounding for typical magnitudes.
_FIX_RTOL = 1e-8
# Fallback cost slack (relative) if the exact cost-equality refinement fails.
_COST_RTOL = 1e-9


@dataclass
class SourceData:
    id: str
    max_volume: float
    cost: float
    calcium: float
    magnesium: float
    alkalinity: float


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


def solve_blend(
    sources: List[SourceData],
    target_volume: float,
    limits: Dict[str, Tuple[float, float]],
) -> SolveOutput:
    """Solve the blending LP.  ``limits`` maps indicator name -> (min, max)."""
    sources = sorted(sources, key=lambda s: s.id)
    n = len(sources)
    costs = np.array([s.cost for s in sources], dtype=float)
    bounds = [(0.0, float(s.max_volume)) for s in sources]

    A_eq = np.ones((1, n))
    b_eq = np.array([float(target_volume)])

    A_ub, b_ub = _indicator_constraints(sources, target_volume, limits)

    res = linprog(
        costs, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method="highs",
    )
    if res.status == 2:  # provably infeasible
        return _infeasible(sources, target_volume, limits, bounds)
    if res.status != 0:
        raise RuntimeError(f"LP solver failed (status {res.status}): {res.message}")

    x = _lexicographic_min(costs, res.x, A_ub, b_ub, A_eq, b_eq, bounds)
    indicators = {
        name: float(np.array([getattr(s, name) for s in sources]) @ x) / target_volume
        for name in INDICATORS
    }
    return SolveOutput(
        status="optimal",
        volumes=x,
        total_cost=float(costs @ x),
        indicators=indicators,
    )


def _indicator_constraints(
    sources: List[SourceData],
    target_volume: float,
    limits: Mapping[str, Tuple[float, float]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Indicator-window rows: lo_j*V <= sum(conc_ij * x_i) <= hi_j*V."""
    rows, rhs = [], []
    for name in INDICATORS:
        rng = limits.get(name)
        if rng is None:
            continue
        lo, hi = rng
        conc = np.array([getattr(s, name) for s in sources], dtype=float)
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

    Target volume, indicator windows and unit costs are unchanged.
    ``new_caps`` only needs to list sources whose cap is reduced; every
    other source keeps its original ``max_volume``.
    """
    sources = sorted(sources, key=lambda s: s.id)
    n = len(sources)
    x0 = np.array([float(original_volumes[s.id]) for s in sources], dtype=float)
    bounds = [(0.0, float(new_caps.get(s.id, s.max_volume))) for s in sources]
    costs = np.array([s.cost for s in sources], dtype=float)

    A_eq = np.ones((1, n))
    b_eq = np.array([float(target_volume)])
    A_ub, b_ub = _indicator_constraints(sources, target_volume, limits)

    # Phase 1: minimize sum(d_i) subject to |x_i - x0_i| <= d_i.
    # Variables are ordered (x_1..x_n, d_1..d_n).
    A1, b1, A1_eq, b1_eq, bounds1 = _l1_model(
        x0, A_ub, b_ub, A_eq, b_eq, bounds
    )
    p1 = linprog(
        np.concatenate([np.zeros(n), np.ones(n)]),
        A_ub=A1, b_ub=b1, A_eq=A1_eq, b_eq=b1_eq,
        bounds=bounds1, method="highs",
    )
    if p1.status == 2:  # provably infeasible under the new caps
        achievable, message = _analyze_infeasible(
            sources, target_volume, limits, bounds
        )
        return ContingencyOutput(
            status="infeasible",
            achievable_ranges=achievable,
            message=message,
        )
    if p1.status != 0:
        raise RuntimeError(f"LP solver failed (status {p1.status}): {p1.message}")
    adj_star = float(p1.fun)

    x = _contingency_lexicographic(
        costs, x0, adj_star, A_ub, b_ub, A_eq, b_eq, bounds
    )

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
        total_cost=float(costs @ x),
        cost_delta=float(costs @ (x - x0)),
        indicators=indicators,
    )


def _l1_model(x0, A_ub, b_ub, A_eq, b_eq, bounds):
    """Build the (x, d) model whose feasible set is exactly the allocations
    x satisfying the original constraints with |x_i - x0_i| <= d_i."""
    n = len(x0)
    eye = np.eye(n)
    rows = [
        np.hstack([eye, -eye]),       # x_i - d_i <= x0_i
        np.hstack([-eye, -eye]),      # -x_i - d_i <= -x0_i
    ]
    rhs = [x0, -x0]
    if A_ub is not None:
        rows.insert(0, np.hstack([A_ub, np.zeros((A_ub.shape[0], n))]))
        rhs.insert(0, b_ub)
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    A_eq_ext = np.hstack([A_eq, np.zeros((A_eq.shape[0], n))])
    bounds_ext = list(bounds) + [(0.0, None)] * n
    return A, b, A_eq_ext, b_eq, bounds_ext


def _contingency_lexicographic(
    costs, x0, adj_star, A_ub, b_ub, A_eq, b_eq, bounds
) -> np.ndarray:
    """Phase 2+3: within the minimum-change allocations, minimize cost and
    then the id-sorted usage vector lexicographically."""
    n = len(x0)
    A_l1, b_l1, A_eq_ext, b_eq_ext, bounds_ext = _l1_model(
        x0, A_ub, b_ub, A_eq, b_eq, bounds
    )
    change_row = np.concatenate([np.zeros(n), np.ones(n)])
    cost_row = np.concatenate([costs, np.zeros(n)])
    adj_tol = _FIX_RTOL * max(1.0, abs(adj_star))

    # Phase 2: minimum cost with the total change pinned at its minimum.
    A_eq2 = np.vstack([A_eq_ext, change_row])
    b_eq2 = np.append(b_eq_ext, adj_star)
    p2 = linprog(
        cost_row, A_ub=A_l1, b_ub=b_l1, A_eq=A_eq2, b_eq=b_eq2,
        bounds=bounds_ext, method="highs",
    )
    if p2.status != 0:  # numerical issue: relax the pin to a tight inequality
        p2 = linprog(
            cost_row,
            A_ub=np.vstack([A_l1, change_row]),
            b_ub=np.append(b_l1, adj_star + adj_tol),
            A_eq=A_eq_ext, b_eq=b_eq_ext, bounds=bounds_ext, method="highs",
        )
        if p2.status != 0:
            raise RuntimeError(
                f"contingency cost refinement failed (status {p2.status}): {p2.message}"
            )
    cost_star = float(p2.fun)

    # Phase 3: lex-min (x_1, ..., x_n) among min-change, min-cost solutions.
    A_eq3 = np.vstack([A_eq2, cost_row])
    b_eq3 = np.append(b_eq2, cost_star)
    try:
        return _sequential_min(n, A_l1, b_l1, A_eq3, b_eq3, bounds_ext)
    except RuntimeError:
        cost_tol = _COST_RTOL * max(1.0, abs(cost_star))
        A_ub3 = np.vstack([A_l1, change_row, cost_row])
        b_ub3 = np.append(b_l1, [adj_star + adj_tol, cost_star + cost_tol])
        return _sequential_min(n, A_ub3, b_ub3, A_eq_ext, b_eq_ext, bounds_ext)


def _lexicographic_min(costs, x_star, A_ub, b_ub, A_eq, b_eq, bounds) -> np.ndarray:
    """Among all cost-optimal solutions, minimize (x_1, ..., x_n)
    lexicographically (indices follow the id-sorted source order)."""
    n = len(costs)
    # Pin the optimal cost as an equality; x_star satisfies it by construction.
    cost_value = float(costs @ x_star)
    A_eq2 = np.vstack([A_eq, costs.reshape(1, -1)])
    b_eq2 = np.append(b_eq, cost_value)
    try:
        return _sequential_min(n, A_ub, b_ub, A_eq2, b_eq2, bounds)
    except RuntimeError:
        # Fallback: relax the cost pin to a tight inequality.
        tol = _COST_RTOL * max(1.0, abs(cost_value))
        cost_row = costs.reshape(1, -1)
        A_ub2 = cost_row if A_ub is None else np.vstack([A_ub, cost_row])
        b_ub2 = (
            np.array([cost_value + tol])
            if b_ub is None
            else np.append(b_ub, cost_value + tol)
        )
        return _sequential_min(n, A_ub2, b_ub2, A_eq, b_eq, bounds)


def _sequential_min(n, A_ub, b_ub, A_eq, b_eq, bounds) -> np.ndarray:
    """Lexicographically minimize (x_0, ..., x_{n-1}) by solving one LP per
    coordinate; ``bounds`` may contain additional variables (e.g. slack
    variables), which stay free after the first ``n`` coordinates."""
    bnds = list(bounds)
    n_vars = len(bnds)
    x = np.zeros(n_vars)
    for k in range(n):
        obj = np.zeros(n_vars)
        obj[k] = 1.0
        r = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                    bounds=bnds, method="highs")
        if r.status != 0:
            raise RuntimeError(f"lexicographic refinement failed at index {k}: {r.message}")
        lo0, hi0 = bounds[k]
        v = min(max(float(r.x[k]), lo0), hi0)
        x[k] = v
        eps = _FIX_RTOL * max(1.0, abs(v))
        bnds[k] = (max(lo0, v - eps), min(hi0, v + eps))
    return x[:n]


def _infeasible(sources, target_volume, limits, bounds) -> SolveOutput:
    """Explain infeasibility with per-indicator achievable ranges."""
    achievable, message = _analyze_infeasible(sources, target_volume, limits, bounds)
    return SolveOutput(
        status="infeasible",
        achievable_ranges=achievable,
        message=message,
    )


def _analyze_infeasible(sources, target_volume, limits, bounds):
    """Per-indicator achievable ranges and a human-readable explanation,
    projected onto the supply/volume feasible set."""
    total_supply = sum(hi for _, hi in bounds)
    achievable: Dict[str, dict] = {}
    messages: List[str] = []

    volume = float(target_volume)
    if total_supply < target_volume * (1.0 - 1e-9):
        messages.append(
            f"total available volume {total_supply:g} is below the target {target_volume:g}"
        )
        achievable["volume"] = {"requested": float(target_volume), "max": float(total_supply)}
        volume = float(total_supply)

    if volume > 0:
        n = len(sources)
        A_eq = np.ones((1, n))
        b_eq = np.array([volume])
        for name in INDICATORS:
            conc = np.array([getattr(s, name) for s in sources], dtype=float)
            lo_res = linprog(conc, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            hi_res = linprog(-conc, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
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
    else:
        messages.append("no source water is available")

    if not messages:
        messages.append(
            "indicator ranges are individually attainable but jointly incompatible "
            "under the supply and volume constraints"
        )
    return achievable, "; ".join(messages)
