"""End-to-end API tests."""
import json

import pytest

from tests.conftest import make_request


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_solve_optimal(client):
    r = client.post("/api/v1/blends/solve", json=make_request())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    assert body["plan_id"] is None
    assert body["total_cost"] == pytest.approx(1500.0, abs=1e-6)
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    assert blend["W1"] == pytest.approx(500.0, abs=1e-6)
    assert blend["W2"] == pytest.approx(500.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(60.0, abs=1e-6)
    assert body["indicators"]["magnesium"] == pytest.approx(17.5, abs=1e-6)
    assert body["indicators"]["alkalinity"] == pytest.approx(165.0, abs=1e-6)


def test_solve_tie_break_lexicographic(client):
    payload = {
        "target_volume": 100,
        "sources": [
            {"id": "A", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "B", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "C", "max_volume": 60, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    blend = {a["source_id"]: a["volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(40.0, abs=1e-6)
    assert blend["C"] == pytest.approx(60.0, abs=1e-6)


def test_solve_infeasible_indicator(client):
    r = client.post(
        "/api/v1/blends/solve", json=make_request(calcium={"min": 70, "max": 90})
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    ca = body["achievable_ranges"]["calcium"]
    assert ca["max"] == pytest.approx(68.0, abs=1e-6)
    assert ca["requested_min"] == 70
    assert "calcium" in body["message"]


def test_solve_infeasible_volume(client):
    r = client.post("/api/v1/blends/solve", json=make_request(target_volume=2000))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["achievable_ranges"]["volume"]["max"] == pytest.approx(1400.0)


def test_create_and_get_plan(client):
    r = client.post("/api/v1/blends", json=make_request(idempotency_key="plan-1"))
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "optimal"
    assert body["plan_id"] is not None
    assert body["created_at"]

    g = client.get(f"/api/v1/blends/{body['plan_id']}")
    assert g.status_code == 200
    assert g.json() == body


def test_get_plan_not_found(client):
    assert client.get("/api/v1/blends/9999").status_code == 404


def test_idempotent_replay(client):
    payload = make_request(idempotency_key="replay-me")
    r1 = client.post("/api/v1/blends", json=payload)
    r2 = client.post("/api/v1/blends", json=payload)
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.headers.get("x-idempotent-replay") == "true"
    assert r1.json() == r2.json()


def test_idempotency_conflict(client):
    client.post("/api/v1/blends", json=make_request(idempotency_key="dup"))
    r = client.post(
        "/api/v1/blends",
        json=make_request(idempotency_key="dup", target_volume=500),
    )
    assert r.status_code == 409


def test_idempotency_key_via_header(client):
    r = client.post(
        "/api/v1/blends",
        json=make_request(),
        headers={"Idempotency-Key": "header-key"},
    )
    assert r.status_code == 201
    again = client.post(
        "/api/v1/blends",
        json=make_request(),
        headers={"Idempotency-Key": "header-key"},
    )
    assert again.status_code == 200
    assert again.json()["plan_id"] == r.json()["plan_id"]


def test_missing_idempotency_key(client):
    r = client.post("/api/v1/blends", json=make_request())
    assert r.status_code == 422


def test_infeasible_plan_is_persisted(client):
    r = client.post(
        "/api/v1/blends",
        json=make_request(idempotency_key="infeasible", calcium={"min": 70, "max": 90}),
    )
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "infeasible"
    g = client.get(f"/api/v1/blends/{body['plan_id']}")
    assert g.status_code == 200
    assert g.json()["status"] == "infeasible"
    assert "calcium" in g.json()["achievable_ranges"]


def test_validation_source_count(client):
    one = make_request()
    one["sources"] = one["sources"][:1]
    assert client.post("/api/v1/blends/solve", json=one).status_code == 422

    seven = make_request()
    seven["sources"] = [
        {"id": f"S{i}", "max_volume": 10, "cost": 1, "calcium": 1,
         "magnesium": 1, "alkalinity": 1}
        for i in range(7)
    ]
    assert client.post("/api/v1/blends/solve", json=seven).status_code == 422


def test_validation_bad_range_and_duplicate_ids(client):
    bad_range = make_request(calcium={"min": 60, "max": 40})
    assert client.post("/api/v1/blends/solve", json=bad_range).status_code == 422

    dup = make_request()
    dup["sources"][1]["id"] = "W1"
    assert client.post("/api/v1/blends/solve", json=dup).status_code == 422

    neg = make_request(target_volume=-5)
    assert client.post("/api/v1/blends/solve", json=neg).status_code == 422


# --------------------------------------------------------------------------- #
# supply-reduction contingencies
# --------------------------------------------------------------------------- #

def _save_optimal_plan(client, key="cont-1", **overrides):
    r = client.post("/api/v1/blends", json=make_request(idempotency_key=key, **overrides))
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "optimal"
    return body


def test_contingency_feasible(client):
    plan = _save_optimal_plan(client)
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W1": 400}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    assert body["plan_id"] == plan["plan_id"]
    assert body["total_volume"] == 1000
    assert body["total_cost"] == pytest.approx(1600.0, abs=1e-6)
    assert body["cost_delta"] == pytest.approx(100.0, abs=1e-6)
    assert body["total_adjustment"] == pytest.approx(200.0, abs=1e-6)
    rows = {a["source_id"]: a for a in body["blend"]}
    assert rows["W1"]["original_volume"] == pytest.approx(500.0, abs=1e-6)
    assert rows["W1"]["new_volume"] == pytest.approx(400.0, abs=1e-6)
    assert rows["W1"]["volume_delta"] == pytest.approx(-100.0, abs=1e-6)
    assert rows["W1"]["new_cap"] == pytest.approx(400.0, abs=1e-6)
    assert rows["W1"]["cost"] == pytest.approx(400.0, abs=1e-6)
    assert rows["W2"]["original_volume"] == pytest.approx(500.0, abs=1e-6)
    assert rows["W2"]["new_volume"] == pytest.approx(600.0, abs=1e-6)
    assert rows["W2"]["volume_delta"] == pytest.approx(100.0, abs=1e-6)
    # W2 was not listed: its new cap is the original max_volume
    assert rows["W2"]["new_cap"] == pytest.approx(800.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(52.0, abs=1e-6)
    assert body["indicators"]["magnesium"] == pytest.approx(15.0, abs=1e-6)
    assert body["indicators"]["alkalinity"] == pytest.approx(148.0, abs=1e-6)


def test_contingency_non_binding_cap_keeps_plan(client):
    plan = _save_optimal_plan(client, key="cont-keep")
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W2": 800}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total_adjustment"] == pytest.approx(0.0, abs=1e-9)
    assert body["cost_delta"] == pytest.approx(0.0, abs=1e-9)
    assert body["total_cost"] == pytest.approx(1500.0, abs=1e-6)


def test_contingency_does_not_modify_or_persist(client):
    plan = _save_optimal_plan(client, key="cont-ro")
    client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"W1": 400}
    )
    # the stored plan is untouched and fetching it again returns the same body
    g = client.get(f"/api/v1/blends/{plan['plan_id']}")
    assert g.status_code == 200
    assert g.json() == plan


def test_contingency_infeasible_reports_reason(client):
    plan = _save_optimal_plan(client, key="cont-inf")
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency",
        json={"W1": 100, "W2": 800},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    assert body["plan_id"] == plan["plan_id"]
    assert body["achievable_ranges"]["volume"]["max"] == pytest.approx(900.0)
    assert "below the target" in body["message"]
    # no plan was saved or modified on infeasibility either
    assert client.get(f"/api/v1/blends/{plan['plan_id']}").json() == plan


def test_contingency_unknown_plan_404(client):
    r = client.post("/api/v1/blends/9999/contingency", json={"W1": 100})
    assert r.status_code == 404


def test_contingency_non_optimal_plan_422(client):
    r = client.post(
        "/api/v1/blends",
        json=make_request(idempotency_key="cont-bad-status", calcium={"min": 70, "max": 90}),
    )
    assert r.status_code == 201
    plan_id = r.json()["plan_id"]
    resp = client.post(f"/api/v1/blends/{plan_id}/contingency", json={"W1": 100})
    assert resp.status_code == 422
    assert "optimal" in resp.json()["detail"]


def test_contingency_validation_errors(client):
    plan = _save_optimal_plan(client, key="cont-validate")
    url = f"/api/v1/blends/{plan['plan_id']}/contingency"
    # empty map
    assert client.post(url, json={}).status_code == 422
    # unknown source id
    assert client.post(url, json={"NOPE": 100}).status_code == 422
    # cap above the original max
    assert client.post(url, json={"W1": 601}).status_code == 422
    # negative cap
    assert client.post(url, json={"W1": -0.001}).status_code == 422
    # non-numeric / null values (numeric strings coerce, like other fields)
    assert client.post(url, json={"W1": "lots"}).status_code == 422
    assert client.post(url, json={"W1": None}).status_code == 422
    # body must be an object
    assert client.post(url, json=[["W1", 100]]).status_code == 422
    # zero cap is legal
    assert client.post(url, json={"W1": 0}).status_code == 200
    # a cap equal to the original max is legal
    assert client.post(url, json={"W1": 600}).status_code == 200


def test_contingency_lexicographic_tie_break(client):
    payload = {
        "target_volume": 100,
        "sources": [
            {"id": "A", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "B", "max_volume": 100, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "C", "max_volume": 60, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
        "idempotency_key": "cont-lex",
    }
    plan = client.post("/api/v1/blends", json=payload).json()
    # saved blend (lexicographic cost minimum): A=0, B=40, C=60
    # cap C at 30: 30 units must leave C (minimum change 60); the freed
    # volume can go to A or B, and the id-ascending lex-min puts it in B.
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency",
        json={"C": 30},
    )
    assert r.status_code == 200
    blend = {a["source_id"]: a["new_volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(0.0, abs=1e-6)
    assert blend["B"] == pytest.approx(70.0, abs=1e-6)
    assert blend["C"] == pytest.approx(30.0, abs=1e-6)
    assert r.json()["total_adjustment"] == pytest.approx(60.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# batch sizes (mixed-integer model)
# --------------------------------------------------------------------------- #

def test_solve_with_batch_size(client):
    # Both sources batched: the continuous optimum W1=W2=500 is forbidden
    # (500 is a multiple of 100 but not of 200), so W1=400, W2=600.
    payload = make_request()
    payload["sources"][0]["batch_size"] = 200
    payload["sources"][1]["batch_size"] = 100
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    assert blend["W1"] == pytest.approx(400.0, abs=1e-6)
    assert blend["W2"] == pytest.approx(600.0, abs=1e-6)
    assert blend["W1"] % 200 == pytest.approx(0.0, abs=1e-6)
    assert blend["W2"] % 100 == pytest.approx(0.0, abs=1e-6)
    assert body["total_cost"] == pytest.approx(1600.0, abs=1e-6)
    assert body["indicators"]["calcium"] == pytest.approx(52.0, abs=1e-6)


def test_solve_mixed_batch_and_continuous(client):
    # A continuous, B batched at 100, equal costs: the lexicographic minimum
    # is A=50, B=100 (A=0 would require the unattainable B=150).
    payload = {
        "target_volume": 150,
        "sources": [
            {"id": "A", "max_volume": 200, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
            {"id": "B", "max_volume": 200, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 100},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    blend = {a["source_id"]: a["volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(50.0, abs=1e-6)
    assert blend["B"] == pytest.approx(100.0, abs=1e-6)


def test_solve_batch_non_divisible_cap(client):
    # max_volume 550 with batch 200 -> A is usable only up to 400.
    payload = {
        "target_volume": 500,
        "sources": [
            {"id": "A", "max_volume": 550, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 200},
            {"id": "B", "max_volume": 1000, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    blend = {a["source_id"]: a["volume"] for a in r.json()["blend"]}
    assert blend["A"] == pytest.approx(400.0, abs=1e-6)
    assert blend["B"] == pytest.approx(100.0, abs=1e-6)


def test_solve_infeasible_unattainable_volume(client):
    # Only multiples of 200 are attainable; 500 is not one of them.
    payload = {
        "target_volume": 500,
        "sources": [
            {"id": "A", "max_volume": 600, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 200},
            {"id": "B", "max_volume": 800, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 200},
        ],
    }
    r = client.post("/api/v1/blends/solve", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    vol = body["achievable_ranges"]["volume"]
    assert vol["requested"] == pytest.approx(500.0)
    assert vol["max_below"] == pytest.approx(400.0, abs=1e-6)
    assert vol["min_above"] == pytest.approx(600.0, abs=1e-6)
    assert "not attainable" in body["message"]


def test_validation_batch_size(client):
    for bad in (0, -5, "lots"):
        payload = make_request()
        payload["sources"][0]["batch_size"] = bad
        r = client.post("/api/v1/blends/solve", json=payload)
        assert r.status_code == 422, bad
    # non-finite values (sent as raw JSON literals) are rejected too
    for literal in ("Infinity", "-Infinity", "NaN", "1e999"):
        payload = make_request()
        payload["sources"][0]["batch_size"] = "__B__"
        raw = json.dumps(payload).replace('"__B__"', literal)
        r = client.post(
            "/api/v1/blends/solve",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, literal
    # a finite positive batch size (and an omitted one) is accepted
    payload = make_request()
    payload["sources"][0]["batch_size"] = 250
    assert client.post("/api/v1/blends/solve", json=payload).status_code == 200


def test_create_and_get_plan_with_batch_size(client):
    payload = make_request(idempotency_key="batch-plan")
    payload["sources"][0]["batch_size"] = 200
    payload["sources"][1]["batch_size"] = 100
    r = client.post("/api/v1/blends", json=payload)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "optimal"
    blend = {a["source_id"]: a["volume"] for a in body["blend"]}
    assert blend["W1"] == pytest.approx(400.0, abs=1e-6)
    assert blend["W2"] == pytest.approx(600.0, abs=1e-6)

    g = client.get(f"/api/v1/blends/{body['plan_id']}")
    assert g.status_code == 200
    assert g.json() == body


def test_idempotency_fingerprint_covers_batch_size(client):
    payload = make_request(idempotency_key="batch-key")
    payload["sources"][0]["batch_size"] = 200
    r1 = client.post("/api/v1/blends", json=payload)
    assert r1.status_code == 201

    # same key, same request except the batch size -> conflict
    changed = make_request(idempotency_key="batch-key")
    changed["sources"][0]["batch_size"] = 100
    assert client.post("/api/v1/blends", json=changed).status_code == 409

    # identical replay still returns the stored plan
    r2 = client.post("/api/v1/blends", json=payload)
    assert r2.status_code == 200
    assert r2.headers.get("x-idempotent-replay") == "true"
    assert r2.json() == r1.json()


def test_contingency_with_batch_size(client):
    payload = {
        "target_volume": 500,
        "sources": [
            {"id": "A", "max_volume": 600, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 200},
            {"id": "B", "max_volume": 800, "cost": 2, "calcium": 50,
             "magnesium": 10, "alkalinity": 100},
        ],
        "idempotency_key": "cont-batch",
    }
    plan = client.post("/api/v1/blends", json=payload).json()
    assert {a["source_id"]: a["volume"] for a in plan["blend"]} == {
        "A": pytest.approx(400.0, abs=1e-6),
        "B": pytest.approx(100.0, abs=1e-6),
    }
    # cap A at 300 -> only one 200 m3 batch remains usable
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"A": 300}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "optimal"
    rows = {a["source_id"]: a for a in body["blend"]}
    assert rows["A"]["new_volume"] == pytest.approx(200.0, abs=1e-6)
    assert rows["A"]["new_volume"] % 200 == pytest.approx(0.0, abs=1e-6)
    assert rows["B"]["new_volume"] == pytest.approx(300.0, abs=1e-6)
    assert body["total_adjustment"] == pytest.approx(400.0, abs=1e-6)
    assert body["total_cost"] == pytest.approx(800.0, abs=1e-6)
    assert body["cost_delta"] == pytest.approx(200.0, abs=1e-6)


def test_contingency_batch_granularity_infeasible(client):
    payload = {
        "target_volume": 500,
        "sources": [
            {"id": "A", "max_volume": 600, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 200},
            {"id": "B", "max_volume": 600, "cost": 1, "calcium": 50,
             "magnesium": 10, "alkalinity": 100, "batch_size": 300},
        ],
        "idempotency_key": "cont-batch-inf",
    }
    plan = client.post("/api/v1/blends", json=payload).json()
    assert plan["status"] == "optimal"
    # disabling A leaves only multiples of 300 -> 500 is unattainable
    r = client.post(
        f"/api/v1/blends/{plan['plan_id']}/contingency", json={"A": 0}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "infeasible"
    vol = body["achievable_ranges"]["volume"]
    assert vol["max_below"] == pytest.approx(300.0, abs=1e-6)
    assert vol["min_above"] == pytest.approx(600.0, abs=1e-6)
    # the original plan is untouched
    assert client.get(f"/api/v1/blends/{plan['plan_id']}").json() == plan
