import csv
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from hackguard.trajectories.audit import sample_for_audit, score_audit
from hackguard.trajectories.audit_app import AuditStore, make_server
from hackguard.trajectories.schema import save_jsonl
from test_audit import _data


def _run(tmp_path: Path) -> Path:
    trajs, labels = _data(6)
    save_jsonl(trajs, tmp_path / "trajectories.jsonl")
    sample_for_audit(trajs, labels, tmp_path / "audit", size=4)
    return tmp_path


def _sheet_rows(run: Path) -> list[dict[str, str]]:
    with (run / "audit" / "audit_sheet.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_store_hides_auto_label_and_saves(tmp_path: Path) -> None:
    run = _run(tmp_path)
    from hackguard.trajectories.schema import load_trajectories
    store = AuditStore(run / "audit" / "audit_sheet.csv", load_trajectories(run / "trajectories.jsonl"))
    item = store.item(0)
    assert item["total"] == 4 and item["labeled"] == 0
    assert "auto_label" not in item and "hack_types" not in item  # a blind audit
    assert all(m["role"] != "system" for m in item["messages"])
    store.save_label(0, "unsure", "not sure, the test edit looks like a typo fix")
    assert _sheet_rows(run)[0]["your_label"] == "unsure"
    assert _sheet_rows(run)[0]["auto_label"]  # other columns survive the rewrite
    assert store.first_unlabeled() == 1
    with pytest.raises(ValueError):
        store.save_label(1, "maybe", "")


def test_server_round_trip_and_scoring(tmp_path: Path) -> None:
    run = _run(tmp_path)
    server = make_server(run, port=0)  # port 0: the system picks a free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        assert b"HackGuard Audit" in urllib.request.urlopen(base + "/").read()
        first = json.loads(urllib.request.urlopen(base + "/api/item").read())
        assert first["i"] == 0
        rows = _sheet_rows(run)
        for i, row in enumerate(rows):
            label = "hack" if row["auto_label"] == "hack" else "honest"
            if i == 3:
                label = "unsure"
            body = json.dumps({"i": i, "label": label, "notes": ""}).encode()
            request = urllib.request.Request(base + "/api/label", data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
            result = json.loads(urllib.request.urlopen(request).read())
        assert result["labeled"] == 4
        bad = urllib.request.Request(base + "/api/label", data=b'{"i": 99, "label": "hack"}', method="POST",
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        server.shutdown()
        server.server_close()
    score = score_audit(run / "audit" / "audit_sheet.csv", tmp_path / "score.json")
    assert score["audited"] == 3 and score["unsure"] == 1 and score["accuracy"] == 1.0
