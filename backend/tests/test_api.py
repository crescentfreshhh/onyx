import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ONYX_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("ONYX_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("ONYX_DISABLE_WORKER", "1")
    for mod in [m for m in list(sys.modules) if m.startswith("onyx")]:
        del sys.modules[mod]
    main = importlib.import_module("onyx.main")
    with TestClient(main.app) as tc:
        tc.input_dir = tmp_path / "input"
        yield tc


def test_empty_queue(client):
    assert client.get("/api/jobs").json() == []


def test_create_job_and_cancel(client):
    video = client.input_dir / "sample.mkv"
    video.write_bytes(b"fake")

    resp = client.post("/api/jobs", json={"input_path": "sample.mkv"})
    assert resp.status_code == 201
    job = resp.json()
    assert job["status"] == "queued"
    assert job["output_path"].endswith("sample_onyx.mkv")

    resp = client.post(f"/api/jobs/{job['id']}/cancel")
    assert resp.status_code == 200
    assert client.get("/api/jobs").json()[0]["status"] == "canceled"


def test_output_collision_appends_suffix(client):
    (client.input_dir / "movie.mkv").write_bytes(b"x")
    first = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    second = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    third = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    assert first["output_path"].endswith("movie_onyx.mkv")
    assert second["output_path"].endswith("movie_onyx (1).mkv")
    assert third["output_path"].endswith("movie_onyx (2).mkv")


def test_output_collision_with_existing_file(client, tmp_path):
    (client.input_dir / "movie.mkv").write_bytes(b"x")
    (tmp_path / "output" / "movie_onyx.mkv").write_bytes(b"old render")
    job = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    assert job["output_path"].endswith("movie_onyx (1).mkv")


def test_create_job_missing_file(client):
    resp = client.post("/api/jobs", json={"input_path": "nope.mkv"})
    assert resp.status_code == 404


def test_path_traversal_rejected(client):
    resp = client.post("/api/jobs", json={"input_path": "../../etc/passwd"})
    assert resp.status_code in (400, 404)
    resp = client.get("/api/files", params={"path": "../.."})
    assert resp.status_code in (400, 404)


def test_file_listing_filters_non_video(client):
    (client.input_dir / "movie.mkv").write_bytes(b"x")
    (client.input_dir / "notes.txt").write_text("x")
    (client.input_dir / "subdir").mkdir()
    entries = client.get("/api/files").json()["entries"]
    assert [e["name"] for e in entries] == ["subdir", "movie.mkv"]


def test_builtin_presets_seeded(client):
    presets = client.get("/api/presets").json()
    assert any(p["builtin"] for p in presets)


def test_save_and_delete_user_preset(client):
    resp = client.post("/api/presets", json={
        "name": "My preset",
        "settings": {"interpolate": {"enabled": True, "fps": 60}},
    })
    assert resp.status_code == 201
    preset = next(p for p in client.get("/api/presets").json() if p["name"] == "My preset")
    assert preset["settings"]["interpolate"]["enabled"] is True
    assert client.delete(f"/api/presets/{preset['id']}").status_code == 204
