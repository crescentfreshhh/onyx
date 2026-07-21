import importlib
import os
import shutil
import subprocess
import sys
import time

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
    # Distinct jobs (differing fps) must not clobber each other's output.
    (client.input_dir / "movie.mkv").write_bytes(b"x")
    first = client.post("/api/jobs", json={
        "input_path": "movie.mkv",
        "settings": {"interpolate": {"enabled": True, "fps": 50}},
    }).json()
    second = client.post("/api/jobs", json={
        "input_path": "movie.mkv",
        "settings": {"interpolate": {"enabled": True, "fps": 60}},
    }).json()
    third = client.post("/api/jobs", json={
        "input_path": "movie.mkv",
        "settings": {"interpolate": {"enabled": True, "fps": 30}},
    }).json()
    assert first["output_path"].endswith("movie_onyx.mkv")
    assert second["output_path"].endswith("movie_onyx (1).mkv")
    assert third["output_path"].endswith("movie_onyx (2).mkv")


def test_output_collision_with_existing_file(client, tmp_path):
    (client.input_dir / "movie.mkv").write_bytes(b"x")
    (tmp_path / "output" / "movie_onyx.mkv").write_bytes(b"old render")
    job = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    assert job["output_path"].endswith("movie_onyx (1).mkv")


def test_tagged_filename(client):
    (client.input_dir / "clip.mkv").write_bytes(b"x")
    job = client.post("/api/jobs", json={
        "input_path": "clip.mkv",
        "settings": {
            "enhance": {"enabled": True, "model": "custom:2xNomos.onnx", "scale": 2},
            "interpolate": {"enabled": True, "model": "custom:rife_v4.6.onnx", "fps": 60},
            "encode": {"tag_filename": True, "quality": 18, "codec": "libx264"},
        },
    }).json()
    name = job["output_path"].rsplit("/", 1)[-1]
    assert name == "clip_onyx_2x-2xNomos_60fps-rife_v4.6_crf18-libx264.mkv"


def test_identical_pending_job_not_duplicated(client):
    (client.input_dir / "movie.mkv").write_bytes(b"x")
    first = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    second = client.post("/api/jobs", json={"input_path": "movie.mkv"}).json()
    # same file + settings while still pending -> returns the existing job
    assert first["id"] == second["id"]
    assert len(client.get("/api/jobs").json()) == 1
    # different settings -> a genuinely distinct job
    third = client.post("/api/jobs", json={
        "input_path": "movie.mkv",
        "settings": {"interpolate": {"enabled": True, "fps": 60}},
    }).json()
    assert third["id"] != first["id"]
    assert len(client.get("/api/jobs").json()) == 2


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_cancel_terminates_running_job(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ONYX_INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("ONYX_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.delenv("ONYX_DISABLE_WORKER", raising=False)
    for mod in [m for m in list(sys.modules) if m.startswith("onyx")]:
        del sys.modules[mod]
    main = importlib.import_module("onyx.main")

    with TestClient(main.app) as client:
        source = tmp_path / "input" / "long.mkv"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
             "testsrc=duration=30:size=128x96:rate=25", "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )
        # minterpolate is slow enough that the job is reliably still running
        # when the cancel lands.
        job = client.post("/api/jobs", json={
            "input_path": "long.mkv",
            "settings": {"interpolate": {"enabled": True, "model": "minterpolate", "fps": 120}},
        }).json()

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = client.get("/api/jobs").json()[0]["status"]
            if status == "running":
                break
            time.sleep(0.2)
        assert status == "running"

        assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            status = client.get("/api/jobs").json()[0]["status"]
            if status == "canceled":
                break
            time.sleep(0.2)
        assert status == "canceled", f"job stuck in {status!r} after cancel"
        assert not (tmp_path / "output" / "long_onyx.mkv").exists()


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
