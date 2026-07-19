import importlib
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


def test_preview_missing_file(client):
    resp = client.post("/api/preview", json={"input_path": "nope.mkv"})
    assert resp.status_code == 404


def test_preview_status_unknown_id(client):
    assert client.get("/api/preview/doesnotexist").status_code == 404


def test_preview_duration_capped(client):
    (client.input_dir / "a.mkv").write_bytes(b"x")
    resp = client.post("/api/preview", json={"input_path": "a.mkv", "duration": 120})
    assert resp.status_code == 422


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_preview_end_to_end(client):
    source = client.input_dir / "clip.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=10",
         "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    resp = client.post("/api/preview", json={
        "input_path": "clip.mkv",
        "start_seconds": 0.5,
        "duration": 1,
        "settings": {"enhance": {"enabled": True, "model": "lanczos", "scale": 2}},
    })
    assert resp.status_code == 202
    preview_id = resp.json()["id"]

    status = None
    for _ in range(60):
        status = client.get(f"/api/preview/{preview_id}").json()
        if status["status"] != "rendering":
            break
        time.sleep(0.5)
    assert status is not None and status["status"] == "ready", status

    for side in ("original", "processed"):
        clip = client.get(f"/api/preview/{preview_id}/{side}")
        assert clip.status_code == 200
        assert clip.headers["content-type"] == "video/mp4"
        assert len(clip.content) > 0
