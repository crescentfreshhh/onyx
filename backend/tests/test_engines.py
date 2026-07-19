import asyncio
import shutil
import subprocess

import numpy as np
import pytest

from onyx import engines, modelstore
from onyx.models import JobSettings


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="nearest")
    graph = helper.make_graph(
        [node],
        "upscale2x",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h2", "w2"])],
        initializer=[scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # Newer onnx libs stamp an IR version too new for older runtimes; pin it
    # so the fixture loads on every onnxruntime we support.
    model.ir_version = 10
    path = tmp_path_factory.mktemp("models") / "2x_test.onnx"
    onnx.save(model, str(path))
    return path


def test_upscaler_doubles_dimensions(tiny_model):
    pytest.importorskip("onnxruntime")
    upscaler = engines.OnnxUpscaler(tiny_model)
    frame = np.random.randint(0, 255, (8, 6, 3), dtype=np.uint8)
    out = upscaler.upscale(frame)
    assert out.shape == (16, 12, 3)
    assert out.dtype == np.uint8
    assert upscaler.probe_scale(6, 8) == 2


def test_decode_command_includes_deinterlace():
    settings = JobSettings.model_validate({"deinterlace": {"enabled": True, "engine": "bwdif"}})
    cmd = engines.decode_command("/in.mkv", settings)
    assert "-vf" in cmd and "bwdif" in cmd
    assert cmd[-1] == "pipe:1"


def test_encode_command_muxes_source_streams():
    settings = JobSettings.model_validate({
        "interpolate": {"enabled": True, "fps": 60},
        "encode": {"container": "mkv"},
    })
    cmd = engines.encode_command("/in.mkv", "/out.mkv", settings, 1280, 720, 25)
    assert "1280x720" in cmd
    assert "fps=60.0" in ",".join(cmd)
    assert "1:a?" in cmd and "1:s?" in cmd and "-map_chapters" in cmd


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_onnx_pipeline_end_to_end(tiny_model, tmp_path):
    pytest.importorskip("onnxruntime")
    source = tmp_path / "in.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=0.5:size=64x48:rate=10",
         "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    output = tmp_path / "out.mkv"
    info = {"width": 64, "height": 48, "fps": 10.0, "duration": 0.5}
    updates = []

    async def on_progress(progress, fps, eta):
        updates.append(progress)

    async def run():
        await engines.run_ai(
            str(source), str(output), JobSettings(), info,
            on_progress, asyncio.Event(), enhance_model=tiny_model,
        )

    asyncio.run(run())
    assert output.is_file() and output.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "128,96"


def test_custom_model_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(modelstore.config, "MODELS_DIR", tmp_path)
    (tmp_path / "4x_MySuperModel.onnx").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    entries = [m for m in modelstore.catalog() if m["id"].startswith("custom:")]
    assert len(entries) == 1
    assert entries[0]["scale"] == 4
    assert entries[0]["status"] == "installed"
    assert modelstore.installed_path("custom:4x_MySuperModel.onnx") is not None


def test_download_falls_back_across_urls(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setattr(modelstore.config, "MODELS_DIR", models_dir)
    source = tmp_path / "payload.onnx"
    source.write_bytes(b"model-bytes")
    entry = {
        "id": "test-model",
        "filename": "2x_test_dl.onnx",
        "sha256": None,
        "urls": [
            (tmp_path / "does_not_exist.onnx").as_uri(),
            source.as_uri(),
        ],
    }
    modelstore._downloads["test-model"] = {"status": "downloading", "progress": 0.0}
    modelstore._download(entry)
    assert (models_dir / "2x_test_dl.onnx").read_bytes() == b"model-bytes"
    assert modelstore._downloads["test-model"]["status"] == "installed"


def test_download_reports_failure_when_all_urls_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(modelstore.config, "MODELS_DIR", tmp_path / "models")
    entry = {
        "id": "test-fail",
        "filename": "2x_fail.onnx",
        "sha256": None,
        "urls": [(tmp_path / "missing_a.onnx").as_uri(), (tmp_path / "missing_b.onnx").as_uri()],
    }
    modelstore._downloads["test-fail"] = {"status": "downloading", "progress": 0.0}
    modelstore._download(entry)
    state = modelstore._downloads["test-fail"]
    assert state["status"] == "failed"
    assert "2 source(s) failed" in state["error"]


def test_import_rejects_unsupported_url():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        modelstore.start_import("https://example.com/model.zip")


def test_pth_conversion_end_to_end(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("spandrel")
    from spandrel.architectures.Compact import Compact

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(modelstore.config, "MODELS_DIR", models_dir)

    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    pth = models_dir / "testcompact.pth"
    torch.save(net.state_dict(), pth)

    entries = {m["id"]: m for m in modelstore.catalog()}
    assert entries["pth:testcompact.pth"]["status"] == "convertible"

    modelstore._downloads["pth:testcompact.pth"] = {"status": "converting", "progress": 0.0}
    modelstore._convert_file(pth, "pth:testcompact.pth")

    onnx_path = models_dir / "2x_testcompact.onnx"
    assert onnx_path.is_file()
    assert modelstore._downloads["pth:testcompact.pth"]["status"] == "installed"

    # Checkpoint entry disappears once its ONNX counterpart exists.
    ids = [m["id"] for m in modelstore.catalog()]
    assert "pth:testcompact.pth" not in ids
    assert "custom:2x_testcompact.onnx" in ids

    pytest.importorskip("onnxruntime")
    upscaler = engines.OnnxUpscaler(onnx_path)
    out = upscaler.upscale(np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8))
    assert out.shape == (20, 24, 3)


def test_import_downloads_and_appears_as_custom(tmp_path, monkeypatch):
    import time as _time

    models_dir = tmp_path / "models"
    monkeypatch.setattr(modelstore.config, "MODELS_DIR", models_dir)
    source = tmp_path / "2x_imported.onnx"
    source.write_bytes(b"imported-bytes")

    model_id = modelstore.start_import(source.as_uri())
    assert model_id == "import:2x_imported.onnx"
    for _ in range(50):
        if modelstore._downloads[model_id]["status"] != "downloading":
            break
        _time.sleep(0.1)
    assert modelstore._downloads[model_id]["status"] == "installed"
    assert (models_dir / "2x_imported.onnx").read_bytes() == b"imported-bytes"
    assert any(m["id"] == "custom:2x_imported.onnx" for m in modelstore.catalog())
