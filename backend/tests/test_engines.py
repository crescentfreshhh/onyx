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
        await engines.run_onnx(
            str(source), str(output), JobSettings(), info, tiny_model,
            on_progress, asyncio.Event(),
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
