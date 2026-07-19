import asyncio
import shutil
import subprocess

import numpy as np
import pytest

from onyx import engines
from onyx.models import JobSettings


@pytest.fixture(scope="module")
def blend_model(tmp_path_factory):
    """Linear-blend interpolator in the triple-input contract:
    y = img0 * (1 - t) + img1 * t."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    img = lambda name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, "h", "w"])
    nodes = [
        helper.make_node("Sub", ["img1", "img0"], ["diff"]),
        helper.make_node("Mul", ["diff", "timestep"], ["scaled"]),
        helper.make_node("Add", ["img0", "scaled"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "blend",
        [img("img0"), img("img1"),
         helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])],
        [img("y")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    path = tmp_path_factory.mktemp("models") / "rife_blend_test.onnx"
    onnx.save(model, str(path))
    return path


def test_interpolator_blends_at_timestep(blend_model):
    pytest.importorskip("onnxruntime")
    interp = engines.OnnxInterpolator(blend_model)
    a = np.zeros((48, 64, 3), dtype=np.uint8)
    b = np.full((48, 64, 3), 200, dtype=np.uint8)
    mid = interp.interpolate(a, b, 0.5)
    assert mid.shape == (48, 64, 3)
    assert abs(int(mid.mean()) - 100) <= 1
    quarter = interp.interpolate(a, b, 0.25)
    assert abs(int(quarter.mean()) - 50) <= 1


def test_interpolator_pads_odd_dimensions(blend_model):
    pytest.importorskip("onnxruntime")
    interp = engines.OnnxInterpolator(blend_model)
    a = np.random.randint(0, 255, (37, 51, 3), dtype=np.uint8)
    b = np.random.randint(0, 255, (37, 51, 3), dtype=np.uint8)
    out = interp.interpolate(a, b, 0.5)
    assert out.shape == (37, 51, 3)


def test_scene_change_detection():
    same = np.full((64, 64, 3), 100, dtype=np.uint8)
    assert not engines.scene_change(same, same)
    cut = np.zeros((64, 64, 3), dtype=np.uint8)
    other = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert engines.scene_change(cut, other)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_fps_change_end_to_end(blend_model, tmp_path):
    pytest.importorskip("onnxruntime")
    source = tmp_path / "in.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
         "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    output = tmp_path / "out.mkv"
    info = {"width": 64, "height": 48, "fps": 10.0, "duration": 1.0}
    settings = JobSettings.model_validate({"interpolate": {"enabled": True, "fps": 25}})

    async def noop(progress, fps, eta):
        return None

    asyncio.run(engines.run_ai(
        str(source), str(output), settings, info, noop, asyncio.Event(),
        interp_model=blend_model,
    ))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    rate, frames = probe.stdout.strip().split(",")
    assert rate == "25/1"
    assert int(frames) == 25


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_upscale_and_interpolate_combined(blend_model, tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0])
    node = helper.make_node("Resize", ["x", "", "scales"], ["y"], mode="nearest")
    graph = helper.make_graph(
        [node], "up2x",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h2", "w2"])],
        initializer=[scales],
    )
    up_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    up_model.ir_version = 10
    up_path = tmp_path / "2x_up.onnx"
    onnx.save(up_model, str(up_path))

    source = tmp_path / "in.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
         "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    output = tmp_path / "out.mkv"
    info = {"width": 64, "height": 48, "fps": 10.0, "duration": 1.0}
    settings = JobSettings.model_validate({
        "enhance": {"enabled": True, "model": "whatever", "scale": 2},
        "interpolate": {"enabled": True, "fps": 20},
    })

    async def noop(progress, fps, eta):
        return None

    asyncio.run(engines.run_ai(
        str(source), str(output), settings, info, noop, asyncio.Event(),
        enhance_model=up_path, interp_model=blend_model,
    ))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == "128,96,20/1"
