from onyx import pipeline
from onyx.models import JobSettings


def test_default_settings_passthrough_no_filters():
    settings = JobSettings()
    assert pipeline.build_filters(settings) == []


def test_full_stack_filter_order():
    settings = JobSettings.model_validate({
        "deinterlace": {"enabled": True, "engine": "bwdif"},
        "enhance": {"enabled": True, "scale": 2},
        "interpolate": {"enabled": True, "fps": 60},
        "grain": {"enabled": True, "amount": 4},
    })
    filters = pipeline.build_filters(settings)
    assert filters == [
        "bwdif",
        "scale=iw*2:ih*2:flags=lanczos",
        "fps=60.0",
        "noise=alls=4.0:allf=t",
    ]


def test_command_maps_streams_and_encoder():
    settings = JobSettings.model_validate({
        "encode": {"codec": "hevc_nvenc", "quality": 22, "container": "mkv"},
    })
    cmd = pipeline.build_command("/input/a.mkv", "/output/a.mkv", settings)
    assert cmd[0].endswith("ffmpeg")
    assert "-vf" not in cmd
    assert ["-c:v", "hevc_nvenc"] == cmd[cmd.index("-c:v"):cmd.index("-c:v") + 2]
    assert "-cq" in cmd and "22" in cmd
    assert ["-map", "0:v:0"] == cmd[cmd.index("0:v:0") - 1:cmd.index("0:v:0") + 1]
    assert "0:s?" in cmd and "-map_chapters" in cmd
    assert cmd[-1] == "/output/a.mkv"


def test_mp4_container_skips_subtitle_mapping():
    settings = JobSettings.model_validate({"encode": {"container": "mp4"}})
    cmd = pipeline.build_command("/input/a.mkv", "/output/a.mp4", settings)
    assert "0:s?" not in cmd


def test_unknown_codec_falls_back_to_x264():
    settings = JobSettings.model_validate({"encode": {"codec": "bogus"}})
    cmd = pipeline.build_command("/i.mkv", "/o.mkv", settings)
    assert "libx264" in cmd


def test_arbitrary_fps_targets():
    settings = JobSettings.model_validate({"interpolate": {"enabled": True, "fps": 72.5}})
    assert pipeline.post_filters(settings) == ["fps=72.5"]


def test_ntsc_rates_use_exact_rationals():
    for decimal, rational in [(23.976, "24000/1001"), (29.97, "30000/1001"), (59.94, "60000/1001")]:
        settings = JobSettings.model_validate({"interpolate": {"enabled": True, "fps": decimal}})
        assert pipeline.post_filters(settings) == [f"fps={rational}"]


def test_fps_out_of_range_rejected():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        JobSettings.model_validate({"interpolate": {"enabled": True, "fps": 0}})
    with pytest.raises(pydantic.ValidationError):
        JobSettings.model_validate({"interpolate": {"enabled": True, "fps": 500}})
