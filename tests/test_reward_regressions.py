from __future__ import annotations

from trl_cad.reward import (
    RewardConfig,
    _parse_compile_signal,
    parse_think_and_scad,
    score_scad_rlvr,
)


def test_parse_compile_signal_empty_geometry() -> None:
    signal = _parse_compile_signal(True, "Current top level object is empty")
    assert signal["compile_ok"] is True
    assert signal["empty_geometry"] is True


def test_parse_think_and_scad_clean_format() -> None:
    out = parse_think_and_scad("<think>step by step</think>\ncube([1,1,1]);")
    assert out.format_clean is True
    assert out.think == "step by step"
    assert "cube" in out.scad


def test_reward_format_bonus_and_penalty() -> None:
    cfg = RewardConfig(
        verify_with_openscad=False,
        format_ok_reward=0.5,
        format_missing_think_penalty=-0.5,
        semantic_model_path=None,
    )
    good = "<think>plan</think>\ncube([1,1,1]);"
    bad = "cube([1,1,1]);"

    good_score, good_info = score_scad_rlvr("make cube", good, cfg=cfg)
    bad_score, bad_info = score_scad_rlvr("make cube", bad, cfg=cfg)

    assert good_info["format_clean"] is True
    assert bad_info["format_clean"] is False
    assert good_score > bad_score


def test_reward_semantic_unavailable_fallback() -> None:
    cfg = RewardConfig(
        verify_with_openscad=False,
        semantic_model_path="/path/not/exists",
        semantic_unavailable_reward=0.0,
    )
    score, info = score_scad_rlvr("make cube", "<think>x</think>\ncube([1,1,1]);", cfg=cfg)

    assert info["semantic_available"] is False
    assert isinstance(score, float)
