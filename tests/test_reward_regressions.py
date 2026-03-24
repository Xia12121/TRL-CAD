from __future__ import annotations

from trl_cad.reward import (
    RewardConfig,
    _bracket_balance_score,
    _extract_prompt_keywords,
    _module_stats,
    _parse_compile_signal,
    score_scad_rlvr,
)


def test_hardcode_repeat_penalty_not_bypassed_by_unrelated_for_loop() -> None:
    repeated_lines = "\n".join(["translate([1,2,3])" for _ in range(6)])
    code_without_for = f"{repeated_lines}\ncube([1,1,1]);"
    code_with_for = f"{repeated_lines}\nfor(i=[0:1]){{ cube([1,1,1]); }}"

    cfg = RewardConfig(
        min_len=0,
        max_len=10000,
        for_loop_bonus=0.0,
        hardcode_repeat_penalty=-1.0,
        hardcode_repeat_threshold=6,
    )

    score_no_for, info_no_for = score_scad_rlvr("", code_without_for, cfg=cfg)
    score_with_for, info_with_for = score_scad_rlvr("", code_with_for, cfg=cfg)

    assert info_no_for["max_repeated_transform_lines"] == 6
    assert info_with_for["max_repeated_transform_lines"] == 6
    assert score_with_for == score_no_for


def test_parse_compile_signal_does_not_flag_negative_error_phrase() -> None:
    signal = _parse_compile_signal(False, "timeout reached, no errors encountered")
    assert signal["has_error"] is False

    signal_positive = _parse_compile_signal(False, "runtime error in CSG operation")
    assert signal_positive["has_error"] is True


def test_module_stats_ignores_builtin_name_collisions_and_comments() -> None:
    code_builtin_collision = """
module sphere(r=1) { cube([1,1,1]); }
// sphere(5);
sphere(2);
"""
    defs, calls = _module_stats(code_builtin_collision)
    assert defs == 0
    assert calls == 0

    code_comment_ref = """
module bracket() { cube([1,1,1]); }
// bracket();
bracket();
"""
    defs2, calls2 = _module_stats(code_comment_ref)
    assert defs2 == 1
    assert calls2 == 1


def test_bracket_balance_score_is_continuous() -> None:
    score_perfect, imbalance_perfect = _bracket_balance_score("cube([1,1,1]);")
    score_small, imbalance_small = _bracket_balance_score("translate([1,2,3]) { cube([1,1,1]);")
    score_large, imbalance_large = _bracket_balance_score("translate([1,2,3]) { if(true) { cube([1,1,1]);")

    assert imbalance_perfect == 0
    assert score_perfect == 1.0
    assert imbalance_small > 0
    assert imbalance_large > imbalance_small
    assert score_small < score_perfect
    assert score_large < score_small


def test_prompt_keywords_filter_scad_common_words() -> None:
    query = "create a rotating box and add sphere details"
    words = _extract_prompt_keywords(query)

    assert "create" not in words
    assert "sphere" not in words
    assert "rotating" in words
    assert "details" in words


def test_dedup_penalty_default_no_impact() -> None:
    code = "cube([10,10,10]);"
    cfg = RewardConfig(min_len=0, max_len=1000, dedup_repeat_penalty=0.0)
    seen_hashes: set[str] = set()

    score_first, info_first = score_scad_rlvr("", code, cfg=cfg, seen_hashes=seen_hashes)
    score_second, info_second = score_scad_rlvr("", code, cfg=cfg, seen_hashes=seen_hashes)

    assert info_first["dedup_hit"] is False
    assert info_second["dedup_hit"] is True
    assert score_first == score_second
