from __future__ import annotations

from src.eval.scorer import score_consistency, score_instruction_following


def test_instruction_following_constraints_json_keys():
    response = '{"name":"a","age":1,"city":"x"}'
    score = score_instruction_following(
        response=response,
        constraints={"format": "json", "required_keys": ["name", "age", "city"]},
    )
    assert score == 1.0


def test_consistency_semantic_proxy_bounds():
    grouped = {"g1": ["Paris is the capital of France.", "The capital of France is Paris."]}
    score = score_consistency(grouped)
    assert 0.0 <= score <= 1.0
