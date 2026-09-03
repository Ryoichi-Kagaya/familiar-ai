import pytest

from familiar_agent.response_normalization import collapse_exact_repetition


@pytest.mark.parametrize("copies", [2, 3, 4])
def test_collapses_exact_whole_response_repetition(copies: int) -> None:
    response = "また旅行行こうな。\n\n楽しみにしてるで。"
    assert collapse_exact_repetition(("\n\n\n".join([response] * copies))) == response


@pytest.mark.parametrize(
    "response",
    [
        "大丈夫。大丈夫やで。",
        "はいはいはいはい",
        "また旅行行こうな。\n\n楽しみにしてるで。" * 5,
    ],
)
def test_keeps_non_anomalous_repetition(response: str) -> None:
    assert collapse_exact_repetition(response) == response
