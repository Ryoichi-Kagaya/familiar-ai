from familiar_agent._ui_helpers import collapse_exact_repetition


def test_collapses_exact_whole_response_repetition() -> None:
    response = "また旅行行こうな。\n\n楽しみにしてるで。"
    assert collapse_exact_repetition(response + response) == response


def test_keeps_intentional_near_repetition() -> None:
    response = "大丈夫。大丈夫やで。"
    assert collapse_exact_repetition(response) == response


def test_keeps_short_emphatic_repetition() -> None:
    response = "はいはいはいはい"
    assert collapse_exact_repetition(response) == response
