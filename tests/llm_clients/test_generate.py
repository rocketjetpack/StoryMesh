import pytest

from storymesh import generate_synopsis


@pytest.mark.real_api
def test_generate_returns_result(): # noqa: ANN201
    result = generate_synopsis("dark fantasy")
    assert result.final_synopsis is not None
    assert result.metadata["input_genre"] == "dark fantasy"