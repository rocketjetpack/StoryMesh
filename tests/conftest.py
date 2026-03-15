
import pytest
from pytest import Config, Item, Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--real-apis",
        action="store_true",
        default=False,
        help="Run tests that hit real LLM APIs"
    )

def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    if config.getoption("--real-apis"):
        return
    
    skip_marker = pytest.mark.skip(reason="requires --real-apis option to run")

    for item in items:
        if "real_api" in item.keywords:
            item.add_marker(skip_marker)