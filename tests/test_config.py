import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from api.config import DOCS_DIR, OPENAI_MODEL


def test_defaults_are_strings():
    assert isinstance(DOCS_DIR, str)
    assert isinstance(OPENAI_MODEL, str)
