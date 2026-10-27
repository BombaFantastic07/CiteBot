import pytest

from ragbot.config import AppConfig


def test_defaults_valid():
    AppConfig().validate()


def test_bad_provider():
    with pytest.raises(ValueError):
        AppConfig(provider="gemini").validate()


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        AppConfig(chunk_size=100, chunk_overlap=100).validate()


def test_top_k_positive():
    with pytest.raises(ValueError):
        AppConfig(top_k=0).validate()
