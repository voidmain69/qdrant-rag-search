import pytest

from app.core.config import LogFormat, Settings


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


class TestProductionGuard:
    def test_production_requires_api_keys(self):
        with pytest.raises(ValueError, match="API_KEYS"):
            make_settings(environment="production", api_keys="")

    def test_production_with_keys_starts(self):
        s = make_settings(environment="production", api_keys="k1, k2")
        assert s.api_key_list == ["k1", "k2"]

    def test_development_allows_empty_keys(self):
        assert make_settings(api_keys="").api_key_list == []


class TestLogFormatDefaults:
    def test_development_defaults_to_text(self):
        assert make_settings().log_format == LogFormat.TEXT

    def test_production_defaults_to_json(self):
        s = make_settings(environment="production", api_keys="k")
        assert s.log_format == LogFormat.JSON

    def test_explicit_format_wins(self):
        s = make_settings(environment="production", api_keys="k", log_format="text")
        assert s.log_format == LogFormat.TEXT
