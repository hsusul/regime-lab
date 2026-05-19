"""Configuration loading helpers for RegimeLab CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
ConfigDict = dict[str, Any]
T = TypeVar("T")


class ConfigError(Exception):
    """Base exception for configuration failures."""


class ConfigNotFoundError(ConfigError):
    """Raised when a requested config file does not exist."""


class InvalidConfigError(ConfigError):
    """Raised when a config file cannot be parsed into a mapping."""


def load_config(config_path: Path | str | None) -> ConfigDict:
    """Load a YAML config file, returning an empty config when omitted."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise ConfigNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise InvalidConfigError(f"Config file must contain a mapping: {path}")
    return content


def get_config_value(config: ConfigDict, dotted_path: str, default: T | None = None) -> Any:
    """Return a nested config value using dotted path syntax."""
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def cli_or_config(
    cli_value: T | None,
    config: ConfigDict,
    dotted_path: str,
    default: T,
) -> T:
    """Resolve a value with CLI precedence over config and built-in default."""
    if cli_value is not None:
        return cli_value
    value = get_config_value(config, dotted_path)
    if value is None:
        return default
    return value


def cli_or_config_path(
    cli_value: Path | None,
    config: ConfigDict,
    dotted_path: str,
    default: Path,
) -> Path:
    """Resolve a path with CLI precedence over config and built-in default."""
    value = cli_or_config(cli_value, config, dotted_path, default)
    return value if isinstance(value, Path) else Path(value)
