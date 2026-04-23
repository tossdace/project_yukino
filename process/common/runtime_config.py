from functools import lru_cache
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHARACTER_DIR = PROJECT_ROOT / "character_files"
AUDIO_DIR = PROJECT_ROOT / "audio"

_CONFIG_CANDIDATES = (
    CHARACTER_DIR / "config.yaml",
    CHARACTER_DIR / "config.public.yaml",
)


@lru_cache(maxsize=1)
def get_config_path() -> Path:
    for candidate in _CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate

    searched_paths = ", ".join(str(path) for path in _CONFIG_CANDIDATES)
    raise FileNotFoundError(f"No character config file found. Looked for: {searched_paths}")


@lru_cache(maxsize=1)
def load_character_config() -> dict:
    config_path = get_config_path()

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Character config must be a mapping: {config_path}")

    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
