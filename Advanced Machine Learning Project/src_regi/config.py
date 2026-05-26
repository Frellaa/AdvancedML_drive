import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = path.resolve()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], relative: str) -> Path:
    return Path(cfg["_project_root"]) / relative


def artifacts_dir(cfg: dict[str, Any]) -> Path:
    d = resolve_path(cfg, cfg["paths"]["artifacts_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d
