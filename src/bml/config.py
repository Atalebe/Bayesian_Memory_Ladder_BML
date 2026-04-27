from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw: dict

    @property
    def run_id(self) -> str:
        return str(self.raw.get("run_id", "unknown"))

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output"]["root"])

def load_config(path: str | Path) -> Config:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
