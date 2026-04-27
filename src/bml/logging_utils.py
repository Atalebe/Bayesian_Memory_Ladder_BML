from __future__ import annotations
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def try_cmd(cmd: list[str]) -> dict[str, Any]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return {"ok": True, "cmd": cmd, "output": out.strip()}
    except Exception as e:
        return {"ok": False, "cmd": cmd, "error": repr(e)}

def git_info() -> dict[str, Any]:
    return {
        "rev_parse": try_cmd(["git", "rev-parse", "HEAD"]),
        "status": try_cmd(["git", "status", "--porcelain"]),
    }

def system_info() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "git": git_info(),
    }

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
