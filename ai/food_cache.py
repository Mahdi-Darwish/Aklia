import json
import os
import tempfile
import threading

CACHE_VERSION = 2
_FILENAME = f"confirmed_food_matches_v{CACHE_VERSION}.json"
def _pick_cache_path() -> str:
    preferred_dir = os.path.dirname(__file__)
    if os.access(preferred_dir, os.W_OK) and not os.environ.get("VERCEL"):
        return os.path.join(preferred_dir, _FILENAME)
    return os.path.join(tempfile.gettempdir(), _FILENAME)
CACHE_PATH = _pick_cache_path()
_lock = threading.Lock()

def _normalize_key(name: str) -> str:
    return " ".join((name or "").strip().lower().split())

def _load() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def _save(data: dict) -> None:
    try:
        tmp_path = CACHE_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, CACHE_PATH)
    except OSError as exc:
        print(f"[food_cache] could not write cache ({exc}) - continuing without caching")

def get_cached_match(name: str) -> dict | None:
    """Entry shape: {"fdc_id": int, "description": str, "per_100g": {...}}"""
    with _lock:
        return _load().get(_normalize_key(name))

def save_confirmed_match(name: str, fdc_id: int, description: str, per_100g: dict) -> None:
    with _lock:
        data = _load()
        data[_normalize_key(name)] = {
            "fdc_id": fdc_id,
            "description": description,
            "per_100g": per_100g,}
        _save(data)
