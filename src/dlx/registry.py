from typing import Callable, Dict, Tuple

_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "model": {},
    "dataset": {},
}

def register(kind: str, name: str):
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown kind {kind}. Valid: {list(_REGISTRY.keys())}")
    def decorator(obj):
        key = name.lower()
        if key in _REGISTRY[kind]:
            raise ValueError(f"{kind} '{name}' already registered")
        _REGISTRY[kind][key] = obj
        return obj
    return decorator

def get(kind: str, name: str) -> Callable:
    key = name.lower()
    try:
        return _REGISTRY[kind][key]
    except KeyError as e:
        available = ", ".join(sorted(_REGISTRY[kind].keys()))
        raise KeyError(f"{kind} '{name}' not found. Available: {available}") from e

def available(kind: str) -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY[kind].keys()))