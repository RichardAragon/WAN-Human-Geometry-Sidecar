from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Iterable


class CompatError(RuntimeError):
    pass


def import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise CompatError(f"Failed to import module '{name}': {e}") from e


def resolve_callable(mod, candidates: Iterable[str]) -> Callable[..., Any]:
    """
    Return the first callable attribute in `mod` among `candidates`.
    Raises a helpful error if none exist.
    """
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn

    public = sorted([k for k in dir(mod) if not k.startswith("_")])
    raise CompatError(
        f"None of the expected callables exist in {mod.__name__}. "
        f"Tried: {list(candidates)}. Available (public): {public}"
    )


def call_with_fallback(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Try calling with kwargs first; if that fails due to unexpected kwargs,
    drop kwargs that are not present in the signature and retry once.
    """
    try:
        return fn(*args, **kwargs)
    except TypeError:
        sig = inspect.signature(fn)

        # If fn accepts **kwargs, the original error isn't from unexpected kw
        for p in sig.parameters.values():
            if p.kind == p.VAR_KEYWORD:
                raise

        allowed = {
            p.name
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}

        if filtered == kwargs:
            raise

        return fn(*args, **filtered)

