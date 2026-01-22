# bgfs/compat.py
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Iterable, Optional


class CompatError(RuntimeError):
    pass


def import_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:
        raise CompatError(f"Failed to import module '{name}': {e}") from e


def resolve_callable(mod, candidates: Iterable[str]) -> Callable[..., Any]:
    """
    Returns the first callable attribute found in `mod` among `candidates`.
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
    progressively drop kwargs not in the signature.
    """
    try:
        return fn(*args, **kwargs)
    except TypeError as e:
        # Attempt to filter kwargs based on signature
        try:
            sig = inspect.signature(fn)
        except Exception:
            raise

        allowed = set()
        for p in sig.parameters.values():
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                allowed.add(p.name)
            elif p.kind == p.VAR_KEYWORD:
                # accepts **kwargs, so original error isn't from unexpected kw
                raise

        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        if filtered == kwargs:
            # signature filtering didn't change anything; re-raise
            raise

        return fn(*args, **filtered)
