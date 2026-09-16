"""L0/L1 shared: a tiny name->class registry used by all pluggable strategies.

Keeping it in its own module (no qslab deps) means every layer can import it
without violating the layer rule.
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    """name -> factory registry with decorator registration.

    Usage:
        BACKENDS = Registry("quant backend")

        @BACKENDS.register("w4.v1")
        class W4V1Backend: ...
    """

    def __init__(self, what: str):
        self._what = what
        self._items: dict[str, object] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._what} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str):
        if name not in self._items:
            raise KeyError(
                f"unknown {self._what} '{name}'; available: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
