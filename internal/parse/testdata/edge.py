"""Edge cases.

This second paragraph is dropped from the captured docstring.
"""

import asyncio


def annotated(a: int, b: str = "x") -> dict[str, int]:
    """Has annotations."""
    return {a: len(b)}


async def gen() -> "AsyncIterator[int]":
    """Async generator."""
    yield 1


def multiline_sig(
    a: int,
    b: str = "x",
    *args,
    **kwargs,
) -> dict[str, int]:
    """Wraps onto many lines."""
    return {}


@property
def computed(self) -> int:
    """Computed value."""
    return 42


class Service:
    """Service.

    Long description.
    """

    @classmethod
    def make(cls, name: str = "x") -> "Service":
        """Construct."""
        return cls()


UNICODE_CONST = "héllo"
mixed_var = 1
