"""Outer module."""

class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def ping(self):
            return "pong"

    def outer_method(self, x):
        """Outer method."""
        def helper():
            return 1
        return helper() + x

class StandAlone:
    pass
