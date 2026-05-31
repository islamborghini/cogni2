"""Module docstring."""
import os

CONSTANT = 42

def hello(name):
    """Greet."""
    return f"hi {name}"

async def fetch(url):
    return url

@staticmethod
def utility():
    return 1

class Greeter:
    def greet(self, name):
        return name

@dataclass
class Point:
    x: int
    y: int
