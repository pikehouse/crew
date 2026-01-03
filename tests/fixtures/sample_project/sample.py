"""Sample module for e2e testing."""


def hello(name: str = "World") -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    print(hello())
