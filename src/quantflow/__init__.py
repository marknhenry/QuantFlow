from .core import calculate_returns
from . import load_data

# important: importing this registers df.qf
from .accessors import QFAccessor, ReturnsNamespace

__all__ = [
    "calculate_returns",
    "load_data",
    "QFAccessor",
    "ReturnsNamespace",
]