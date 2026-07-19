"""A compact AlphaEvolve-style controller for autoresearch."""

from .config import AppConfig, load_config
from .types import Program

__all__ = ["AppConfig", "Program", "load_config"]
