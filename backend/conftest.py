"""Root conftest: ensure the project root is importable for all test suites."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
