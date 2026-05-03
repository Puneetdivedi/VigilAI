"""
Conftest for VigilAI pytest suite.
Sets up an isolated test database and ensures the project root is on sys.path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Use a dedicated test database so tests don't pollute dev data
os.environ.setdefault("DATABASE_URL", "sqlite:///./data/vigilai_test.db")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_FORMAT", "text")
os.environ.setdefault("LOG_LEVEL", "WARNING")   # suppress noise during tests
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")


@pytest.fixture(scope="session", autouse=True)
def ensure_db():
    """Create all DB tables once per test session."""
    from db.models import Base, engine
    Base.metadata.create_all(bind=engine)
    yield
