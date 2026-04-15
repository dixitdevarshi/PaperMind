"""
utils.py
────────
Shared helper functions used across PaperMind components.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)


# ── File I/O ─────────────────────────────────────────────────────────────────

def save_json(data: Any, path: str) -> None:
    """Save a Python object as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON → {path}")


def load_json(path: str) -> Any:
    """Load a JSON file and return as Python object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON ← {path}")
    return data


def save_pickle(obj: Any, path: str) -> None:
    """Serialize a Python object to disk with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle → {path}")


def load_pickle(path: str) -> Any:
    """Load a pickle file from disk."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle ← {path}")
    return obj


# ── Path helpers ─────────────────────────────────────────────────────────────

def get_project_root() -> Path:
    """Returns the absolute path to the project root (where app.py lives)."""
    return Path(__file__).resolve().parents[1]


def get_vectorstore_path() -> Path:
    """Returns the absolute path to the ChromaDB persistence directory."""
    path = get_project_root() / "artifacts" / "vectorstore"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_uploads_path() -> Path:
    """Returns the absolute path to the uploads directory."""
    path = get_project_root() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Text helpers ─────────────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 300) -> str:
    """Truncate text for display/logging purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_source(metadata: dict) -> str:
    """
    Format chunk metadata into a human-readable source string.
    Example: 'GDPR_EN.pdf — Page 12 — Section: Article 5'
    """
    source  = metadata.get("source", "Unknown")
    page    = metadata.get("page", "?")
    section = metadata.get("section", "Unknown")
    return f"{source} — Page {page} — Section: {section}"