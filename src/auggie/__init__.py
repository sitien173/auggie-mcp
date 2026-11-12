"""
auggie - Local semantic grep powered by SQLite and Voyage embeddings.

Search your codebase by meaning, not just keywords. auggie indexes your
source files into a local SQLite database and uses Voyage embeddings
to enable semantic search.

Basic Usage:
    CLI:
        $ auggie index .
        $ auggie query "where is authentication handled?"

    Python API:
        >>> from auggie import index_path, query
        >>> from pathlib import Path
        >>> index_path(Path("."), Path(".auggie/index.sqlite"))
        >>> hits = query(Path(".auggie/index.sqlite"), "database connection")
        >>> for h in hits:
        ...     print(f"{h.path}:{h.start_line}")

Modules:
    indexer: File scanning and embedding generation
    search: Semantic query functionality
    chunking: Text splitting with overlap
    db: SQLite database management
    embed: Voyage embedding API wrapper
    commands: CLI command implementations
"""

from .chunking import Chunk, chunk_lines
from .db import connect
from .embed import embed_texts
from .filetype import (
    FileTypeResult,
    detect_file_types_batch,
    has_file_command,
)
from .indexer import DEFAULT_EXCLUDES, index_path, iter_files, load_auggieignore
from .models import (
    DEFAULT_CHUNK_LINES,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP_LINES,
    MODELS,
    EmbeddingModel,
    get_model,
    get_optimal_chunk_lines,
    get_optimal_overlap,
    list_models,
    resolve_model,
)
from .search import Hit, query

__version__ = "0.12.0"

__all__ = [
    "__version__",
    # Core functions
    "index_path",
    "query",
    "iter_files",
    "load_auggieignore",
    "embed_texts",
    "connect",
    "chunk_lines",
    # File type detection
    "detect_file_types_batch",
    "has_file_command",
    "FileTypeResult",
    # Constants
    "DEFAULT_EXCLUDES",
    # Model utilities
    "resolve_model",
    "get_model",
    "get_optimal_chunk_lines",
    "get_optimal_overlap",
    "list_models",
    "MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_OVERLAP_LINES",
    # Data classes
    "Hit",
    "Chunk",
    "EmbeddingModel",
]
