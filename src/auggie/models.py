"""
Embedding model definitions for auggie.

Defines the available Voyage embedding models and chunking defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

ENV_MODEL = "AUGGIE_MODEL"
ENV_DIMENSIONS = "AUGGIE_DIMENSIONS"
ENV_CHUNK_LINES = "AUGGIE_CHUNK_LINES"
ENV_OVERLAP_LINES = "AUGGIE_OVERLAP_LINES"

DEFAULT_MODEL = "voyage-code-3"
DEFAULT_CHUNK_LINES = 30
DEFAULT_OVERLAP_LINES = 15
DEFAULT_MAX_BATCH_SIZE = 128


@dataclass(frozen=True)
class EmbeddingModel:
    """Definition of a Voyage embedding model."""

    id: str
    name: str
    description: str
    dimensions: int
    max_dimensions: int | None
    price_per_million: float
    use_cases: tuple[str, ...]
    notes: str | None = None
    optimal_chunk_lines: int = DEFAULT_CHUNK_LINES
    optimal_overlap_lines: int = DEFAULT_OVERLAP_LINES
    context_tokens: int = 32000
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE


MODELS: dict[str, EmbeddingModel] = {
    "voyage-code-3": EmbeddingModel(
        id="voyage-code-3",
        name="Voyage Code 3",
        description="Code-optimized Voyage embedding model",
        dimensions=1024,
        max_dimensions=1024,
        price_per_million=0.06,
        use_cases=(
            "Code search",
            "Code retrieval",
            "Technical documentation",
            "Multi-language code",
        ),
        notes="Requires VOYAGE_API_KEY.",
    ),
    "voyage-3": EmbeddingModel(
        id="voyage-3",
        name="Voyage 3",
        description="General-purpose Voyage embedding model",
        dimensions=1024,
        max_dimensions=1024,
        price_per_million=0.06,
        use_cases=(
            "General semantic search",
            "Multi-domain retrieval",
            "Document retrieval",
        ),
        notes="Requires VOYAGE_API_KEY.",
    ),
    "voyage-3-lite": EmbeddingModel(
        id="voyage-3-lite",
        name="Voyage 3 Lite",
        description="Lightweight Voyage embedding model",
        dimensions=512,
        max_dimensions=512,
        price_per_million=0.02,
        use_cases=(
            "Fast semantic search",
            "Cost-effective retrieval",
            "High throughput",
        ),
        notes="Requires VOYAGE_API_KEY.",
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "voyage": "voyage-code-3",
    "voyage-code": "voyage-code-3",
    "voyage-3": "voyage-3",
    "voyage-lite": "voyage-3-lite",
}


def _get_default_model() -> str:
    """Get the default embedding model."""
    return DEFAULT_MODEL


def resolve_model(model: str | None = None) -> str:
    """Resolve model identifier from argument, alias, or environment."""
    model_input = model or os.environ.get(ENV_MODEL) or _get_default_model()
    resolved = MODEL_ALIASES.get(model_input, model_input)

    if resolved not in MODELS:
        valid = ", ".join(sorted(MODELS.keys()))
        aliases = ", ".join(sorted(MODEL_ALIASES.keys()))
        raise ValueError(
            f"Unknown model: {model_input!r}. Valid models: {valid}. Aliases: {aliases}."
        )

    return resolved


def resolve_dimensions(dimensions: int | None = None, model: str | None = None) -> int | None:
    """Resolve embedding dimensions from argument or environment."""
    if dimensions is not None:
        return dimensions

    env_dim = os.environ.get(ENV_DIMENSIONS)
    if env_dim:
        return int(env_dim)

    return None


def get_model(model_id: str) -> EmbeddingModel:
    """Get model definition by ID."""
    return MODELS[model_id]


def list_models() -> list[EmbeddingModel]:
    """List all available embedding models."""
    return sorted(MODELS.values(), key=lambda m: (m.price_per_million, m.id))


def get_optimal_chunk_lines(model: str | None = None) -> int:
    """Get the chunk size for a model."""
    env_chunk = os.environ.get(ENV_CHUNK_LINES)
    if env_chunk:
        return int(env_chunk)

    resolved = resolve_model(model)
    return MODELS[resolved].optimal_chunk_lines


def get_optimal_overlap(model: str | None = None) -> int:
    """Get the overlap size for a model."""
    env_overlap = os.environ.get(ENV_OVERLAP_LINES)
    if env_overlap:
        return int(env_overlap)

    resolved = resolve_model(model)
    return MODELS[resolved].optimal_overlap_lines


def get_max_batch_size(model: str | None = None) -> int:
    """Get the maximum batch size for a model."""
    resolved = resolve_model(model)
    return MODELS[resolved].max_batch_size


def get_context_tokens(model: str | None = None) -> int:
    """Get the context window size for a model."""
    resolved = resolve_model(model)
    return MODELS[resolved].context_tokens


def format_models_table() -> str:
    """Format models as a human-readable table."""
    lines = [
        "Available Voyage Embedding Models",
        "=" * 60,
        "",
    ]

    for model in list_models():
        lines.extend(
            [
                f"{model.name} ({model.id})",
                "-" * 60,
                f"  {model.description}",
                f"  Dimensions: {model.dimensions}"
                + (f" (max: {model.max_dimensions})" if model.max_dimensions else ""),
                f"  Price: ${model.price_per_million:.2f} / million tokens",
                f"  Use cases: {', '.join(model.use_cases)}",
            ]
        )
        if model.notes:
            lines.append(f"  Note: {model.notes}")
        lines.append("")

    lines.extend(
        [
            "Configuration:",
            f"  Set {ENV_MODEL} to change the default model",
            f"  Set {ENV_DIMENSIONS} to change default dimensions",
            "",
            "Aliases:",
            "  voyage      -> voyage-code-3",
            "  voyage-code -> voyage-code-3",
            "  voyage-3    -> voyage-3",
            "  voyage-lite -> voyage-3-lite",
        ]
    )

    return "\n".join(lines)
