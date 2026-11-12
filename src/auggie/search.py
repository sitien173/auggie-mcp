"""
Search module for auggie.

Provides semantic search functionality by computing cosine similarity
between a query embedding and stored chunk embeddings. Supports both
numpy (fast) and pure Python (fallback) implementations.

Search modes:
    - semantic: Embedding similarity only (original behavior)
    - fulltext: SQLite FTS5 keyword matching only
    - hybrid: Combined score (default) - best of both worlds

Hybrid fusion methods:
    - rrf (default): Reciprocal Rank Fusion - combines results by rank position.
      More robust than score weighting since it doesn't depend on score scales.
      Formula: score = 1/(k + semantic_rank) + 1/(k + fts_rank), where k=60.

    - alpha: Linear weighted combination of normalized scores.
      Formula: score = alpha * semantic_score + (1 - alpha) * fts_score.
      Legacy method, available for comparison.

Confidence scoring:
    Results include a confidence level (high/medium/low/very_low) that
    indicates match quality. Two modes are available:

    - relative (default): Compares each score to the top result.
      More meaningful since cosine similarity clusters around 0.3-0.5.
      A score of 0.40 that's 90% of the top score (0.44) = "high".

    - absolute: Uses fixed thresholds (legacy behavior).
      Less meaningful for typical embedding distributions.

Environment variables:
    AUGGIE_SEARCH_MODE: Default search mode (semantic, fulltext, hybrid)
    AUGGIE_FUSION_METHOD: Hybrid fusion method ("rrf" or "alpha", default: "rrf")
    AUGGIE_HYBRID_ALPHA: Semantic weight for alpha fusion (0.0-1.0, default: 0.7)
    AUGGIE_RRF_K: RRF rank constant (default: 60, higher = more weight to lower ranks)
    AUGGIE_CONFIDENCE_MODE: "relative" (default) or "absolute"
    AUGGIE_RELATIVE_HIGH: Fraction of top score for "high" (default: 0.90)
    AUGGIE_RELATIVE_MEDIUM: Fraction of top score for "medium" (default: 0.75)
    AUGGIE_RELATIVE_LOW: Fraction of top score for "low" (default: 0.50)
    AUGGIE_CONFIDENCE_HIGH: Absolute threshold for "high" (default: 0.50)
    AUGGIE_CONFIDENCE_MEDIUM: Absolute threshold for "medium" (default: 0.40)
    AUGGIE_CONFIDENCE_LOW: Absolute threshold for "low" (default: 0.30)
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .db import connect, has_fts5
from .embed import embed_texts
from .lance_store import LanceStore, get_lance_dir
from .models import MODEL_ALIASES

# Search mode type
SearchMode = Literal["semantic", "fulltext", "hybrid"]

# Default search mode from environment
DEFAULT_SEARCH_MODE: SearchMode = os.environ.get("AUGGIE_SEARCH_MODE", "hybrid")  # type: ignore[assignment]

# Fusion method for hybrid search: "rrf" (default) or "alpha" (legacy)
# RRF is more robust since it uses ranks instead of scores
FUSION_METHOD = os.environ.get("AUGGIE_FUSION_METHOD", "rrf").lower()

# Hybrid scoring weight for alpha fusion (0.0-1.0, 1.0 = all semantic, 0.0 = all fulltext)
HYBRID_ALPHA = float(os.environ.get("AUGGIE_HYBRID_ALPHA", "0.7"))

# RRF constant k (default: 60, standard in literature)
# Higher k = more weight to lower-ranked results, smoother ranking
RRF_K = int(os.environ.get("AUGGIE_RRF_K", "60"))

# Confidence mode: "relative" (percentile-based, default) or "absolute" (threshold-based)
# Relative mode compares scores to the top result, which is more meaningful since
# cosine similarity for text embeddings clusters around 0.3-0.5, not uniformly [0,1]
CONFIDENCE_MODE = os.environ.get("AUGGIE_CONFIDENCE_MODE", "relative")

# Absolute confidence thresholds (calibrated for typical embedding distributions)
# These are used when AUGGIE_CONFIDENCE_MODE=absolute
CONFIDENCE_HIGH = float(os.environ.get("AUGGIE_CONFIDENCE_HIGH", "0.50"))
CONFIDENCE_MEDIUM = float(os.environ.get("AUGGIE_CONFIDENCE_MEDIUM", "0.40"))
CONFIDENCE_LOW = float(os.environ.get("AUGGIE_CONFIDENCE_LOW", "0.30"))

# Relative confidence thresholds (as percentage of top score)
# These define what fraction of the top score qualifies for each level
RELATIVE_HIGH = float(os.environ.get("AUGGIE_RELATIVE_HIGH", "0.90"))  # >= 90% of top
RELATIVE_MEDIUM = float(os.environ.get("AUGGIE_RELATIVE_MEDIUM", "0.75"))  # >= 75% of top
RELATIVE_LOW = float(os.environ.get("AUGGIE_RELATIVE_LOW", "0.50"))  # >= 50% of top

# Absolute quality thresholds for Voyage embeddings
# These are used in hybrid confidence mode to assess absolute match quality
ABSOLUTE_STRONG = float(os.environ.get("AUGGIE_ABSOLUTE_STRONG", "0.50"))  # top 10%
ABSOLUTE_EXPECTED = float(os.environ.get("AUGGIE_ABSOLUTE_EXPECTED", "0.35"))  # typical good match
ABSOLUTE_WEAK = float(os.environ.get("AUGGIE_ABSOLUTE_WEAK", "0.25"))  # borderline
ANN_OVERFETCH = max(1, int(os.environ.get("AUGGIE_ANN_OVERFETCH", "3")))


@dataclass(frozen=True)
class HybridConfidence:
    """
    Hybrid confidence scoring combining relative and absolute signals.

    This addresses the problem where AI tools can't trust low absolute scores
    even when they represent the best available matches. By combining relative
    position with absolute quality signals, AI tools get actionable guidance.

    Attributes:
        level: Overall confidence level (high/medium/low/very_low).
        relative_pct: Percentage of top score (100 = top result).
        absolute_quality: Quality band (strong/expected_range/weak/very_weak).
        signal: Human-readable explanation of the scoring decision.
    """

    level: str
    relative_pct: float
    absolute_quality: str
    signal: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level,
            "relative_pct": round(self.relative_pct, 1),
            "absolute_quality": self.absolute_quality,
            "signal": self.signal,
        }


def get_absolute_quality(score: float) -> str:
    """
    Determine absolute quality band for a score.

    Calibrated for Voyage embeddings. Different models may need
    different thresholds (see AUGGIE_ABSOLUTE_* env vars).

    Args:
        score: Similarity score (0.0 to 1.0).

    Returns:
        Quality band: "strong", "expected_range", "weak", or "very_weak".
    """
    if score >= ABSOLUTE_STRONG:
        return "strong"
    elif score >= ABSOLUTE_EXPECTED:
        return "expected_range"
    elif score >= ABSOLUTE_WEAK:
        return "weak"
    else:
        return "very_weak"


def compute_hybrid_confidence(score: float, top_score: float, rank: int = 1) -> HybridConfidence:
    """
    Compute hybrid confidence combining relative position and absolute quality.

    This is the core of the improved confidence scoring. It addresses scenarios
    where correct results have low absolute scores (e.g., 0.03 for "export CSV").

    Logic:
        1. Compute relative_pct = (score / top_score) * 100
        2. Determine absolute_quality from fixed thresholds
        3. Combine into level:
           - "high" = relative >= 90% AND absolute in (strong, expected_range)
           - "medium" = relative >= 75% OR absolute == expected_range
           - "low" = relative >= 50% OR absolute == weak
           - "very_low" = everything else
        4. Generate human-readable signal

    Args:
        score: Similarity score for this result.
        top_score: Highest score in the result set.
        rank: 1-indexed rank position (1 = top result).

    Returns:
        HybridConfidence with level, relative_pct, absolute_quality, and signal.
    """
    # Handle edge cases
    if top_score <= 0:
        return HybridConfidence(
            level="very_low",
            relative_pct=0.0,
            absolute_quality="very_weak",
            signal="no_valid_top_score",
        )

    # Step 1: Compute relative percentage
    relative_pct = (score / top_score) * 100

    # Step 2: Determine absolute quality
    absolute_quality = get_absolute_quality(score)

    # Step 3: Combine into level
    is_relative_high = relative_pct >= RELATIVE_HIGH * 100  # >= 90%
    is_relative_medium = relative_pct >= RELATIVE_MEDIUM * 100  # >= 75%
    is_relative_low = relative_pct >= RELATIVE_LOW * 100  # >= 50%

    is_absolute_strong_or_expected = absolute_quality in ("strong", "expected_range")
    is_absolute_expected = absolute_quality == "expected_range"
    is_absolute_weak = absolute_quality == "weak"

    if is_relative_high and is_absolute_strong_or_expected:
        level = "high"
    elif is_relative_high and absolute_quality == "weak":
        # Close to top but weak absolute - still useful but cautious
        level = "medium"
    elif is_relative_high and absolute_quality == "very_weak":
        # Top result but very weak absolute - AI should know this is the best available
        # but may not be good enough
        level = "medium" if rank == 1 else "low"
    elif is_relative_medium or is_absolute_expected:
        level = "medium"
    elif is_relative_low or is_absolute_weak:
        level = "low"
    else:
        level = "very_low"

    # Step 4: Generate signal
    signal = _generate_confidence_signal(level, relative_pct, absolute_quality, rank, top_score)

    return HybridConfidence(
        level=level,
        relative_pct=relative_pct,
        absolute_quality=absolute_quality,
        signal=signal,
    )


def _generate_confidence_signal(
    level: str,
    relative_pct: float,
    absolute_quality: str,
    rank: int,
    top_score: float,
) -> str:
    """
    Generate a human-readable signal explaining the confidence decision.

    Args:
        level: Computed confidence level.
        relative_pct: Percentage of top score.
        absolute_quality: Absolute quality band.
        rank: 1-indexed rank position.
        top_score: Top score in result set.

    Returns:
        Signal string for AI/human interpretation.
    """
    if rank == 1:
        if absolute_quality == "strong":
            return "top_result_strong_match"
        elif absolute_quality == "expected_range":
            return "top_result_in_typical_range"
        elif absolute_quality == "weak":
            return "top_result_weak_absolute"
        else:
            return "top_result_very_weak_but_best_available"
    else:
        if relative_pct >= 90:
            return "close_to_top"
        elif relative_pct >= 75:
            if absolute_quality in ("strong", "expected_range"):
                return "good_alternative"
            else:
                return "close_to_top_but_weak_absolute"
        elif relative_pct >= 50:
            return "moderate_relevance"
        else:
            return "score_drop_from_top"


# =============================================================================
# Path Filtering (Phase 3 - Adoption Improvements)
# =============================================================================


@dataclass(frozen=True)
class PathFilter:
    """
    Configuration for path-based filtering of search results.

    Supports two types of patterns:
    - Simple patterns: *.py, src/* (can use SQL GLOB pre-filter)
    - Complex patterns: **/*.py, multiple patterns (requires fnmatch post-filter)

    Attributes:
        glob_patterns: List of glob patterns to include (None = all files).
        exclude_patterns: List of glob patterns to exclude (None = no exclusions).
    """

    glob_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None

    def is_empty(self) -> bool:
        """Check if no filtering is configured."""
        return not self.glob_patterns and not self.exclude_patterns


@dataclass(frozen=True)
class FilterStats:
    """
    Statistics about path filtering applied to search results.

    Attributes:
        candidates_before: Number of results before filtering.
        candidates_after: Number of results after filtering.
        removal_pct: Percentage of results removed by filter.
        method: Filtering method used ("sql_prefilter", "fnmatch_postfilter", "none").
        warning: Warning message if filter removed too many results.
    """

    candidates_before: int
    candidates_after: int
    removal_pct: float
    method: str
    warning: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
            "removal_pct": round(self.removal_pct, 1),
            "method": self.method,
            "warning": self.warning,
        }


def is_simple_pattern(pattern: str) -> bool:
    """
    Check if a glob pattern is simple enough for SQL GLOB pre-filter.

    Simple patterns:
    - *.py (single wildcard at start)
    - src/* (single directory level)
    - templates/*.j2 (single directory with extension)

    Complex patterns (need fnmatch):
    - **/*.py (recursive)
    - src/**/*.j2 (recursive in subdirectory)
    - *.{py,php} (brace expansion - not supported by SQLite GLOB)

    Args:
        pattern: Glob pattern to analyze.

    Returns:
        True if pattern can be handled by SQLite GLOB, False otherwise.
    """
    # Complex patterns that need fnmatch
    if "**" in pattern:
        return False
    if "{" in pattern or "}" in pattern:
        return False
    # SQLite GLOB uses * for any characters, ? for single char
    # fnmatch [abc] syntax works in SQLite too
    return True


def pattern_to_sql_glob(pattern: str) -> str:
    """
    Convert a simple glob pattern to SQLite GLOB syntax.

    SQLite GLOB:
    - * matches any sequence of characters
    - ? matches any single character
    - [...] matches character class

    Fnmatch patterns that translate directly to SQLite GLOB:
    - *.py → *.py
    - src/* → src/*
    - test_*.py → test_*.py

    Args:
        pattern: Simple glob pattern (must pass is_simple_pattern).

    Returns:
        Pattern suitable for SQLite GLOB clause.
    """
    # Most simple patterns work directly with SQLite GLOB
    # Just ensure we handle any edge cases
    return pattern


def _glob_to_regex(pattern: str) -> str:
    """
    Convert a glob pattern to a regex pattern.

    Handles:
    - ** for recursive matching (any path depth, including zero)
    - * for single path component matching (no /)
    - ? for single character matching (no /)
    - Escapes other regex special characters

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/*.js").

    Returns:
        Regex pattern string.
    """
    # Escape special regex chars first (except * and ?)
    regex = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            # Check for ** (recursive)
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                # ** followed by / means "any directory path or empty"
                if i + 2 < len(pattern) and pattern[i + 2] == "/":
                    regex += "(?:.*/)?"  # Optional path with trailing /
                    i += 3
                    continue
                else:
                    # ** at end or not followed by / - match anything
                    regex += ".*"
                    i += 2
                continue
            else:
                # Single * matches any chars except /
                regex += "[^/]*"
        elif c == "?":
            regex += "[^/]"
        elif c in ".^$+{}[]|()\\":
            regex += "\\" + c
        else:
            regex += c
        i += 1
    return f"^{regex}$"


def matches_any_pattern(path: str, patterns: list[str]) -> bool:
    """
    Check if a path matches any of the given glob patterns.

    Handles both absolute and relative paths by trying multiple match strategies:
    1. Match against full path
    2. Match against path without leading /
    3. Match against basename

    Args:
        path: File path to check (can be absolute or relative).
        patterns: List of glob patterns to match against.

    Returns:
        True if path matches at least one pattern, False otherwise.
    """
    # Prepare different path variants for matching
    path_variants = [path]
    if path.startswith("/"):
        path_variants.append(path[1:])  # Without leading /
    # Add basename for simple extension patterns
    basename = path.rsplit("/", 1)[-1] if "/" in path else path
    if basename not in path_variants:
        path_variants.append(basename)

    for pattern in patterns:
        if "**" in pattern:
            # Use regex for patterns with **
            regex = _glob_to_regex(pattern)
            for variant in path_variants:
                if re.match(regex, variant):
                    return True
        else:
            # Use fnmatch for simple patterns
            for variant in path_variants:
                if fnmatch.fnmatch(variant, pattern):
                    return True
    return False


def filter_hits_by_path(
    hits: list[Hit],
    path_filter: PathFilter,
    requested_n: int,
) -> tuple[list[Hit], FilterStats]:
    """
    Filter search results by path patterns.

    Uses fnmatch for flexible pattern matching. This is the post-filter
    approach used when SQL pre-filtering isn't possible.

    Args:
        hits: List of Hit objects to filter.
        path_filter: PathFilter configuration.
        requested_n: Number of results originally requested.

    Returns:
        Tuple of (filtered_hits, filter_stats).
    """
    if path_filter.is_empty():
        return hits, FilterStats(
            candidates_before=len(hits),
            candidates_after=len(hits),
            removal_pct=0.0,
            method="none",
        )

    before_count = len(hits)
    filtered = []

    # Compute relative paths from cwd for pattern matching
    cwd = os.getcwd()
    cwd_prefix = cwd + "/" if not cwd.endswith("/") else cwd

    for hit in hits:
        # Get relative path for matching
        path = hit.path

        # Compute relative path from cwd
        if path.startswith(cwd_prefix):
            rel_path = path[len(cwd_prefix) :]
        elif path.startswith("/"):
            # Fallback: use everything after first component for absolute paths
            # e.g., /home/user/repo/tests/foo.py -> tests/foo.py (try last 2+ components)
            parts = path.split("/")
            # Try progressively shorter paths to find a match
            rel_path = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        else:
            rel_path = path

        # Check glob patterns (include)
        if path_filter.glob_patterns:
            if not matches_any_pattern(rel_path, path_filter.glob_patterns):
                # Also try matching against full path
                if not matches_any_pattern(path, path_filter.glob_patterns):
                    continue

        # Check exclude patterns
        if path_filter.exclude_patterns:
            if matches_any_pattern(rel_path, path_filter.exclude_patterns):
                continue
            if matches_any_pattern(path, path_filter.exclude_patterns):
                continue

        filtered.append(hit)

    after_count = len(filtered)
    removal_pct = ((before_count - after_count) / before_count * 100) if before_count > 0 else 0.0

    # Generate warning if too many results removed
    warning = None
    if removal_pct > 50:
        warning = (
            f"Filter removed {before_count - after_count} of {before_count} results. "
            f"Consider increasing -n or broadening glob."
        )

    return filtered, FilterStats(
        candidates_before=before_count,
        candidates_after=after_count,
        removal_pct=removal_pct,
        method="fnmatch_postfilter",
        warning=warning,
    )


def get_confidence_level(score: float) -> str:
    """
    Convert numeric score to human-readable confidence level (absolute mode).

    Uses configurable thresholds from environment variables:
        AUGGIE_CONFIDENCE_HIGH: Threshold for "high" (default: 0.85)
        AUGGIE_CONFIDENCE_MEDIUM: Threshold for "medium" (default: 0.70)
        AUGGIE_CONFIDENCE_LOW: Threshold for "low" (default: 0.50)

    Args:
        score: Similarity score (0.0 to 1.0).

    Returns:
        Confidence level: "high", "medium", "low", or "very_low".
    """
    if score >= CONFIDENCE_HIGH:
        return "high"
    elif score >= CONFIDENCE_MEDIUM:
        return "medium"
    elif score >= CONFIDENCE_LOW:
        return "low"
    else:
        return "very_low"


def get_relative_confidence(score: float, top_score: float) -> str:
    """
    Convert numeric score to confidence level relative to top result.

    Instead of absolute thresholds, this compares the score to the best
    match in the result set. This accounts for the fact that cosine
    similarity scores for text embeddings cluster around 0.3-0.5, making
    absolute thresholds misleading.

    Uses configurable thresholds from environment variables:
        AUGGIE_RELATIVE_HIGH: Fraction of top score for "high" (default: 0.90)
        AUGGIE_RELATIVE_MEDIUM: Fraction of top score for "medium" (default: 0.75)
        AUGGIE_RELATIVE_LOW: Fraction of top score for "low" (default: 0.50)

    Args:
        score: Similarity score for this result.
        top_score: Highest score in the result set.

    Returns:
        Confidence level: "high", "medium", "low", or "very_low".

    Example:
        If top_score=0.45 and score=0.42:
        - ratio = 0.42/0.45 = 0.93 (93% of top)
        - Since 0.93 >= 0.90, this is "high" confidence

        This is more meaningful than absolute scoring, where 0.42
        might appear as "low" despite being nearly as good as the
        best match.
    """
    if top_score <= 0:
        return "very_low"

    ratio = score / top_score

    if ratio >= RELATIVE_HIGH:
        return "high"
    elif ratio >= RELATIVE_MEDIUM:
        return "medium"
    elif ratio >= RELATIVE_LOW:
        return "low"
    else:
        return "very_low"


def assign_confidence(
    score: float,
    top_score: float | None = None,
    rank: int = 1,
) -> tuple[str, HybridConfidence | None]:
    """
    Assign confidence level based on current mode.

    Now supports hybrid mode which combines relative and absolute signals.
    This gives AI tools actionable guidance even when absolute scores are low.

    Args:
        score: Similarity score for this result.
        top_score: Highest score in result set (required for relative/hybrid mode).
        rank: 1-indexed rank position (1 = top result).

    Returns:
        Tuple of (level, details):
        - level: Confidence string ("high", "medium", "low", or "very_low")
        - details: HybridConfidence object (None if using absolute mode)
    """
    # Always compute hybrid confidence when top_score is available
    # This gives the most useful information to AI tools
    if top_score is not None and top_score > 0:
        hybrid = compute_hybrid_confidence(score, top_score, rank)
        return hybrid.level, hybrid
    else:
        # Fallback to absolute mode (legacy behavior)
        return get_confidence_level(score), None


@dataclass(frozen=True)
class Hit:
    """
    A search result representing a matching code chunk.

    Attributes:
        score: Cosine similarity score (0.0 to 1.0, higher is better).
        path: Absolute path to the source file.
        start_line: First line number of the chunk (1-indexed).
        end_line: Last line number of the chunk (inclusive).
        text: The actual text content of the chunk.
        chunk_id: Internal database ID for the chunk.
        chunk_index: Index of this chunk within its file (0-indexed).
        confidence: Human-readable confidence level based on score.
        confidence_details: Detailed hybrid confidence (None for legacy mode).
    """

    score: float
    path: str
    start_line: int
    end_line: int
    text: str
    chunk_id: int
    chunk_index: int
    confidence: str
    confidence_details: HybridConfidence | None = None


def _escape_fts5_query(q: str) -> str:
    """
    Escape special characters for FTS5 query.

    FTS5 has special syntax characters that need escaping.
    We quote each term to handle special characters.

    Args:
        q: Raw query string.

    Returns:
        Escaped query safe for FTS5 MATCH.
    """
    # Split into words and quote each term
    # This handles underscores, dots, and other special chars
    terms = q.split()
    return " ".join(f'"{term}"' for term in terms)


def _rrf_score(
    semantic_rank: int | None,
    fts_rank: int | None,
    k: int = 60,
) -> float:
    """
    Compute Reciprocal Rank Fusion (RRF) score.

    RRF combines results from multiple ranking systems by using their
    rank positions rather than raw scores. This is more robust because:
    - Scores from different systems have different distributions
    - Ranks are comparable across systems
    - No need to tune weighting parameters

    Formula: RRF(d) = sum(1 / (k + rank_i)) for each ranking system i

    The k parameter (default 60) controls how much weight is given to
    lower-ranked results. Higher k = smoother distribution.

    Reference: Cormack, Clarke, Buettcher. "Reciprocal Rank Fusion
    outperforms Condorcet and individual Rank Learning Methods" (SIGIR 2009)

    Args:
        semantic_rank: 1-indexed rank from semantic search (None if not ranked).
        fts_rank: 1-indexed rank from full-text search (None if not ranked).
        k: Rank constant (default: 60, standard in literature).

    Returns:
        Combined RRF score (higher is better).

    Example:
        >>> _rrf_score(1, 3)  # Top semantic, 3rd in FTS
        0.032258...  # 1/(60+1) + 1/(60+3)
        >>> _rrf_score(10, None)  # Only in semantic results
        0.014285...  # 1/(60+10)
    """
    score = 0.0
    if semantic_rank is not None:
        score += 1.0 / (k + semantic_rank)
    if fts_rank is not None:
        score += 1.0 / (k + fts_rank)
    return score


def _fulltext_search(
    con,
    q: str,
    top_k: int,
) -> tuple[dict[int, float], dict[int, int]]:
    """
    Perform FTS5 full-text search and return normalized BM25 scores and ranks.

    Args:
        con: Database connection.
        q: Search query string.
        top_k: Maximum number of results.

    Returns:
        Tuple of (scores, ranks):
        - scores: Dict mapping chunk_id to normalized score (0.0 to 1.0).
        - ranks: Dict mapping chunk_id to 1-indexed rank position.
    """
    # Escape query for FTS5 syntax
    escaped_q = _escape_fts5_query(q)

    # FTS5 bm25() returns negative scores (lower is better)
    # We need to normalize them to positive scores (higher is better)
    try:
        rows = con.execute(
            """SELECT rowid, bm25(chunks_fts) as rank
               FROM chunks_fts
               WHERE text MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (escaped_q, top_k * 2),  # Fetch extra for hybrid merging
        ).fetchall()
    except Exception:
        # If FTS5 query fails for any reason, return empty results
        return {}, {}

    if not rows:
        return {}, {}

    # Normalize BM25 scores to 0.0-1.0 range
    # BM25 scores are negative, so we invert and normalize
    scores = [-r[1] for r in rows]  # Make positive
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score if max_score != min_score else 1.0

    score_dict = {row[0]: ((-row[1]) - min_score) / score_range for row in rows}
    # Ranks are 1-indexed (1 = best match)
    rank_dict = {row[0]: i + 1 for i, row in enumerate(rows)}

    return score_dict, rank_dict


def _l2_distance_to_cosine(distance: float) -> float:
    """
    Convert Euclidean distance on unit vectors back to cosine similarity.

    Query and chunk embeddings are L2-normalized in embed.py. Lance defaults
    to L2 distance, so for unit vectors:
        ||u - v||^2 = 2 - 2cos(theta)
        cos(theta) = 1 - (distance^2 / 2)
    """
    score = 1.0 - ((distance * distance) / 2.0)
    return max(-1.0, min(1.0, score))


def _fetch_chunk_rows(
    con,
    chunk_ids: list[int],
    branch: str,
) -> dict[int, tuple[int, int, str, int, int, str]]:
    """Fetch chunk metadata for chunk ids on the active branch."""
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"""SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE c.id IN ({placeholders}) AND f.branch = ?""",
        [*chunk_ids, branch],
    ).fetchall()
    return {
        int(row[0]): (
            int(row[0]),
            int(row[1]),
            str(row[2]),
            int(row[3]),
            int(row[4]),
            str(row[5]),
        )
        for row in rows
    }


def _semantic_ann_search(
    con,
    db_path: Path,
    query_vector: bytes,
    top_k: int,
    branch: str,
) -> list[tuple[float, int, int, str, int, int, str]]:
    """
    Fetch semantic candidates from Lance ANN and hydrate them from SQLite.
    """
    lance_store = LanceStore(get_lance_dir(db_path))
    total_vectors = lance_store.count()
    if total_vectors == 0:
        return []

    limit = min(total_vectors, max(top_k * ANN_OVERFETCH, top_k))

    while True:
        ann_hits = lance_store.ann_query(query_vector, limit)
        chunk_rows = _fetch_chunk_rows(con, [hit.chunk_id for hit in ann_hits], branch)

        semantic_results = []
        for ann_hit in ann_hits:
            row = chunk_rows.get(ann_hit.chunk_id)
            if row is None:
                continue
            chunk_id, chunk_idx, path, start_line, end_line, text = row
            semantic_results.append(
                (
                    _l2_distance_to_cosine(ann_hit.distance),
                    chunk_id,
                    chunk_idx,
                    path,
                    start_line,
                    end_line,
                    text,
                )
            )

        exhausted = limit >= total_vectors or len(ann_hits) < limit
        if len(semantic_results) >= top_k or exhausted:
            return semantic_results

        limit = min(total_vectors, max(limit * 2, top_k))


def query(
    db_path: Path,
    q: str,
    top_k: int = 10,
    model: str = "voyage-code-3",
    dimensions: int | None = None,
    mode: SearchMode | None = None,
    no_cache: bool = False,
    branch: str | None = None,
) -> tuple[list[Hit], bool]:
    """
    Perform search against the indexed codebase.

    Supports three search modes:
    - semantic: Embedding similarity only (original behavior)
    - fulltext: SQLite FTS5 keyword matching only
    - hybrid: Combined score (default) - best of both worlds

    Args:
        db_path: Path to the SQLite database.
        q: Natural language query string.
        top_k: Maximum number of results to return (default: 10).
        model: Voyage embedding model. Must match the model used during
            indexing for meaningful results.
        dimensions: Embedding dimensions. Must match indexing dimensions.
        mode: Search mode (semantic, fulltext, hybrid). Default from
            AUGGIE_SEARCH_MODE env var or "hybrid".
        no_cache: Disable L1/L2 caching for this query (default: False).
        branch: Git branch to search (None for auto-detect from current).
            Only files indexed on this branch will be searched.

    Returns:
        Tuple of (hits, fts_available):
        - hits: List of Hit objects sorted by descending score.
        - fts_available: Whether FTS5 was available for this search.

    Note:
        Uses numpy for similarity calculations if available, falling
        back to pure Python otherwise. Numpy is ~10x faster for large
        indexes.

        If FTS5 is unavailable and mode is hybrid/fulltext, falls back
        to semantic search and returns fts_available=False.

        Caching: Query embeddings are cached in L1 (TTL-based, 24h default).
        Set no_cache=True or AUGGIE_CACHE_DISABLED=1 to bypass.

    Example:
        >>> hits, fts_ok = query(Path(".auggie/index.sqlite"), "database connection")
        >>> for h in hits[:3]:
        ...     print(f"{h.path}:{h.start_line} (score={h.score:.3f})")
    """
    if mode is None:
        mode = DEFAULT_SEARCH_MODE

    con = connect(db_path, init_fts=False)
    cache_con = None

    def _finish(result_hits: list[Hit], result_fts_available: bool) -> tuple[list[Hit], bool]:
        if cache_con is not None:
            cache_con.close()
        con.close()
        return result_hits, result_fts_available

    # Resolve branch - auto-detect if not specified
    if branch is None:
        from .commands._common import get_current_branch

        branch = get_current_branch()

    # Cache setup (lazy initialization)
    cache_enabled = not no_cache and not os.environ.get("AUGGIE_CACHE_DISABLED")

    # Check for mixed dimensions (corrupted index)
    distinct_dims = con.execute("SELECT DISTINCT dim FROM chunks").fetchall()
    if len(distinct_dims) > 1:
        dims_list = sorted([d[0] for d in distinct_dims])
        raise ValueError(
            f"Corrupted index: mixed dimensions detected ({dims_list}). "
            f"This can happen if --refresh was run with a different model. "
            f"Run 'auggie reset -f && auggie index .' to rebuild."
        )

    # Check index model/dimensions before querying
    index_info = con.execute("SELECT model, dim FROM chunks LIMIT 1").fetchone()
    if index_info is None:
        return _finish([], True)  # Empty index

    index_model, index_dim = index_info

    # Check FTS5 availability
    fts_available = has_fts5(con)

    # Determine effective mode (fallback to semantic if FTS5 unavailable)
    effective_mode = mode
    if mode in ("hybrid", "fulltext") and not fts_available:
        effective_mode = "semantic"

    # Full-text only mode
    if effective_mode == "fulltext":
        fts_scores, _ = _fulltext_search(con, q, top_k)
        if not fts_scores:
            return _finish([], fts_available)

        # Fetch chunk details for FTS matches (filtered by branch)
        chunk_ids = list(fts_scores.keys())
        placeholders = ",".join("?" * len(chunk_ids))
        rows = con.execute(
            f"""SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.id IN ({placeholders}) AND f.branch = ?""",
            [*chunk_ids, branch],
        ).fetchall()

        # Build hits with scores, sorted by score descending
        scored_rows = sorted(
            [(fts_scores[row[0]], row) for row in rows],
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        # Get top score for relative confidence
        top_score = scored_rows[0][0] if scored_rows else 0.0

        hits = []
        for rank, (score, row) in enumerate(scored_rows, start=1):
            level, details = assign_confidence(score, top_score, rank)
            hits.append(
                Hit(
                    score=score,
                    path=row[2],
                    start_line=int(row[3]),
                    end_line=int(row[4]),
                    text=row[5],
                    chunk_id=int(row[0]),
                    chunk_index=int(row[1]),
                    confidence=level,
                    confidence_details=details,
                )
            )
        return _finish(hits, fts_available)

    # Semantic search (needed for semantic and hybrid modes)
    # Embed the query (with L1 cache if enabled)
    embedding_key = None

    if cache_enabled:
        from .cache import (
            connect_cache,
            get_cache_path,
            get_query_embedding,
            log_cache_event,
            set_query_embedding,
        )

        cache_path = get_cache_path(db_path)
        cache_con = connect_cache(cache_path)
        # Check L1 cache
        result = get_query_embedding(cache_con, q, model, dimensions, None)
        if result.hit:
            # Cache hit - use cached embedding
            q_blob = [result.data]
            # Dimension = number of float32 values = byte_length / 4
            q_dim = len(result.data) // 4 if result.data else 0
            log_cache_event(cache_con, "L1", "hit", time_saved_ms=result.time_saved_ms)
        else:
            # Cache miss - embed and store
            q_blob, q_dim = embed_texts([q], model=model, dimensions=dimensions)
            set_query_embedding(cache_con, q, model, dimensions, None, q_blob[0])
            log_cache_event(cache_con, "L1", "miss")
    else:
        # No caching - embed directly
        q_blob, q_dim = embed_texts([q], model=model, dimensions=dimensions)

    # Validate dimensions match
    if q_dim != index_dim:
        # Reverse lookup for model aliases (prefer shortest alias)
        alias_lookup: dict[str, str] = {}
        for alias, full_name in MODEL_ALIASES.items():
            if full_name not in alias_lookup or len(alias) < len(alias_lookup[full_name]):
                alias_lookup[full_name] = alias
        index_alias = alias_lookup.get(index_model, index_model)
        query_alias = alias_lookup.get(model, model)
        raise ValueError(
            f"Dimension mismatch: query uses {q_dim}D ({model}) "
            f"but index was built with {index_dim}D ({index_model}). "
            f"Use -m {index_alias} or reindex with -m {query_alias}."
        )

    # L2 Cache: Check for cached search results
    if cache_enabled and cache_con is not None:
        from .cache import cache_key, get_search_results, log_cache_event, set_search_results

        embedding_key = cache_key(q, model, dimensions, "voyage")

        # Check L2 cache (db_version validated inside get_search_results)
        # Branch is included in cache key to prevent cross-branch cache pollution
        l2_result = get_search_results(cache_con, con, embedding_key, effective_mode, top_k, branch)
        if l2_result.hit:
            # Cache hit - reconstruct Hits from cached chunk_ids and scores
            cached_results = l2_result.data  # List of (chunk_id, score)
            chunk_ids = [r[0] for r in cached_results]

            if chunk_ids:
                placeholders = ",".join("?" * len(chunk_ids))
                rows = con.execute(
                    f"""SELECT c.id, c.chunk_index, f.path, c.start_line, c.end_line, c.text
                        FROM chunks c
                        JOIN files f ON f.id = c.file_id
                        WHERE c.id IN ({placeholders}) AND f.branch = ?""",
                    [*chunk_ids, branch],
                ).fetchall()

                # Build lookup for chunk details
                chunk_lookup = {row[0]: row for row in rows}

                # Get top score for relative confidence
                top_score = cached_results[0][1] if cached_results else 0.0

                hits = []
                for rank, (chunk_id, score) in enumerate(cached_results, start=1):
                    if chunk_id in chunk_lookup:
                        row = chunk_lookup[chunk_id]
                        level, details = assign_confidence(score, top_score, rank)
                        hits.append(
                            Hit(
                                score=score,
                                path=row[2],
                                start_line=int(row[3]),
                                end_line=int(row[4]),
                                text=row[5],
                                chunk_id=int(row[0]),
                                chunk_index=int(row[1]),
                                confidence=level,
                                confidence_details=details,
                            )
                        )

                log_cache_event(cache_con, "L2", "hit", time_saved_ms=l2_result.time_saved_ms)
                return _finish(
                    hits,
                    l2_result.fts_available
                    if l2_result.fts_available is not None
                    else fts_available,
                )

            # Empty cached results
            log_cache_event(cache_con, "L2", "hit", time_saved_ms=l2_result.time_saved_ms)
            return _finish([], fts_available)

    # Get FTS scores and ranks if in hybrid mode
    fts_scores: dict[int, float] = {}
    fts_ranks: dict[int, int] = {}
    if effective_mode == "hybrid" and fts_available:
        fts_scores, fts_ranks = _fulltext_search(con, q, top_k * 2)

    semantic_target = top_k * 2 if effective_mode == "hybrid" else top_k
    semantic_results = _semantic_ann_search(con, db_path, q_blob[0], semantic_target, branch)

    # Build semantic rank lookup (1-indexed). ANN results are already ordered.
    semantic_ranks: dict[int, int] = {}
    for rank, (_, chunk_id, *_rest) in enumerate(semantic_results, start=1):
        semantic_ranks[chunk_id] = rank
    semantic_scores = {chunk_id: score for score, chunk_id, *_rest in semantic_results}

    # Step 3: Compute final scores based on fusion method
    scored_results: list[tuple[float, int, int, str, int, int, str]] = []

    if effective_mode == "hybrid" and (fts_scores or fts_ranks):
        # Hybrid mode: combine semantic and FTS results
        use_rrf = FUSION_METHOD == "rrf"

        # Collect all chunk IDs that appear in either result set
        all_chunk_ids = set(semantic_ranks.keys())
        if fts_ranks:
            all_chunk_ids.update(fts_ranks.keys())

        # Build lookup for chunk details
        chunk_details = {
            chunk_id: (chunk_idx, path, sl, el, text)
            for (_, chunk_id, chunk_idx, path, sl, el, text) in semantic_results
        }

        for chunk_id in all_chunk_ids:
            if chunk_id not in chunk_details:
                continue  # Skip if we don't have chunk details

            chunk_idx, path, sl, el, text = chunk_details[chunk_id]
            sem_rank = semantic_ranks.get(chunk_id)
            fts_rank = fts_ranks.get(chunk_id)

            if use_rrf:
                # RRF fusion: combine by ranks
                score = _rrf_score(sem_rank, fts_rank, k=RRF_K)
            else:
                # Alpha fusion: combine by scores (legacy)
                sem_score = semantic_scores.get(chunk_id, 0.0)
                fts_score = fts_scores.get(chunk_id, 0.0)
                score = HYBRID_ALPHA * sem_score + (1 - HYBRID_ALPHA) * fts_score

            scored_results.append((score, chunk_id, chunk_idx, path, sl, el, text))
    else:
        # Semantic-only mode: use semantic scores directly
        scored_results = list(semantic_results)

    # Sort by final score descending and take top-k
    scored_results.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_results[:top_k]

    # Get top score for relative confidence calculation
    top_score = top_results[0][0] if top_results else 0.0

    # Build final Hit objects with confidence
    hits = []
    for rank, (score, chunk_id, chunk_idx, path, sl, el, text) in enumerate(top_results, start=1):
        level, details = assign_confidence(score, top_score, rank)
        hits.append(
            Hit(
                score=score,
                path=path,
                start_line=sl,
                end_line=el,
                text=text,
                chunk_id=chunk_id,
                chunk_index=chunk_idx,
                confidence=level,
                confidence_details=details,
            )
        )

    # L2 Cache: Store search results
    if cache_enabled and cache_con is not None:
        from .cache import cache_key, log_cache_event, set_search_results

        embedding_key = cache_key(q, model, dimensions, "voyage")

        # Store (chunk_id, score) pairs (branch-scoped cache key)
        results_to_cache = [(h.chunk_id, h.score) for h in hits]
        set_search_results(
            cache_con,
            con,
            embedding_key,
            effective_mode,
            top_k,
            results_to_cache,
            fts_available,
            branch,
        )
        log_cache_event(cache_con, "L2", "miss")

    return _finish(hits, fts_available)
