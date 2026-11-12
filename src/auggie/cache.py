"""
Multi-level output caching for auggie.

Storage: Separate SQLite file (.auggie/cache.sqlite)
- L1: Query embeddings (TTL-based, 24h default)
- L2: Search results (db_version invalidated)
- L3: Rerank results (content-addressable)

Environment Variables:
    AUGGIE_CACHE_DISABLED: Set to '1' to disable all caching
    AUGGIE_CACHE_L1_TTL: L1 TTL in seconds (default: 86400 = 24h)
    AUGGIE_CACHE_L1_MAX: Max L1 entries (default: 1000)
    AUGGIE_CACHE_L2_MAX: Max L2 entries (default: 500)
    AUGGIE_CACHE_L3_MAX: Max L3 entries (default: 200)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Configuration from environment
L1_TTL_SECONDS = int(os.environ.get("AUGGIE_CACHE_L1_TTL", str(24 * 60 * 60)))  # 24h
L1_MAX_ENTRIES = int(os.environ.get("AUGGIE_CACHE_L1_MAX", "1000"))
L2_MAX_ENTRIES = int(os.environ.get("AUGGIE_CACHE_L2_MAX", "500"))
L3_MAX_ENTRIES = int(os.environ.get("AUGGIE_CACHE_L3_MAX", "200"))


# Cache schema
CACHE_SCHEMA = """
-- Pragma settings for CLI performance
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-8000;

-- L1: Query Embedding Cache (TTL-based)
CREATE TABLE IF NOT EXISTS query_embeddings (
  cache_key TEXT PRIMARY KEY,
  query_text TEXT NOT NULL,
  model TEXT NOT NULL,
  base_url TEXT,
  dimensions INTEGER,
  embedding BLOB NOT NULL,
  created_at REAL NOT NULL,
  hit_count INTEGER DEFAULT 0
);

-- L2: Search Results Cache (db_version invalidated)
CREATE TABLE IF NOT EXISTS search_results (
  cache_key TEXT PRIMARY KEY,
  embedding_key TEXT NOT NULL,
  mode TEXT NOT NULL,
  top_k INTEGER NOT NULL,
  db_version INTEGER NOT NULL,
  results TEXT NOT NULL,
  fts_available INTEGER NOT NULL,
  created_at REAL NOT NULL,
  hit_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_search_db_version ON search_results(db_version);

-- L3: Rerank Results Cache (content-addressable)
CREATE TABLE IF NOT EXISTS rerank_results (
  cache_key TEXT PRIMARY KEY,
  query_text TEXT NOT NULL,
  chunk_content_hash TEXT NOT NULL,
  rerank_model TEXT NOT NULL,
  rerank_top INTEGER NOT NULL,
  results TEXT NOT NULL,
  created_at REAL NOT NULL,
  hit_count INTEGER DEFAULT 0
);

-- Stats tracking for cache effect reporting
CREATE TABLE IF NOT EXISTS cache_stats (
  id INTEGER PRIMARY KEY,
  timestamp REAL NOT NULL,
  level TEXT NOT NULL,
  event TEXT NOT NULL,
  time_saved_ms INTEGER,
  details TEXT
);
CREATE INDEX IF NOT EXISTS idx_stats_timestamp ON cache_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_stats_level ON cache_stats(level);
"""


@dataclass
class CacheResult:
    """Result from cache lookup with metadata."""

    hit: bool
    data: Any = None
    time_saved_ms: int = 0
    fts_available: bool | None = None


def get_cache_path(index_path: Path) -> Path:
    """
    Get cache.sqlite path from index.sqlite path.

    Args:
        index_path: Path to the index.sqlite file.

    Returns:
        Path to cache.sqlite in the same directory.
    """
    return index_path.parent / "cache.sqlite"


def connect_cache(cache_path: Path) -> sqlite3.Connection:
    """
    Connect to cache database, creating if needed.

    Args:
        cache_path: Path to cache.sqlite file.

    Returns:
        SQLite connection with schema initialized.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(cache_path))

    # Initialize schema
    for statement in CACHE_SCHEMA.split(";"):
        statement = statement.strip()
        if statement:
            try:
                con.execute(statement)
            except sqlite3.OperationalError:
                pass  # Index might already exist

    con.commit()
    return con


def cache_key(*components) -> str:
    """
    Compute SHA-256 cache key from components.

    Args:
        *components: Arbitrary values to hash together.

    Returns:
        Hex digest of SHA-256 hash.
    """
    return hashlib.sha256("|".join(str(c) for c in components).encode()).hexdigest()


# =============================================================================
# L1: Query Embedding Cache
# =============================================================================


def get_query_embedding(
    cache_con: sqlite3.Connection,
    query: str,
    model: str,
    dimensions: int | None,
    base_url: str | None,
) -> CacheResult:
    """
    Look up cached query embedding.

    Args:
        cache_con: Cache database connection.
        query: Query text.
        model: Embedding model name.
        dimensions: Embedding dimensions.
        base_url: Provider scope for cache partitioning.

    Returns:
        CacheResult with hit=True and embedding bytes if found and not expired.
    """
    key = cache_key(query, model, dimensions, base_url or "voyage")

    row = cache_con.execute(
        "SELECT embedding, created_at FROM query_embeddings WHERE cache_key = ?",
        (key,),
    ).fetchone()

    if row is None:
        return CacheResult(hit=False)

    embedding, created_at = row

    # Check TTL
    if time.time() - created_at > L1_TTL_SECONDS:
        # Expired - don't return, will be cleaned up later
        return CacheResult(hit=False)

    # Update hit count
    cache_con.execute(
        "UPDATE query_embeddings SET hit_count = hit_count + 1 WHERE cache_key = ?",
        (key,),
    )
    cache_con.commit()

    return CacheResult(hit=True, data=embedding, time_saved_ms=350)


def set_query_embedding(
    cache_con: sqlite3.Connection,
    query: str,
    model: str,
    dimensions: int | None,
    base_url: str | None,
    embedding: bytes,
) -> None:
    """
    Store query embedding in cache.

    Args:
        cache_con: Cache database connection.
        query: Query text.
        model: Embedding model name.
        dimensions: Embedding dimensions.
        base_url: Provider scope for cache partitioning.
        embedding: Embedding bytes to store.
    """
    key = cache_key(query, model, dimensions, base_url or "voyage")

    cache_con.execute(
        """INSERT OR REPLACE INTO query_embeddings
           (cache_key, query_text, model, base_url, dimensions, embedding, created_at, hit_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
        (key, query, model, base_url, dimensions, embedding, time.time()),
    )
    cache_con.commit()

    # Evict if over limit
    evict_lru(cache_con, "query_embeddings", L1_MAX_ENTRIES)


# =============================================================================
# L2: Search Results Cache
# =============================================================================


def get_db_version(index_con: sqlite3.Connection) -> int:
    """
    Get current db_version from index metadata.

    Args:
        index_con: Index database connection.

    Returns:
        Current db_version (0 if not set).
    """
    try:
        row = index_con.execute(
            "SELECT value FROM index_metadata WHERE key = 'db_version'"
        ).fetchone()
        if row:
            return int(row[0])
    except sqlite3.OperationalError:
        pass  # Table doesn't exist
    return 0


def increment_db_version(index_con: sqlite3.Connection) -> int:
    """
    Increment db_version atomically.

    Args:
        index_con: Index database connection.

    Returns:
        New db_version value.
    """
    current = get_db_version(index_con)
    new_version = current + 1

    index_con.execute(
        """INSERT OR REPLACE INTO index_metadata (key, value, updated_at)
           VALUES ('db_version', ?, datetime('now'))""",
        (str(new_version),),
    )
    index_con.commit()

    return new_version


def get_search_results(
    cache_con: sqlite3.Connection,
    index_con: sqlite3.Connection,
    embedding_key: str,
    mode: str,
    top_k: int,
    branch: str = "default",
) -> CacheResult:
    """
    Look up cached search results, checking db_version.

    Args:
        cache_con: Cache database connection.
        index_con: Index database connection (for db_version check).
        embedding_key: Key from L1 embedding cache.
        mode: Search mode (semantic, fulltext, hybrid).
        top_k: Number of results requested.
        branch: Git branch for this search (cache is branch-scoped).

    Returns:
        CacheResult with hit=True and results if found and db_version matches.
    """
    key = cache_key(embedding_key, mode, top_k, branch)
    current_version = get_db_version(index_con)

    row = cache_con.execute(
        """SELECT results, fts_available, db_version FROM search_results
           WHERE cache_key = ?""",
        (key,),
    ).fetchone()

    if row is None:
        return CacheResult(hit=False)

    results_json, fts_available, cached_version = row

    # Check db_version
    if cached_version != current_version:
        return CacheResult(hit=False)

    # Update hit count
    cache_con.execute(
        "UPDATE search_results SET hit_count = hit_count + 1 WHERE cache_key = ?",
        (key,),
    )
    cache_con.commit()

    results = json.loads(results_json)
    # Convert back to list of tuples
    results = [tuple(r) for r in results]

    return CacheResult(
        hit=True,
        data=results,
        time_saved_ms=180,
        fts_available=bool(fts_available),
    )


def set_search_results(
    cache_con: sqlite3.Connection,
    index_con: sqlite3.Connection,
    embedding_key: str,
    mode: str,
    top_k: int,
    results: list,
    fts_available: bool,
    branch: str = "default",
) -> None:
    """
    Store search results with current db_version.

    Args:
        cache_con: Cache database connection.
        index_con: Index database connection (for db_version).
        embedding_key: Key from L1 embedding cache.
        mode: Search mode.
        top_k: Number of results.
        results: List of (chunk_id, score) tuples.
        fts_available: Whether FTS5 was available.
        branch: Git branch for this search (cache is branch-scoped).
    """
    key = cache_key(embedding_key, mode, top_k, branch)
    current_version = get_db_version(index_con)

    cache_con.execute(
        """INSERT OR REPLACE INTO search_results
           (cache_key, embedding_key, mode, top_k, db_version, results, fts_available, created_at, hit_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            key,
            embedding_key,
            mode,
            top_k,
            current_version,
            json.dumps(results),
            1 if fts_available else 0,
            time.time(),
        ),
    )
    cache_con.commit()

    # Evict if over limit
    evict_lru(cache_con, "search_results", L2_MAX_ENTRIES)


# =============================================================================
# L3: Rerank Results Cache
# =============================================================================


def get_rerank_results(
    cache_con: sqlite3.Connection,
    query: str,
    chunk_hashes: list[str],
    rerank_model: str,
    rerank_top: int,
) -> CacheResult:
    """
    Look up cached rerank results (content-addressable).

    Args:
        cache_con: Cache database connection.
        query: Query text.
        chunk_hashes: List of chunk text hashes (will be sorted for key).
        rerank_model: Reranker model name.
        rerank_top: Number of candidates reranked.

    Returns:
        CacheResult with hit=True and scores if found.
    """
    # Sort hashes for content-addressable key
    sorted_hashes = sorted(chunk_hashes)
    content_hash = hashlib.sha256("|".join(sorted_hashes).encode()).hexdigest()
    key = cache_key(query, content_hash, rerank_model, rerank_top)

    row = cache_con.execute(
        "SELECT results FROM rerank_results WHERE cache_key = ?",
        (key,),
    ).fetchone()

    if row is None:
        return CacheResult(hit=False)

    # Update hit count
    cache_con.execute(
        "UPDATE rerank_results SET hit_count = hit_count + 1 WHERE cache_key = ?",
        (key,),
    )
    cache_con.commit()

    results = json.loads(row[0])
    # Convert back to list of tuples
    results = [tuple(r) for r in results]

    return CacheResult(hit=True, data=results, time_saved_ms=3200)


def set_rerank_results(
    cache_con: sqlite3.Connection,
    query: str,
    chunk_hashes: list[str],
    rerank_model: str,
    rerank_top: int,
    results: list,
) -> None:
    """
    Store rerank results.

    Args:
        cache_con: Cache database connection.
        query: Query text.
        chunk_hashes: List of chunk text hashes.
        rerank_model: Reranker model name.
        rerank_top: Number of candidates reranked.
        results: List of (chunk_id, score) tuples.
    """
    # Sort hashes for content-addressable key
    sorted_hashes = sorted(chunk_hashes)
    content_hash = hashlib.sha256("|".join(sorted_hashes).encode()).hexdigest()
    key = cache_key(query, content_hash, rerank_model, rerank_top)

    cache_con.execute(
        """INSERT OR REPLACE INTO rerank_results
           (cache_key, query_text, chunk_content_hash, rerank_model, rerank_top, results, created_at, hit_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
        (
            key,
            query,
            content_hash,
            rerank_model,
            rerank_top,
            json.dumps(results),
            time.time(),
        ),
    )
    cache_con.commit()

    # Evict if over limit
    evict_lru(cache_con, "rerank_results", L3_MAX_ENTRIES)


# =============================================================================
# Stats Tracking
# =============================================================================


def log_cache_event(
    cache_con: sqlite3.Connection,
    level: str,
    event: str,
    time_saved_ms: int = 0,
    details: dict | None = None,
) -> None:
    """
    Log cache hit/miss/evict for reporting.

    Args:
        cache_con: Cache database connection.
        level: Cache level ('L1', 'L2', 'L3').
        event: Event type ('hit', 'miss', 'evict', 'expire').
        time_saved_ms: Estimated time saved on hit.
        details: Optional JSON details.
    """
    cache_con.execute(
        """INSERT INTO cache_stats (timestamp, level, event, time_saved_ms, details)
           VALUES (?, ?, ?, ?, ?)""",
        (
            time.time(),
            level,
            event,
            time_saved_ms,
            json.dumps(details) if details else None,
        ),
    )
    cache_con.commit()


def get_cache_report(cache_con: sqlite3.Connection, since_hours: int = 24) -> dict:
    """
    Generate cache effectiveness report.

    Args:
        cache_con: Cache database connection.
        since_hours: Number of hours to look back.

    Returns:
        Dict with stats per level: hits, misses, hit_rate, time_saved_ms.
    """
    cutoff = time.time() - (since_hours * 60 * 60)

    report = {}

    for level in ["L1", "L2", "L3"]:
        hits = cache_con.execute(
            """SELECT COUNT(*), COALESCE(SUM(time_saved_ms), 0)
               FROM cache_stats
               WHERE level = ? AND event = 'hit' AND timestamp > ?""",
            (level, cutoff),
        ).fetchone()

        misses = cache_con.execute(
            """SELECT COUNT(*) FROM cache_stats
               WHERE level = ? AND event = 'miss' AND timestamp > ?""",
            (level, cutoff),
        ).fetchone()

        hit_count = hits[0] if hits else 0
        time_saved = hits[1] if hits else 0
        miss_count = misses[0] if misses else 0
        total = hit_count + miss_count

        report[level] = {
            "hits": hit_count,
            "misses": miss_count,
            "hit_rate": (hit_count / total * 100) if total > 0 else 0.0,
            "time_saved_ms": time_saved,
        }

    # Add totals
    total_hits = sum(r["hits"] for r in report.values())
    total_misses = sum(r["misses"] for r in report.values())
    total_saved = sum(r["time_saved_ms"] for r in report.values())
    total = total_hits + total_misses

    report["total"] = {
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": (total_hits / total * 100) if total > 0 else 0.0,
        "time_saved_ms": total_saved,
    }

    # Add cache sizes
    for table, level in [
        ("query_embeddings", "L1"),
        ("search_results", "L2"),
        ("rerank_results", "L3"),
    ]:
        try:
            count = cache_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            report[level]["entries"] = count
        except sqlite3.OperationalError:
            report[level]["entries"] = 0

    return report


def get_cache_health_stats(cache_con: sqlite3.Connection) -> dict:
    """
    Get comprehensive cache health statistics for the health command.

    Returns stats including:
    - Entry counts per level
    - Hit/miss ratios (lifetime and recent)
    - Time saved estimates
    - Cache age information
    - Total hit counts from entries

    Args:
        cache_con: Cache database connection.

    Returns:
        Dict with comprehensive cache statistics.
    """
    stats: dict = {
        "levels": {},
        "totals": {
            "entries": 0,
            "lifetime_hits": 0,
            "lifetime_misses": 0,
            "recent_hits": 0,
            "recent_misses": 0,
            "time_saved_ms": 0,
        },
    }

    # Per-level stats
    tables = [
        ("L1", "query_embeddings", "Embeddings"),
        ("L2", "search_results", "Search"),
        ("L3", "rerank_results", "Rerank"),
    ]

    for level, table, description in tables:
        try:
            # Entry count and total hit count from entries
            row = cache_con.execute(
                f"SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM {table}"
            ).fetchone()
            entries = row[0] if row else 0
            entry_hits = row[1] if row else 0

            # Oldest and newest entry
            oldest = cache_con.execute(f"SELECT MIN(created_at) FROM {table}").fetchone()
            newest = cache_con.execute(f"SELECT MAX(created_at) FROM {table}").fetchone()

            # Recent stats from cache_stats (last 24h)
            cutoff_24h = time.time() - (24 * 60 * 60)
            recent = cache_con.execute(
                """SELECT
                    SUM(CASE WHEN event = 'hit' THEN 1 ELSE 0 END) as hits,
                    SUM(CASE WHEN event = 'miss' THEN 1 ELSE 0 END) as misses,
                    COALESCE(SUM(CASE WHEN event = 'hit' THEN time_saved_ms ELSE 0 END), 0) as saved
                FROM cache_stats
                WHERE level = ? AND timestamp > ?""",
                (level, cutoff_24h),
            ).fetchone()

            # Lifetime stats from cache_stats
            lifetime = cache_con.execute(
                """SELECT
                    SUM(CASE WHEN event = 'hit' THEN 1 ELSE 0 END) as hits,
                    SUM(CASE WHEN event = 'miss' THEN 1 ELSE 0 END) as misses,
                    COALESCE(SUM(CASE WHEN event = 'hit' THEN time_saved_ms ELSE 0 END), 0) as saved
                FROM cache_stats
                WHERE level = ?""",
                (level,),
            ).fetchone()

            recent_hits = recent[0] or 0 if recent else 0
            recent_misses = recent[1] or 0 if recent else 0
            _ = recent[2] or 0 if recent else 0  # recent_saved (unused but part of query)
            lifetime_hits = lifetime[0] or 0 if lifetime else 0
            lifetime_misses = lifetime[1] or 0 if lifetime else 0
            lifetime_saved = lifetime[2] or 0 if lifetime else 0

            # Calculate hit rates
            recent_total = recent_hits + recent_misses
            lifetime_total = lifetime_hits + lifetime_misses

            level_stats = {
                "description": description,
                "entries": entries,
                "entry_hits": entry_hits,  # Cumulative hits on cached entries
                "recent_hits": recent_hits,
                "recent_misses": recent_misses,
                "recent_hit_rate": (recent_hits / recent_total * 100) if recent_total > 0 else None,
                "lifetime_hits": lifetime_hits,
                "lifetime_misses": lifetime_misses,
                "lifetime_hit_rate": (lifetime_hits / lifetime_total * 100)
                if lifetime_total > 0
                else None,
                "time_saved_ms": lifetime_saved,
                "oldest_entry": oldest[0] if oldest and oldest[0] else None,
                "newest_entry": newest[0] if newest and newest[0] else None,
            }

            stats["levels"][level] = level_stats

            # Aggregate totals
            stats["totals"]["entries"] += entries
            stats["totals"]["lifetime_hits"] += lifetime_hits
            stats["totals"]["lifetime_misses"] += lifetime_misses
            stats["totals"]["recent_hits"] += recent_hits
            stats["totals"]["recent_misses"] += recent_misses
            stats["totals"]["time_saved_ms"] += lifetime_saved

        except sqlite3.OperationalError:
            # Table doesn't exist
            stats["levels"][level] = {
                "description": description,
                "entries": 0,
                "error": "table not found",
            }

    # Calculate total hit rates
    totals = stats["totals"]
    recent_total = totals["recent_hits"] + totals["recent_misses"]
    lifetime_total = totals["lifetime_hits"] + totals["lifetime_misses"]
    totals["recent_hit_rate"] = (
        (totals["recent_hits"] / recent_total * 100) if recent_total > 0 else None
    )
    totals["lifetime_hit_rate"] = (
        (totals["lifetime_hits"] / lifetime_total * 100) if lifetime_total > 0 else None
    )

    return stats


# =============================================================================
# Maintenance
# =============================================================================


def evict_lru(cache_con: sqlite3.Connection, table: str, max_entries: int) -> int:
    """
    Evict oldest entries beyond max.

    Args:
        cache_con: Cache database connection.
        table: Table name to evict from.
        max_entries: Maximum entries to keep.

    Returns:
        Number of entries evicted.
    """
    # Count current entries
    count = cache_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    if count <= max_entries:
        return 0

    to_evict = count - max_entries

    # Delete oldest entries (by created_at)
    cache_con.execute(
        f"""DELETE FROM {table}
            WHERE cache_key IN (
                SELECT cache_key FROM {table}
                ORDER BY created_at ASC
                LIMIT ?
            )""",
        (to_evict,),
    )
    cache_con.commit()

    return to_evict


def clear_expired_l1(cache_con: sqlite3.Connection) -> int:
    """
    Remove expired L1 entries.

    Returns:
        Number of entries removed.
    """
    cutoff = time.time() - L1_TTL_SECONDS

    cursor = cache_con.execute(
        "DELETE FROM query_embeddings WHERE created_at < ?",
        (cutoff,),
    )
    cache_con.commit()

    return cursor.rowcount


def clear_stale_l2(cache_con: sqlite3.Connection, current_db_version: int) -> int:
    """
    Remove L2 entries with old db_version.

    Args:
        cache_con: Cache database connection.
        current_db_version: Current db_version from index.

    Returns:
        Number of entries removed.
    """
    cursor = cache_con.execute(
        "DELETE FROM search_results WHERE db_version < ?",
        (current_db_version,),
    )
    cache_con.commit()

    return cursor.rowcount


def clear_all_caches(cache_con: sqlite3.Connection) -> dict:
    """
    Clear all cache tables.

    Returns:
        Dict with count of entries cleared per table.
    """
    result = {}

    for table in ["query_embeddings", "search_results", "rerank_results", "cache_stats"]:
        try:
            count = cache_con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cache_con.execute(f"DELETE FROM {table}")
            result[table] = count
        except sqlite3.OperationalError:
            result[table] = 0

    cache_con.commit()
    return result
