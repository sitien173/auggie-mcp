"""Deterministic output processing foundation for Jev MCP integration.

Provides typed replacement result envelopes, configuration loaders, conservative
token estimation, source-aware chunking, local protection rules, structured-document
detection, omission rendering, and private archive persistence.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

# Fixed constants (not environment-configurable)
PRUNING_THRESHOLD_TOKENS: int = 10_000
DEFAULT_CHUNK_LINES: int = 20
MAX_LINE_CHARS: int = 2_000
MAX_CHUNKS: int = 200
RETENTION_UNCERTAINTY_FLOOR: float = 0.1

Category = Literal["reference", "diagnostic", "result", "progress", "unknown"]
CATEGORIES: Tuple[Category, ...] = (
    "reference",
    "diagnostic",
    "result",
    "progress",
    "unknown",
)

ProcessingStatus = Literal["classified", "unclassified"]
PROCESSING_STATUSES: Tuple[ProcessingStatus, ...] = ("classified", "unclassified")

RetentionState = Literal[
    "below_threshold", "protected", "kept_all", "pruned", "failed_open"
]
RETENTION_STATES: Tuple[RetentionState, ...] = (
    "below_threshold",
    "protected",
    "kept_all",
    "pruned",
    "failed_open",
)


def is_prune_eligible(estimated_tokens: int) -> bool:
    """Check if result estimated tokens strictly exceed pruning threshold."""
    return estimated_tokens > PRUNING_THRESHOLD_TOKENS


class ChunkMetadata(TypedDict):
    id: str
    source: str
    start_line: int
    end_line: int
    category: Category
    protected: bool
    retained: bool


class ProcessingMetadata(TypedDict):
    status: ProcessingStatus
    retention: RetentionState
    estimated_tokens: int
    recovery_path: Optional[str]
    chunks: List[ChunkMetadata]


class RetrievalData(TypedDict):
    result: str


class RetrievalEnvelope(TypedDict):
    tool: Literal["codebase_retrieval"]
    data: RetrievalData
    processing: ProcessingMetadata


class SearchData(TypedDict):
    stdout: str
    stderr: str
    exit_code: int


class SearchEnvelope(TypedDict):
    tool: Literal["codebase_search"]
    data: SearchData
    processing: ProcessingMetadata


def get_default_archive_dir(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve default archive directory based on environment hierarchy."""
    env = os.environ if environ is None else environ

    explicit = env.get("AUGGIE_JEV_ARCHIVE_DIR")
    if explicit:
        return Path(explicit)

    xdg_cache = env.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "auggie-mcp" / "jev"

    return Path.home() / ".cache" / "auggie-mcp" / "jev"


@dataclass(frozen=True)
class JevConfig:
    api_key: Optional[str]
    model: str = "jev-latest"
    timeout_seconds: float = 30.0
    archive_dir: Path = Path.home() / ".cache" / "auggie-mcp" / "jev"
    max_requests: int = 40


def load_config(environ: Optional[Mapping[str, str]] = None) -> JevConfig:
    """Load Jev configuration from environment variables."""
    env = os.environ if environ is None else environ

    api_key = env.get("TYPESAFE_API_KEY")
    model = env.get("AUGGIE_JEV_MODEL", "jev-latest")

    raw_timeout = env.get("AUGGIE_JEV_TIMEOUT_SECONDS")
    if raw_timeout is not None:
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError:
            timeout_seconds = 30.0
    else:
        timeout_seconds = 30.0

    raw_max_requests = env.get("AUGGIE_JEV_MAX_REQUESTS")
    if raw_max_requests is not None:
        try:
            max_requests = int(raw_max_requests)
        except ValueError:
            max_requests = 40
    else:
        max_requests = 40

    archive_dir = get_default_archive_dir(env)

    return JevConfig(
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        archive_dir=archive_dir,
        max_requests=max_requests,
    )


# --- Token Estimation ---

TOKEN_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens without tokenizer.

    Word piece costs 1 token per 6 letters; digit costs 0.5; symbol costs 0.9.
    """
    if not text:
        return 0
    tokens = 0.0
    for match in TOKEN_PIECES.finditer(text):
        piece = match.group()
        first = ord(piece[0])
        if 48 <= first <= 57:  # '0'-'9'
            tokens += len(piece) / 2.0
        elif (65 <= first <= 90) or (97 <= first <= 122):  # 'A'-'Z', 'a'-'z'
            tokens += 1.0 + math.floor((len(piece) - 1) / 6.0)
        else:
            tokens += 0.9
    return math.ceil(tokens)


def estimate_search_tokens(stdout: str, stderr: str) -> int:
    """Estimate total tokens combining search stdout and stderr as one logical result."""
    return estimate_tokens(stdout) + estimate_tokens(stderr)


# --- Protection Patterns and Detection ---

DIAGNOSTIC_PATTERN = re.compile(
    r"(?:\b(?:ERROR|FATAL|FAILED|FAILURE|PANIC|WARN|WARNING)\b)"
    r"|(?:\b(?:error|warning|failure|exception|panic|traceback|assertion)s?\s*:)"
    r"|(?:\berror TS\d+:|^E\s+\S)"
    r"|(?:\b(?:failed|failing|cannot|could not|unable to|denied|refused|timed out)\s+\w)"
    r"|(?:\b\w*(?:Error|Exception)\b\s*[:(])"
    r"|(?:\bTraceback \(most recent call last\))"
    r"|(?:^\s*at\s+\S+\(.*:\d+)"
    r"|(?:\b(?:severity )?vulnerabilit(?:y|ies)\b)"
    r"|(?:\bCrashLoopBackOff\b|\bOOMKilled\b)"
    r"|(?:\bHTTP/[0-9.]+ [45]\d\d\b|\bstatus[=: ]\s*[45]\d\d\b)",
    re.MULTILINE,
)

RESULT_PATTERN = re.compile(
    r"^\s*(?:(?:Test Suites|Tests|Snapshots|Coverage|Results?|Summary|Exit code|Exit status)\s*:"
    r"|(?:Build|Compilation|Tests?)\s+(?:succeeded|completed|finished|passed|failed)\b"
    r"|(?:Artifact|Output file|Report|Coverage report)(?: path)?\s*[:=]\s*\S)",
    re.IGNORECASE | re.MULTILINE,
)

PYTEST_RESULT_PATTERN = re.compile(
    r"^=+\s+.*\b\d+\s+(?:passed|failed|skipped|deselected|xfailed|xpassed|errors?|warnings?)\b.*=+\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_critical_line(line: str) -> bool:
    """Return True if line contains critical diagnostic or result material."""
    return bool(
        DIAGNOSTIC_PATTERN.search(line)
        or RESULT_PATTERN.search(line)
        or PYTEST_RESULT_PATTERN.search(line)
    )


def is_structured_document(text: str) -> bool:
    """Conservative detection for documents where holes would corrupt structure."""
    stripped = text.strip()
    if not stripped:
        return False
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                return True
        except (json.JSONDecodeError, ValueError):
            pass
    if stripped.startswith("<?xml") or stripped.startswith("<!DOCTYPE"):
        return True
    return False


# --- Chunking ---


@dataclass
class LinePiece:
    """A slice of a logical line, mapped to its 1-indexed original line number."""

    text: str
    line_number: int


@dataclass
class Chunk:
    """A chunk of output text belonging to a named source."""

    id: str
    source: str
    start_line: int
    end_line: int
    category: Category = "unknown"
    protected: bool = False
    retained: bool = True
    pieces: List[LinePiece] = None  # type: ignore

    def __post_init__(self):
        if self.pieces is None:
            self.pieces = []

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.pieces)

    @property
    def line_count(self) -> int:
        if self.pieces:
            return len({p.line_number for p in self.pieces})
        return max(1, self.end_line - self.start_line + 1)

    def to_metadata(self) -> ChunkMetadata:
        return {
            "id": self.id,
            "source": self.source,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "category": self.category,
            "protected": self.protected,
            "retained": self.retained,
        }


def _split_into_pieces(text: str) -> List[LinePiece]:
    """Split text into LinePieces respecting MAX_LINE_CHARS without content loss."""
    if not text:
        return []
    pieces: List[LinePiece] = []
    lines = text.splitlines(keepends=True)
    for line_idx, line in enumerate(lines, start=1):
        if len(line) <= MAX_LINE_CHARS:
            pieces.append(LinePiece(text=line, line_number=line_idx))
        else:
            for at in range(0, len(line), MAX_LINE_CHARS):
                pieces.append(
                    LinePiece(
                        text=line[at : at + MAX_LINE_CHARS],
                        line_number=line_idx,
                    )
                )
    return pieces


def _group_pieces(pieces: List[LinePiece], chunk_lines: int) -> List[List[LinePiece]]:
    """Group pieces into nominal chunks of at most chunk_lines pieces."""
    groups: List[List[LinePiece]] = []
    current: List[LinePiece] = []
    for piece in pieces:
        if len(current) >= chunk_lines:
            groups.append(current)
            current = []
        current.append(piece)
    if current:
        groups.append(current)
    return groups


def chunk_result(sources: Sequence[Tuple[str, str]]) -> List[Chunk]:
    """Chunk one complete MCP result across all its textual sources.

    Preserves source identity and bounds the combined result to at most MAX_CHUNKS.
    """
    raw_source_groups: List[Tuple[str, str, List[List[LinePiece]]]] = []
    total_groups = 0

    for source_name, text in sources:
        if not text:
            raw_source_groups.append((source_name, text, []))
            continue
        pieces = _split_into_pieces(text)
        groups = _group_pieces(pieces, DEFAULT_CHUNK_LINES)
        raw_source_groups.append((source_name, text, groups))
        total_groups += len(groups)

    if total_groups <= MAX_CHUNKS:
        merge_factor = 1
    else:
        merge_factor = math.ceil(total_groups / MAX_CHUNKS)
        while (
            sum(
                math.ceil(len(groups) / merge_factor)
                for _, _, groups in raw_source_groups
                if groups
            )
            > MAX_CHUNKS
        ):
            merge_factor += 1

    all_chunks: List[Chunk] = []

    for source_name, source_text, groups in raw_source_groups:
        if not groups:
            continue

        source_chunks: List[Chunk] = []
        for start_idx in range(0, len(groups), merge_factor):
            merged_pieces: List[LinePiece] = []
            for g in groups[start_idx : start_idx + merge_factor]:
                merged_pieces.extend(g)

            chunk_id = f"{source_name}:{len(source_chunks) + 1}"
            chunk = Chunk(
                id=chunk_id,
                source=source_name,
                start_line=merged_pieces[0].line_number,
                end_line=merged_pieces[-1].line_number,
                pieces=merged_pieces,
            )
            source_chunks.append(chunk)

        # Apply protection rules
        _apply_protections(source_chunks, source_text)
        all_chunks.extend(source_chunks)

    return all_chunks


def chunk_source(source_name: str, text: str) -> List[Chunk]:
    """Chunk a single textual source."""
    return chunk_result([(source_name, text)])


def _apply_protections(chunks: List[Chunk], full_text: str) -> None:
    """Apply local deterministic protection rules to a source's chunks."""
    if not chunks:
        return

    # Structured document protection protects every chunk
    if is_structured_document(full_text):
        for c in chunks:
            c.protected = True
        return

    # First and last chunks of each non-empty source are always protected
    chunks[0].protected = True
    chunks[-1].protected = True

    # Critical diagnostic/result protection and boundary adjacency
    for i, chunk in enumerate(chunks):
        lines = chunk.text.splitlines()
        has_critical = any(is_critical_line(line) for line in lines)
        if has_critical:
            chunk.protected = True

        # Check boundary adjacency: if previous chunk's last line is critical
        if i > 0:
            prev_lines = chunks[i - 1].text.splitlines()
            if prev_lines and is_critical_line(prev_lines[-1]):
                chunk.protected = True

        # Check boundary adjacency: if next chunk's first line is critical
        if i < len(chunks) - 1:
            next_lines = chunks[i + 1].text.splitlines()
            if next_lines and is_critical_line(next_lines[0]):
                chunk.protected = True

        # If current chunk has critical at first line, protect previous
        if i > 0 and lines and is_critical_line(lines[0]):
            chunks[i - 1].protected = True

        # If current chunk has critical at last line, protect next
        if i < len(chunks) - 1 and lines and is_critical_line(lines[-1]):
            chunks[i + 1].protected = True


# --- Omission Rendering ---


def render_omissions(
    chunks: Sequence[Chunk], recovery_path: Optional[str] = None
) -> str:
    """Render retained chunks verbatim and collapsed omission markers for dropped runs."""
    if not chunks:
        return ""
    if all(c.retained for c in chunks):
        return "".join(c.text for c in chunks)

    parts: List[str] = []
    i = 0
    while i < len(chunks):
        if chunks[i].retained:
            parts.append(chunks[i].text)
            i += 1
        else:
            dropped: List[Chunk] = []
            while i < len(chunks) and not chunks[i].retained:
                dropped.append(chunks[i])
                i += 1
            unique_source_lines: set[Tuple[str, int]] = set()
            for c in dropped:
                if c.pieces:
                    unique_source_lines.update(
                        (c.source, p.line_number) for p in c.pieces
                    )
                else:
                    unique_source_lines.update(
                        (c.source, ln) for ln in range(c.start_line, c.end_line + 1)
                    )
            dropped_lines = len(unique_source_lines)
            dropped_chars = sum(len(c.text) for c in dropped)
            if recovery_path:
                marker = f"[omitted {dropped_lines} lines ({dropped_chars} characters); full output: {recovery_path}]"
            else:
                marker = f"[omitted {dropped_lines} lines ({dropped_chars} characters)]"

            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
            parts.append(marker)
            if i < len(chunks):
                parts.append("\n")

    return "".join(parts)


def render_omissions_by_source(
    chunks: Sequence[Chunk], recovery_path: Optional[str] = None
) -> Dict[str, str]:
    """Render chunks grouped by source, preserving independent source boundaries."""
    by_source: Dict[str, List[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)
    return {
        source: render_omissions(source_chunks, recovery_path)
        for source, source_chunks in by_source.items()
    }


# --- Private Archive Creation ---

ARCHIVE_FORMAT_VERSION: int = 1


def create_archive(
    tool: str,
    data: Mapping[str, Any],
    archive_dir: Optional[Path] = None,
) -> Path:
    """Persist original runner output in a private, versioned JSON archive.

    Security:
    - Directory mode 0700
    - File mode 0600
    - Exclusive creation with random UUID
    - Returns absolute Path
    """
    target_dir = get_default_archive_dir() if archive_dir is None else archive_dir
    target_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target_dir, 0o700)

    filename = f"{uuid.uuid4()}.json"
    archive_file = (target_dir / filename).resolve()

    payload = json.dumps(
        {
            "version": ARCHIVE_FORMAT_VERSION,
            "tool": tool,
            "data": dict(data),
        },
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(archive_file, flags, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)

    os.chmod(archive_file, 0o600)
    return archive_file
