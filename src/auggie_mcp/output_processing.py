"""Deterministic output processing foundation for Jev MCP integration.

Provides typed replacement result envelopes, configuration loaders, conservative
token estimation, source-aware chunking, local protection rules, structured-document
detection, omission rendering, and private archive persistence.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
)

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    RetryPolicy,
)

# Fixed constants (not environment-configurable)
PRUNING_THRESHOLD_TOKENS: int = 10_000
DEFAULT_CHUNK_LINES: int = 20
MAX_LINE_CHARS: int = 2_000
MAX_CHUNKS: int = 200
RETENTION_UNCERTAINTY_FLOOR: float = 0.1
DEFAULT_BATCH_SIZE: int = 50
MAX_ARCHIVE_WORKERS: int = 4

CLASSIFICATION_INSTRUCTIONS: str = (
    "Classify this output chunk into exactly one of the five allowed categories."
)

CLASSIFICATION_CRITERIA: Mapping[str, str] = {
    "reference": "Source code, reference material, file contents, or context useful to inspect.",
    "diagnostic": "Errors, warnings, failures, stack traces, assertion details, or operational diagnostics.",
    "result": "Direct answer, match, conclusion, summary counts, or produced result.",
    "progress": "Transient progress, status updates, or log noise not itself a result.",
    "unknown": "Mixed, ambiguous, or not safely classifiable into the above categories.",
}

RETENTION_INSTRUCTIONS: str = (
    "Does this chunk contain at least one line that should remain available to satisfy the current goal?"
)


class RequestBudgetExhaustedError(Exception):
    """Raised when an operation would exceed the configured Jev request budget."""

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


class SearchIndexInfo(TypedDict):
    stale_modified: int
    stale_deleted: int
    last_indexed: Optional[str]


class SearchData(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    mode: Literal["exact", "semantic"]
    index: Optional[SearchIndexInfo]


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


def is_valid_timeout(val: Any) -> bool:
    """Return True if val is a valid positive finite timeout."""
    if not isinstance(val, (int, float)) or isinstance(val, bool):
        return False
    if not math.isfinite(val) or val <= 0:
        return False
    return True


def is_valid_max_requests(val: Any) -> bool:
    """Return True if val is a valid positive integer max_requests."""
    if not isinstance(val, int) or isinstance(val, bool):
        return False
    if val <= 0:
        return False
    return True


def is_valid_batch_size(val: Any) -> bool:
    """Return True if val is a valid positive integer batch_size."""
    if not isinstance(val, int) or isinstance(val, bool):
        return False
    if val <= 0:
        return False
    return True


def load_config(environ: Optional[Mapping[str, str]] = None) -> JevConfig:
    """Load Jev configuration from environment variables."""
    env = os.environ if environ is None else environ

    api_key = env.get("TYPESAFE_API_KEY")
    model = env.get("AUGGIE_JEV_MODEL", "jev-latest")

    raw_timeout = env.get("AUGGIE_JEV_TIMEOUT_SECONDS")
    if raw_timeout is not None:
        try:
            parsed_timeout = float(raw_timeout)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid AUGGIE_JEV_TIMEOUT_SECONDS: {raw_timeout!r}")
        if not is_valid_timeout(parsed_timeout):
            raise ValueError(f"Invalid AUGGIE_JEV_TIMEOUT_SECONDS: {raw_timeout!r}")
        timeout_seconds = parsed_timeout
    else:
        timeout_seconds = 30.0

    raw_max_requests = env.get("AUGGIE_JEV_MAX_REQUESTS")
    if raw_max_requests is not None:
        try:
            parsed_max_requests = int(raw_max_requests)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid AUGGIE_JEV_MAX_REQUESTS: {raw_max_requests!r}")
        if not is_valid_max_requests(parsed_max_requests):
            raise ValueError(f"Invalid AUGGIE_JEV_MAX_REQUESTS: {raw_max_requests!r}")
        max_requests = parsed_max_requests
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


_archive_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_archive_executor_lock = threading.Lock()


def get_archive_executor(max_workers: int = MAX_ARCHIVE_WORKERS) -> concurrent.futures.ThreadPoolExecutor:
    """Return the shared bounded ThreadPoolExecutor for archive operations."""
    global _archive_executor
    with _archive_executor_lock:
        if _archive_executor is None:
            _archive_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="auggie-archive",
            )
        return _archive_executor


def reset_archive_executor(wait: bool = True) -> None:
    """Shut down and reset the shared archive ThreadPoolExecutor."""
    global _archive_executor
    with _archive_executor_lock:
        if _archive_executor is not None:
            _archive_executor.shutdown(wait=wait)
            _archive_executor = None


class ArchiveOperation:
    """Coordinates execution, cancellation, and cleanup ownership of an archive task."""

    def __init__(self) -> None:
        self.cancel_token = threading.Event()
        self.archive_path: Optional[Path] = None
        self._accepted = False
        self._aborted = False
        self._lock = threading.Lock()
        self.future: Optional[concurrent.futures.Future[Path]] = None

    def accept(self) -> None:
        """Mark the archive path as accepted by a successful pass."""
        with self._lock:
            self._accepted = True

    def abort(self) -> None:
        """Abort the archive task and unlink any created archive if not accepted."""
        with self._lock:
            self._aborted = True
            self.cancel_token.set()
            if not self._accepted and self.archive_path is not None:
                try:
                    Path(self.archive_path).unlink(missing_ok=True)
                except OSError:
                    pass
                self.archive_path = None

    async def create(
        self,
        tool: str,
        data: Mapping[str, Any],
        archive_dir: Optional[Path],
    ) -> Path:
        """Create archive asynchronously on the bounded archive executor."""
        executor = get_archive_executor()

        def _worker() -> Path:
            if self.cancel_token.is_set():
                raise asyncio.CancelledError("Archive creation cancelled before start")

            p = create_archive(tool, data, archive_dir=archive_dir)

            with self._lock:
                self.archive_path = p
                if self._aborted or self.cancel_token.is_set():
                    if not self._accepted:
                        try:
                            Path(p).unlink(missing_ok=True)
                        except OSError:
                            pass
                        self.archive_path = None
                    raise asyncio.CancelledError("Archive creation cancelled after completion")
            return p

        fut = executor.submit(_worker)
        self.future = fut

        def _on_done(f: concurrent.futures.Future[Path]) -> None:
            try:
                p = f.result()
            except (Exception, asyncio.CancelledError, concurrent.futures.CancelledError):
                return
            with self._lock:
                self.archive_path = p
                if self._aborted and not self._accepted:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass
                    self.archive_path = None

        fut.add_done_callback(_on_done)

        archive_file = await asyncio.wrap_future(fut)
        await asyncio.sleep(0)
        return archive_file


# --- Phase 2: Pipeline, Questions, Validation, and Orchestration ---


class AsyncTypeSafeClientProtocol(Protocol):
    """Protocol for AsyncTypeSafeClient compatibility."""

    async def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any: ...


def create_typesafe_client(config: Optional[JevConfig] = None) -> AsyncTypeSafeClient:
    """Create an AsyncTypeSafeClient with retries explicitly disabled.

    Explicit Phase 3 lifecycle factory. Rejects invalid configuration.
    """
    cfg = config if config is not None else load_config()
    if not is_valid_timeout(cfg.timeout_seconds):
        raise ValueError(f"Invalid timeout_seconds: {cfg.timeout_seconds!r}")
    if not is_valid_max_requests(cfg.max_requests):
        raise ValueError(f"Invalid max_requests: {cfg.max_requests!r}")
    return AsyncTypeSafeClient(
        api_key=cfg.api_key,
        model=cfg.model,
        retry=RetryPolicy(max_retries=0),
        timeout=cfg.timeout_seconds,
    )


def sanitize_search_path(
    path: Optional[str],
    repository: Optional[str | Path] = None,
) -> Optional[str]:
    """Sanitize search path for Jev state.

    Prevents absolute paths from entering Jev state while preserving safe
    relative path context when inside repository.
    """
    if path is None:
        return None
    raw_path = str(path).strip()
    if not raw_path:
        return None
    p = Path(raw_path)
    if not p.is_absolute():
        return raw_path

    if repository is not None:
        try:
            repo_resolved = Path(repository).resolve()
            p_resolved = p.resolve()
            rel = p_resolved.relative_to(repo_resolved)
            rel_str = str(rel)
            if rel_str == ".":
                return None
            return rel_str
        except (ValueError, RuntimeError):
            return None
    return None


def sanitize_search_flags(
    flags: Optional[Sequence[str]],
    repository: Optional[str | Path] = None,
) -> Optional[List[str]]:
    """Sanitize search flags to prevent absolute paths from entering Jev state."""
    if flags is None:
        return None
    sanitized: List[str] = []
    for flag in flags:
        if "=" in flag and flag.startswith("-"):
            opt, sep, val = flag.partition("=")
            clean_val = val.strip("'\"")
            p_val = Path(clean_val)
            if p_val.is_absolute():
                rel = sanitize_search_path(clean_val, repository=repository)
                if rel is not None:
                    sanitized.append(f"{opt}={rel}")
            else:
                sanitized.append(flag)
        else:
            clean_flag = flag.strip("'\"")
            p = Path(clean_flag)
            if p.is_absolute():
                rel = sanitize_search_path(clean_flag, repository=repository)
                if rel is not None:
                    sanitized.append(rel)
            else:
                sanitized.append(flag)
    return sanitized


def build_classification_question() -> Choice:
    """Build a Jev Choice question for chunk classification."""
    return Choice(
        instructions=CLASSIFICATION_INSTRUCTIONS,
        criteria=dict(CLASSIFICATION_CRITERIA),
    )


def build_retention_question() -> Noul:
    """Build a Jev Noul question for chunk retention."""
    return Noul(
        instructions=RETENTION_INSTRUCTIONS,
    )


def build_jev_state(
    task: str,
    tool: str,
    tool_input: Mapping[str, Any],
    chunks: Sequence[Chunk],
    include_categories: bool = False,
) -> Dict[str, Any]:
    """Build sanitized task state payload for Jev System One.

    Contains only task context, safe tool inputs, and chunks.
    Never contains absolute paths, archive paths, or secrets.
    """
    safe_chunks: List[Dict[str, Any]] = []
    for c in chunks:
        item: Dict[str, Any] = {"id": c.id, "text": c.text}
        if include_categories:
            item["category"] = c.category
        safe_chunks.append(item)

    safe_tool_input: Dict[str, Any] = {}
    for k, v in tool_input.items():
        if k in ("repository", "search_root", "archive_path", "archive_dir"):
            continue
        if k == "path" and isinstance(v, (str, Path)):
            if Path(v).is_absolute():
                continue
        if k == "flags" and isinstance(v, (list, tuple)):
            clean_flags = []
            for f in v:
                if not isinstance(f, (str, Path)):
                    continue
                f_str = str(f)
                if "=" in f_str and f_str.startswith("-"):
                    _, _, val = f_str.partition("=")
                    if Path(val.strip("'\"")).is_absolute():
                        continue
                elif Path(f_str.strip("'\"")).is_absolute():
                    continue
                clean_flags.append(f_str)
            v = clean_flags
        safe_tool_input[k] = v

    return {
        "task": task,
        "tool": tool,
        "tool_input": safe_tool_input,
        "chunks": safe_chunks,
    }


def validate_choice_response(
    response: Any,
    expected_ids: Sequence[str],
) -> Dict[str, Category]:
    """Validate that every expected chunk has a valid Choice answer."""
    if not hasattr(response, "choices"):
        raise ValueError("Response missing choices container")
    choices = response.choices
    validated: Dict[str, Category] = {}
    for chunk_id in expected_ids:
        if chunk_id not in choices:
            raise ValueError(f"Missing Choice answer for chunk {chunk_id}")
        ans = choices[chunk_id]
        if not isinstance(ans, ChoiceAnswer):
            raise ValueError(
                f"Answer for chunk {chunk_id} is not an instance of ChoiceAnswer: {type(ans).__name__}"
            )
        choice_val = ans.choice
        if choice_val not in CATEGORIES:
            raise ValueError(f"Invalid category '{choice_val}' for chunk {chunk_id}")
        validated[chunk_id] = choice_val
    return validated


def validate_noul_response(
    response: Any,
    expected_ids: Sequence[str],
) -> Dict[str, float]:
    """Validate that every expected chunk has a valid Noul probability in [0, 1]."""
    if not hasattr(response, "nouls"):
        raise ValueError("Response missing nouls container")
    nouls = response.nouls
    validated: Dict[str, float] = {}
    for chunk_id in expected_ids:
        if chunk_id not in nouls:
            raise ValueError(f"Missing Noul answer for chunk {chunk_id}")
        ans = nouls[chunk_id]
        if not isinstance(ans, NoulAnswer):
            raise ValueError(
                f"Answer for chunk {chunk_id} is not an instance of NoulAnswer: {type(ans).__name__}"
            )
        p = ans.noul
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise ValueError(f"Noul probability for chunk {chunk_id} is not numeric: {p!r}")
        if not math.isfinite(p):
            raise ValueError(f"Noul probability for chunk {chunk_id} is non-finite: {p!r}")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Noul probability for chunk {chunk_id} out of range [0, 1]: {p!r}")
        validated[chunk_id] = float(p)
    return validated


def apply_retention_decisions(
    chunks: Sequence[Chunk],
    scores: Mapping[str, float],
) -> None:
    """Apply Jev retention scores combined with deterministic protections."""
    for chunk in chunks:
        # Diagnostic and result categories are always protected
        if chunk.category in ("diagnostic", "result"):
            chunk.protected = True

        # Phase 1 local protections and category protections always override Jev scores
        if chunk.protected:
            chunk.retained = True
        elif chunk.id not in scores:
            # Unscored chunk is always retained
            chunk.retained = True
        else:
            p = scores[chunk.id]
            if p <= RETENTION_UNCERTAINTY_FLOOR:
                chunk.retained = False
            else:
                chunk.retained = True


def build_fail_open_retrieval(
    original_result: str,
    chunks: Sequence[Chunk],
    estimated_tokens: int,
    recovery_path: Optional[str] = None,
) -> RetrievalEnvelope:
    """Construct fail-open retrieval envelope with exact original result and unknown categories."""
    fail_chunks: List[ChunkMetadata] = []
    for c in chunks:
        meta = c.to_metadata()
        meta["category"] = "unknown"
        meta["retained"] = True
        fail_chunks.append(meta)
    return {
        "tool": "codebase_retrieval",
        "data": {"result": original_result},
        "processing": {
            "status": "unclassified",
            "retention": "failed_open",
            "estimated_tokens": estimated_tokens,
            "recovery_path": recovery_path,
            "chunks": fail_chunks,
        },
    }


def build_fail_open_search(
    original_stdout: str,
    original_stderr: str,
    exit_code: int,
    chunks: Sequence[Chunk],
    estimated_tokens: int,
    recovery_path: Optional[str] = None,
    mode: Literal["exact", "semantic"] = "exact",
    index: Optional[SearchIndexInfo] = None,
) -> SearchEnvelope:
    """Construct fail-open search envelope with exact original outputs and unknown categories."""
    fail_chunks: List[ChunkMetadata] = []
    for c in chunks:
        meta = c.to_metadata()
        meta["category"] = "unknown"
        meta["retained"] = True
        fail_chunks.append(meta)
    return {
        "tool": "codebase_search",
        "data": {
            "stdout": original_stdout,
            "stderr": original_stderr,
            "exit_code": exit_code,
            "mode": mode,
            "index": index,
        },
        "processing": {
            "status": "unclassified",
            "retention": "failed_open",
            "estimated_tokens": estimated_tokens,
            "recovery_path": recovery_path,
            "chunks": fail_chunks,
        },
    }


class OutputProcessor:
    """Asynchronous output processor coordinating Jev classification and retention."""

    def __init__(
        self,
        client: Optional[AsyncTypeSafeClientProtocol] = None,
        config: Optional[JevConfig] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.client = client
        self._config = config
        self.batch_size = batch_size
        self.last_archive_op: Optional[ArchiveOperation] = None

    @property
    def config(self) -> JevConfig:
        if self._config is not None:
            return self._config
        return load_config()

    async def _execute_pass(
        self,
        run_pass_fn: Callable[[], Coroutine[Any, Any, Any]],
        fail_open_fn: Callable[[], Any],
        archive_op: ArchiveOperation,
    ) -> Any:
        try:
            cfg = self.config
            if not is_valid_timeout(cfg.timeout_seconds):
                raise ValueError(f"Invalid timeout_seconds: {cfg.timeout_seconds!r}")
            if not is_valid_max_requests(cfg.max_requests):
                raise ValueError(f"Invalid max_requests: {cfg.max_requests!r}")
            if not is_valid_batch_size(self.batch_size):
                raise ValueError(f"Invalid batch_size: {self.batch_size!r}")
            if self.client is None:
                raise ValueError("TypeSafe client is not available or not configured")

            envelope = await asyncio.wait_for(run_pass_fn(), timeout=cfg.timeout_seconds)
            archive_op.accept()
            return envelope
        except asyncio.CancelledError:
            archive_op.abort()
            raise
        except Exception:
            archive_op.abort()
            return fail_open_fn()

    async def process_retrieval(
        self,
        query: str,
        result: str,
        goal: Optional[str] = None,
    ) -> RetrievalEnvelope:
        chunks = chunk_result([("result", result)])
        estimated_tokens = estimate_tokens(result)

        if not chunks:
            return {
                "tool": "codebase_retrieval",
                "data": {"result": result},
                "processing": {
                    "status": "classified",
                    "retention": "below_threshold",
                    "estimated_tokens": estimated_tokens,
                    "recovery_path": None,
                    "chunks": [],
                },
            }

        archive_op = ArchiveOperation()
        self.last_archive_op = archive_op
        pass_state = {"requests_issued": 0}

        async def _run_pass() -> RetrievalEnvelope:
            cfg = self.config
            client = self.client
            assert client is not None
            effective_goal = goal.strip() if goal and goal.strip() else query
            tool_input = {"query": query}

            # 1. Classification
            await self._classify_chunks(
                client=client,
                task=effective_goal,
                tool="codebase_retrieval",
                tool_input=tool_input,
                chunks=chunks,
                pass_state=pass_state,
                config=cfg,
            )

            # 2. Check Pruning Eligibility
            if not is_prune_eligible(estimated_tokens):
                return {
                    "tool": "codebase_retrieval",
                    "data": {"result": result},
                    "processing": {
                        "status": "classified",
                        "retention": "below_threshold",
                        "estimated_tokens": estimated_tokens,
                        "recovery_path": None,
                        "chunks": [c.to_metadata() for c in chunks],
                    },
                }

            # 3. Create Archive before any retention questions
            archive_file = await archive_op.create(
                "codebase_retrieval",
                {"result": result},
                cfg.archive_dir,
            )
            recovery_path = str(archive_file)

            # 4. Determine all deterministic and category protections before Noul scoring
            for chunk in chunks:
                if chunk.category in ("diagnostic", "result"):
                    chunk.protected = True

            unprotected_chunks = [c for c in chunks if not c.protected]

            # 5. Retention questions only for retention-eligible unprotected chunks
            if unprotected_chunks:
                all_scores = await self._score_retention(
                    client=client,
                    task=effective_goal,
                    tool="codebase_retrieval",
                    tool_input=tool_input,
                    chunks=unprotected_chunks,
                    pass_state=pass_state,
                    config=cfg,
                )
            else:
                all_scores = {}

            # 6. Apply Retention Decisions
            apply_retention_decisions(chunks, all_scores)

            # 7. Assemble Output
            rendered = render_omissions(chunks, recovery_path)
            retention_state = self._determine_retention_state(chunks)

            return {
                "tool": "codebase_retrieval",
                "data": {"result": rendered},
                "processing": {
                    "status": "classified",
                    "retention": retention_state,
                    "estimated_tokens": estimated_tokens,
                    "recovery_path": recovery_path,
                    "chunks": [c.to_metadata() for c in chunks],
                },
            }

        return await self._execute_pass(
            _run_pass,
            lambda: build_fail_open_retrieval(result, chunks, estimated_tokens, recovery_path=None),
            archive_op,
        )

    async def process_search(
        self,
        query: str,
        stdout: str,
        stderr: str,
        exit_code: int,
        path: Optional[str] = None,
        flags: Optional[Sequence[str]] = None,
        goal: Optional[str] = None,
        repository: Optional[str | Path] = None,
        mode: Literal["exact", "semantic"] = "exact",
        index: Optional[SearchIndexInfo] = None,
    ) -> SearchEnvelope:
        sources = [("stdout", stdout), ("stderr", stderr)]
        chunks = chunk_result(sources)
        estimated_tokens = estimate_search_tokens(stdout, stderr)

        if not chunks:
            return {
                "tool": "codebase_search",
                "data": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                    "mode": mode,
                    "index": index,
                },
                "processing": {
                    "status": "classified",
                    "retention": "below_threshold",
                    "estimated_tokens": estimated_tokens,
                    "recovery_path": None,
                    "chunks": [],
                },
            }

        archive_op = ArchiveOperation()
        self.last_archive_op = archive_op
        pass_state = {"requests_issued": 0}

        async def _run_pass() -> SearchEnvelope:
            cfg = self.config
            client = self.client
            assert client is not None
            effective_goal = goal.strip() if goal and goal.strip() else query
            safe_path = sanitize_search_path(path, repository=repository)
            safe_flags = sanitize_search_flags(flags, repository=repository)

            tool_input: Dict[str, Any] = {"query": query, "mode": mode}
            if safe_path is not None:
                tool_input["path"] = safe_path
            if safe_flags is not None:
                tool_input["flags"] = safe_flags

            # 1. Classification
            await self._classify_chunks(
                client=client,
                task=effective_goal,
                tool="codebase_search",
                tool_input=tool_input,
                chunks=chunks,
                pass_state=pass_state,
                config=cfg,
            )

            # 2. Check Pruning Eligibility
            if not is_prune_eligible(estimated_tokens):
                return {
                    "tool": "codebase_search",
                    "data": {
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "mode": mode,
                        "index": index,
                    },
                    "processing": {
                        "status": "classified",
                        "retention": "below_threshold",
                        "estimated_tokens": estimated_tokens,
                        "recovery_path": None,
                        "chunks": [c.to_metadata() for c in chunks],
                    },
                }

            # 3. Create Archive before any retention questions
            data_map = {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
            archive_file = await archive_op.create(
                "codebase_search",
                data_map,
                cfg.archive_dir,
            )
            recovery_path = str(archive_file)

            # 4. Determine all deterministic and category protections before Noul scoring
            for chunk in chunks:
                if chunk.category in ("diagnostic", "result"):
                    chunk.protected = True

            unprotected_chunks = [c for c in chunks if not c.protected]

            # 5. Retention questions only for retention-eligible unprotected chunks
            if unprotected_chunks:
                all_scores = await self._score_retention(
                    client=client,
                    task=effective_goal,
                    tool="codebase_search",
                    tool_input=tool_input,
                    chunks=unprotected_chunks,
                    pass_state=pass_state,
                    config=cfg,
                )
            else:
                all_scores = {}

            # 6. Apply Retention Decisions
            apply_retention_decisions(chunks, all_scores)

            # 7. Assemble Output
            rendered_by_source = render_omissions_by_source(chunks, recovery_path)
            retention_state = self._determine_retention_state(chunks)

            return {
                "tool": "codebase_search",
                "data": {
                    "stdout": rendered_by_source.get("stdout", ""),
                    "stderr": rendered_by_source.get("stderr", ""),
                    "exit_code": exit_code,
                    "mode": mode,
                    "index": index,
                },
                "processing": {
                    "status": "classified",
                    "retention": retention_state,
                    "estimated_tokens": estimated_tokens,
                    "recovery_path": recovery_path,
                    "chunks": [c.to_metadata() for c in chunks],
                },
            }

        return await self._execute_pass(
            _run_pass,
            lambda: build_fail_open_search(
                stdout,
                stderr,
                exit_code,
                chunks,
                estimated_tokens,
                recovery_path=None,
                mode=mode,
                index=index,
            ),
            archive_op,
        )

    async def _classify_chunks(
        self,
        client: AsyncTypeSafeClientProtocol,
        task: str,
        tool: str,
        tool_input: Mapping[str, Any],
        chunks: Sequence[Chunk],
        pass_state: Dict[str, int],
        config: JevConfig,
    ) -> None:
        if not is_valid_batch_size(self.batch_size):
            raise ValueError(f"Invalid batch_size: {self.batch_size!r}")
        batches = [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        for batch in batches:
            if pass_state["requests_issued"] >= config.max_requests:
                raise RequestBudgetExhaustedError("Classification request would exceed budget")
            state = build_jev_state(
                task=task,
                tool=tool,
                tool_input=tool_input,
                chunks=batch,
                include_categories=False,
            )
            questions = {c.id: build_classification_question() for c in batch}
            resp = await client.system_one(
                state=state,
                questions=questions,
                model=config.model,
            )
            pass_state["requests_issued"] += 1
            validated = validate_choice_response(resp, [c.id for c in batch])
            for c in batch:
                c.category = validated[c.id]

    async def _score_retention(
        self,
        client: AsyncTypeSafeClientProtocol,
        task: str,
        tool: str,
        tool_input: Mapping[str, Any],
        chunks: Sequence[Chunk],
        pass_state: Dict[str, int],
        config: JevConfig,
    ) -> Dict[str, float]:
        if not is_valid_batch_size(self.batch_size):
            raise ValueError(f"Invalid batch_size: {self.batch_size!r}")
        batches = [chunks[i : i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        all_scores: Dict[str, float] = {}
        for batch in batches:
            if pass_state["requests_issued"] >= config.max_requests:
                raise RequestBudgetExhaustedError("Retention request would exceed budget")
            state = build_jev_state(
                task=task,
                tool=tool,
                tool_input=tool_input,
                chunks=batch,
                include_categories=True,
            )
            questions = {c.id: build_retention_question() for c in batch}
            resp = await client.system_one(
                state=state,
                questions=questions,
                model=config.model,
            )
            pass_state["requests_issued"] += 1
            validated = validate_noul_response(resp, [c.id for c in batch])
            all_scores.update(validated)
        return all_scores

    @staticmethod
    def _determine_retention_state(chunks: Sequence[Chunk]) -> RetentionState:
        if any(not c.retained for c in chunks):
            return "pruned"
        if all(c.protected for c in chunks):
            return "protected"
        return "kept_all"


async def process_retrieval_output(
    query: str,
    result: str,
    goal: Optional[str] = None,
    client: Optional[AsyncTypeSafeClientProtocol] = None,
    config: Optional[JevConfig] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RetrievalEnvelope:
    """Convenience function processing retrieval output with Jev pipeline."""
    processor = OutputProcessor(client=client, config=config, batch_size=batch_size)
    return await processor.process_retrieval(query=query, result=result, goal=goal)


async def process_search_output(
    query: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    path: Optional[str] = None,
    flags: Optional[Sequence[str]] = None,
    goal: Optional[str] = None,
    repository: Optional[str | Path] = None,
    mode: Literal["exact", "semantic"] = "exact",
    index: Optional[SearchIndexInfo] = None,
    client: Optional[AsyncTypeSafeClientProtocol] = None,
    config: Optional[JevConfig] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> SearchEnvelope:
    """Convenience function processing search output with Jev pipeline."""
    processor = OutputProcessor(client=client, config=config, batch_size=batch_size)
    return await processor.process_search(
        query=query,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        path=path,
        flags=flags,
        goal=goal,
        repository=repository,
        mode=mode,
        index=index,
    )
