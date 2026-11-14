"""Line-window text chunker for auggie-mcp."""

from __future__ import annotations

from dataclasses import dataclass

CHUNKER_VERSION = 1


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    start_line: int
    end_line: int
    text: str


def chunk_file(
    rel_path: str,
    content: str,
    *,
    max_lines: int = 30,
    overlap_lines: int = 15,
) -> list[Chunk]:
    """Split file content into overlapping line-window chunks."""
    if not content:
        return []
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return []

    window = max(1, max_lines)
    overlap = max(0, min(overlap_lines, window - 1))

    chunks: list[Chunk] = []
    start = 0
    ordinal = 0
    while start < total:
        raw_end = min(start + window - 1, total - 1)
        end = raw_end
        if raw_end < total - 1:
            for idx in range(raw_end, start - 1, -1):
                if not lines[idx].strip():
                    end = idx
                    break

        chunk_body = "\n".join(lines[start : end + 1])
        chunks.append(
            Chunk(
                ordinal=ordinal,
                start_line=start + 1,
                end_line=end + 1,
                text=f"{rel_path}\n{chunk_body}",
            )
        )
        if end >= total - 1:
            break
        ordinal += 1
        start = max(start + 1, end + 1 - overlap)

    return chunks
