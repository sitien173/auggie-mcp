"""
Text chunking module for auggie.

Splits source code into overlapping chunks suitable for embedding.
Overlapping ensures that context around chunk boundaries is preserved,
improving search quality for queries that span chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """
    A chunk of text from a source file.

    Attributes:
        chunk_index: Zero-based index of this chunk within the file.
        start_line: First line number (1-indexed, inclusive).
        end_line: Last line number (1-indexed, inclusive).
        text: The actual text content of the chunk.
    """

    chunk_index: int
    start_line: int
    end_line: int
    text: str


def chunk_lines(text: str, chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    """
    Split text into overlapping chunks by line count.

    Creates chunks of approximately chunk_size lines with overlap lines
    of context shared between adjacent chunks. This overlap helps preserve
    context for code that spans chunk boundaries.

    Args:
        text: Source text to chunk.
        chunk_size: Target number of lines per chunk (default: 120).
        overlap: Number of lines to overlap between chunks (default: 20).
            Should be less than chunk_size.

    Returns:
        List of Chunk objects. Empty chunks are filtered out.

    Example:
        >>> text = "line1\\nline2\\nline3\\nline4\\nline5"
        >>> chunks = chunk_lines(text, chunk_size=2, overlap=1)
        >>> len(chunks)
        3
        >>> chunks[0].text
        'line1\\nline2'
        >>> chunks[1].start_line
        2

    Note:
        Line numbers in chunks are 1-indexed to match editor conventions.
        The overlap ensures context is preserved: if you search for
        something near a chunk boundary, relevant context from the
        adjacent chunk will also be included.
    """
    lines = text.splitlines()
    out: list[Chunk] = []
    i = 0
    idx = 0

    while i < len(lines):
        start = i
        end = min(i + chunk_size, len(lines))
        chunk_text = "\n".join(lines[start:end]).strip()

        if chunk_text:
            out.append(
                Chunk(
                    chunk_index=idx,
                    start_line=start + 1,  # 1-indexed
                    end_line=end,
                    text=chunk_text,
                )
            )
            idx += 1

        # Break if we've reached the end
        if end == len(lines):
            break

        # Move forward, keeping overlap
        i = max(end - overlap, start + 1)

    return out
