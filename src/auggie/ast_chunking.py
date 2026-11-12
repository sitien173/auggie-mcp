"""
AST-aware chunking module for auggie.

Uses tree-sitter to parse source code and extract semantic units
(functions, classes, methods) as chunks. Falls back to line-based
chunking for unsupported languages or parse errors.

This produces semantically coherent chunks that improve both:
- BM25 search (function names not split across chunks)
- Embedding quality (complete semantic units)

Install dependencies:
    pip install tree-sitter-<language>  (for example: pip install tree-sitter-python)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .chunking import Chunk, chunk_lines

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser, Tree

# Environment variable to enable AST chunking globally
ENV_AST_CHUNKING = "AUGGIE_AST_CHUNKING"

# Default max lines per chunk (large functions get split)
DEFAULT_MAX_CHUNK_LINES = 150

# Language parsers - lazy loaded
_PARSERS: dict[str, Parser] = {}
_LANGUAGES: dict[str, Language] = {}

# File extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}

# Node types that represent semantic units we want to chunk
# Maps language -> list of node types
SEMANTIC_NODE_TYPES = {
    "python": [
        "function_definition",
        "class_definition",
        "decorated_definition",
    ],
    "javascript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function",
        "lexical_declaration",  # const/let with arrow functions
    ],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "tsx": [
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "function",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
    ],
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
    "rust": [
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    ],
    "ruby": [
        "method",
        "class",
        "module",
        "singleton_method",
    ],
    "java": [
        "method_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "constructor_declaration",
    ],
    "c": [
        "function_definition",
        "struct_specifier",
        "enum_specifier",
    ],
    "cpp": [
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    ],
}

# Track which languages are actually available
SUPPORTED_LANGUAGES: dict[str, bool] = {}


def is_ast_available() -> bool:
    """
    Check if tree-sitter is available.

    Returns:
        True if tree-sitter is installed and usable.
    """
    try:
        import tree_sitter  # noqa: F401

        return True
    except ImportError:
        return False


def _load_language(lang: str) -> Language | None:
    """
    Load a tree-sitter language parser.

    Args:
        lang: Language name (e.g., "python", "javascript").

    Returns:
        Language object or None if not available.
    """
    global SUPPORTED_LANGUAGES

    if lang in _LANGUAGES:
        return _LANGUAGES[lang]

    if lang in SUPPORTED_LANGUAGES and not SUPPORTED_LANGUAGES[lang]:
        return None  # Already tried and failed

    try:
        from tree_sitter import Language

        # Get the language capsule from the language-specific package
        capsule = None
        if lang == "python":
            import tree_sitter_python as tsp

            capsule = tsp.language()
        elif lang == "javascript":
            import tree_sitter_javascript as tsjs

            capsule = tsjs.language()
        elif lang == "typescript":
            import tree_sitter_typescript as tsts

            capsule = tsts.language_typescript()
        elif lang == "tsx":
            import tree_sitter_typescript as tsts

            capsule = tsts.language_tsx()
        elif lang == "go":
            import tree_sitter_go as tsgo

            capsule = tsgo.language()
        elif lang == "rust":
            import tree_sitter_rust as tsrust

            capsule = tsrust.language()
        elif lang == "ruby":
            import tree_sitter_ruby as tsruby

            capsule = tsruby.language()
        elif lang == "java":
            import tree_sitter_java as tsjava

            capsule = tsjava.language()
        elif lang == "c":
            import tree_sitter_c as tsc

            capsule = tsc.language()
        elif lang == "cpp":
            import tree_sitter_cpp as tscpp

            capsule = tscpp.language()
        elif lang == "c_sharp":
            import tree_sitter_c_sharp as tscs

            capsule = tscs.language()
        elif lang == "bash":
            import tree_sitter_bash as tsbash

            capsule = tsbash.language()
        else:
            SUPPORTED_LANGUAGES[lang] = False
            return None

        # Wrap the capsule with Language() for the new tree-sitter API
        _LANGUAGES[lang] = Language(capsule)
        SUPPORTED_LANGUAGES[lang] = True
        return _LANGUAGES[lang]

    except ImportError:
        SUPPORTED_LANGUAGES[lang] = False
        return None


def _get_parser(lang: str) -> Parser | None:
    """
    Get or create a parser for the given language.

    Args:
        lang: Language name.

    Returns:
        Parser object or None if language not available.
    """
    if lang in _PARSERS:
        return _PARSERS[lang]

    language = _load_language(lang)
    if language is None:
        return None

    try:
        from tree_sitter import Parser

        parser = Parser(language)
        _PARSERS[lang] = parser
        return parser
    except Exception:
        return None


def get_language_for_file(filename: str) -> str | None:
    """
    Detect language from filename extension.

    Args:
        filename: File name or path.

    Returns:
        Language name or None if not recognized.
    """
    ext = Path(filename).suffix.lower()
    lang = EXTENSION_TO_LANGUAGE.get(ext)

    if lang is None:
        return None

    # Check if the language parser is actually available
    if lang not in SUPPORTED_LANGUAGES:
        _load_language(lang)

    if SUPPORTED_LANGUAGES.get(lang):
        return lang
    return None


@dataclass
class SemanticUnit:
    """A semantic unit extracted from AST."""

    name: str
    node_type: str
    start_line: int
    end_line: int
    text: str


def _extract_semantic_units(
    tree: Tree,
    source: bytes,
    lang: str,
) -> list[SemanticUnit]:
    """
    Extract semantic units (functions, classes) from AST.

    Args:
        tree: Parsed tree-sitter tree.
        source: Original source code bytes.
        lang: Language name.

    Returns:
        List of semantic units.
    """
    node_types = SEMANTIC_NODE_TYPES.get(lang, [])
    if not node_types:
        return []

    units: list[SemanticUnit] = []
    node_type_set = set(node_types)

    def visit(node: Node, depth: int = 0) -> None:
        """Recursively visit nodes, extracting semantic units."""
        if node.type in node_type_set:
            # Extract the name if available
            name = ""
            for child in node.children:
                if child.type in ("identifier", "name", "property_identifier"):
                    name = source[child.start_byte : child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                    break

            units.append(
                SemanticUnit(
                    name=name,
                    node_type=node.type,
                    start_line=node.start_point[0] + 1,  # 1-indexed
                    end_line=node.end_point[0] + 1,
                    text=source[node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
                )
            )
            # Don't recurse into nested functions/classes - they're part of parent
            return

        # Recurse into children
        for child in node.children:
            visit(child, depth + 1)

    visit(tree.root_node)
    return units


def _find_gaps(
    units: list[SemanticUnit],
    total_lines: int,
    lines: list[str],
) -> list[SemanticUnit]:
    """
    Find module-level code between semantic units.

    Args:
        units: Existing semantic units.
        total_lines: Total lines in the file.
        lines: Source lines.

    Returns:
        Additional units for module-level code.
    """
    if not lines:
        return []

    gaps: list[SemanticUnit] = []
    covered = set()

    for unit in units:
        for line in range(unit.start_line, unit.end_line + 1):
            covered.add(line)

    # Find contiguous gaps
    gap_start = None
    for line_num in range(1, total_lines + 1):
        if line_num not in covered:
            if gap_start is None:
                gap_start = line_num
        else:
            if gap_start is not None:
                # End of gap
                gap_text = "\n".join(lines[gap_start - 1 : line_num - 1]).strip()
                if gap_text and not _is_only_whitespace_or_comments(gap_text):
                    gaps.append(
                        SemanticUnit(
                            name="<module>",
                            node_type="module_code",
                            start_line=gap_start,
                            end_line=line_num - 1,
                            text=gap_text,
                        )
                    )
                gap_start = None

    # Handle trailing gap
    if gap_start is not None:
        gap_text = "\n".join(lines[gap_start - 1 :]).strip()
        if gap_text and not _is_only_whitespace_or_comments(gap_text):
            gaps.append(
                SemanticUnit(
                    name="<module>",
                    node_type="module_code",
                    start_line=gap_start,
                    end_line=total_lines,
                    text=gap_text,
                )
            )

    return gaps


def _is_only_whitespace_or_comments(text: str) -> bool:
    """Check if text is only whitespace and comments."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
            return False
    return True


def _split_large_unit(
    unit: SemanticUnit,
    max_lines: int,
    overlap: int = 10,
) -> list[SemanticUnit]:
    """
    Split a large semantic unit into smaller chunks.

    Args:
        unit: The unit to split.
        max_lines: Maximum lines per chunk.
        overlap: Lines of overlap between chunks.

    Returns:
        List of smaller units.
    """
    lines = unit.text.splitlines()
    if len(lines) <= max_lines:
        return [unit]

    result: list[SemanticUnit] = []
    i = 0
    part = 0

    while i < len(lines):
        end = min(i + max_lines, len(lines))
        chunk_text = "\n".join(lines[i:end])

        if chunk_text.strip():
            result.append(
                SemanticUnit(
                    name=f"{unit.name}[{part}]" if unit.name else f"<part {part}>",
                    node_type=f"{unit.node_type}_part",
                    start_line=unit.start_line + i,
                    end_line=unit.start_line + end - 1,
                    text=chunk_text,
                )
            )
            part += 1

        if end == len(lines):
            break

        i = max(end - overlap, i + 1)

    return result


def chunk_ast(
    text: str,
    language: str | None = None,
    filename: str | None = None,
    max_chunk_lines: int | None = None,
) -> list[Chunk]:
    """
    Split source code into chunks based on AST structure.

    Uses tree-sitter to parse the code and extract semantic units
    (functions, classes, methods) as separate chunks. Falls back to
    line-based chunking if the language is not supported or parsing fails.

    Args:
        text: Source code to chunk.
        language: Language name (e.g., "python", "javascript").
            If not provided, will try to detect from filename.
        filename: Optional filename for language detection.
        max_chunk_lines: Maximum lines per chunk. Large functions will
            be split. Default: 150.

    Returns:
        List of Chunk objects with semantic boundaries.

    Example:
        >>> code = '''
        ... def hello():
        ...     print("Hello!")
        ...
        ... def goodbye():
        ...     print("Bye!")
        ... '''
        >>> chunks = chunk_ast(code, language="python")
        >>> len(chunks)
        2
        >>> "def hello" in chunks[0].text
        True
    """
    if not text or not text.strip():
        return []

    max_lines = max_chunk_lines or DEFAULT_MAX_CHUNK_LINES

    # Determine language
    lang = language
    if lang is None and filename:
        lang = get_language_for_file(filename)

    if lang is None:
        # Fall back to line-based chunking
        return chunk_lines(text, chunk_size=max_lines, overlap=10)

    # Try to get parser
    parser = _get_parser(lang)
    if parser is None:
        return chunk_lines(text, chunk_size=max_lines, overlap=10)

    # Parse the code
    try:
        source = text.encode("utf-8")
        tree = parser.parse(source)
    except Exception:
        # Parse error - fall back
        return chunk_lines(text, chunk_size=max_lines, overlap=10)

    # Extract semantic units
    units = _extract_semantic_units(tree, source, lang)

    # Find module-level code gaps
    lines = text.splitlines()
    gap_units = _find_gaps(units, len(lines), lines)
    all_units = units + gap_units

    # Sort by start line
    all_units.sort(key=lambda u: u.start_line)

    # Split large units
    final_units: list[SemanticUnit] = []
    for unit in all_units:
        final_units.extend(_split_large_unit(unit, max_lines))

    # Convert to Chunk objects
    chunks: list[Chunk] = []
    for i, unit in enumerate(final_units):
        if unit.text.strip():
            chunks.append(
                Chunk(
                    chunk_index=i,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    text=unit.text,
                )
            )

    return chunks


def ast_chunking_enabled() -> bool:
    """
    Check if AST chunking is enabled via environment.

    Returns:
        True if AUGGIE_AST_CHUNKING is set to a truthy value.
    """
    val = os.environ.get(ENV_AST_CHUNKING, "").lower()
    return val in ("1", "true", "yes", "on")
