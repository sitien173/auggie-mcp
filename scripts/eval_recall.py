"""Recall@10 evaluation harness for auggie-mcp semantic search."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol


class SearchBackend(Protocol):
    def prepare(self, corpus: Path) -> None: ...
    def search(self, corpus: Path, query: str) -> list[str]: ...


class NewBackend:
    """Uses the SQLite-only index and semantic search implementation."""

    _TOP_K_STEPS = (10, 20, 40, 50)

    def __init__(self) -> None:
        self._config = None

    def prepare(self, corpus: Path) -> None:
        from auggie_mcp.config import load as load_config
        from auggie_mcp.embed import VoyageEmbedder
        from auggie_mcp.indexer import index_update

        self._config = load_config()
        index_update(
            corpus, config=self._config, embedder=VoyageEmbedder(self._config)
        )

    def search(self, corpus: Path, query: str) -> list[str]:
        from auggie_mcp.semantic import semantic_search

        seen: list[str] = []
        for top_k in self._TOP_K_STEPS:
            search_options = {"top_k": top_k}
            if self._config is not None:
                search_options["config"] = self._config
            result = semantic_search(corpus, query, **search_options)
            for hit in result["hits"]:
                rel = str(hit["path"])
                if rel not in seen:
                    seen.append(rel)
                if len(seen) >= 10:
                    return seen[:10]
            if len(result["hits"]) < top_k:
                break
        return seen[:10]


BACKENDS: dict[str, type[SearchBackend]] = {
    "new": NewBackend,
}


def resolve_ref(ref: str) -> tuple[str, str]:
    """Resolve a git ref to (short_sha, full_sha)."""
    full_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        text=True,
    ).strip()
    short_sha = subprocess.check_output(
        ["git", "rev-parse", "--short", full_sha],
        text=True,
    ).strip()
    return short_sha, full_sha


def extract_corpus(full_sha: str) -> Path:
    """Extract a git commit tree into a fresh temporary directory."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="auggie-eval-"))
    try:
        archive = subprocess.Popen(
            ["git", "archive", full_sha],
            stdout=subprocess.PIPE,
        )
        try:
            subprocess.run(
                ["tar", "-x", "-C", str(tmp_dir)],
                stdin=archive.stdout,
                check=True,
            )
        finally:
            if archive.stdout is not None:
                archive.stdout.close()
        if archive.wait() != 0:
            raise subprocess.CalledProcessError(archive.returncode, ["git", "archive", full_sha])
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir


def evaluate(
    backend: SearchBackend,
    corpus: Path,
    queries: list[dict],
    *,
    corpus_commit: str,
    corpus_sha: str,
) -> dict:
    backend.prepare(corpus)
    per_query = []
    for item in queries:
        q = item["query"]
        expected = item["expected"]
        top10 = list(dict.fromkeys(backend.search(corpus, q)))[:10]
        recall = len(set(expected) & set(top10)) / len(expected)
        per_query.append({"query": q, "recall": recall, "top10": top10})

    mean_recall = sum(r["recall"] for r in per_query) / len(per_query) if per_query else 0.0
    return {
        "backend": None,
        "corpus_commit": corpus_commit,
        "corpus_sha": corpus_sha,
        "queries": len(per_query),
        "recall_at_10": mean_recall,
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate recall@10 on a frozen corpus.")
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument(
        "--ref",
        default="refs/plans/sqlite-only-rewrite/base",
        help="Git ref to extract and evaluate (default: refs/plans/sqlite-only-rewrite/base)",
    )
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args(argv)

    if not os.environ.get("VOYAGE_API_KEY"):
        print(
            "Error: VOYAGE_API_KEY environment variable is required to run recall evaluation.",
            file=sys.stderr,
        )
        return 2

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    short_sha, full_sha = resolve_ref(args.ref)
    baseline = None
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        if baseline.get("corpus_sha") != full_sha:
            print(
                f"baseline corpus_sha {baseline.get('corpus_sha')} does not match {full_sha}",
                file=sys.stderr,
            )
            return 2
    backend = BACKENDS[args.backend]()

    corpus_dir = extract_corpus(full_sha)
    try:
        result = evaluate(
            backend,
            corpus_dir,
            queries,
            corpus_commit=short_sha,
            corpus_sha=full_sha,
        )
    finally:
        shutil.rmtree(corpus_dir, ignore_errors=True)

    result["backend"] = args.backend

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        print(
            f"backend={result['backend']} commit={result['corpus_commit']} "
            f"queries={result['queries']} recall@10={result['recall_at_10']:.4f}"
        )

    if baseline is not None and result["recall_at_10"] < baseline["recall_at_10"] - 0.05:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
