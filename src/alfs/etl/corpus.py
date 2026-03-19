"""Helpers for reading and appending to the docs corpus parquet."""

from pathlib import Path

import polars as pl

from alfs.data_models.doc import Doc


def get_doc_ids(corpus_path: Path) -> set[str]:
    """Return the set of doc_ids already in the corpus."""
    df = pl.read_parquet(corpus_path, columns=["doc_id"])
    return set(df["doc_id"].to_list())


def read_docs(corpus_path: Path) -> pl.DataFrame:
    return pl.read_parquet(corpus_path)


def append_docs(new_docs: list[Doc], corpus_path: Path) -> None:
    """Append new_docs to corpus_path, creating the file if it doesn't exist."""
    new_df = pl.DataFrame([d.model_dump() for d in new_docs])
    if corpus_path.exists():
        existing = pl.read_parquet(corpus_path)
        combined = pl.concat([existing, new_df])
    else:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df
    combined.write_parquet(corpus_path)
