"""Incrementally segment new docs and merge into the by_prefix layout.

Usage:
    python -m alfs.seg.augment \
        --docs ../text_data/docs.parquet \
        --seg-data-dir ../seg_data/by_prefix \
        [--workers 8]
"""

import argparse
import multiprocessing
from pathlib import Path

import polars as pl
import pyarrow as pa  # type: ignore[import-untyped]
import spacy

from alfs.seg.aggregate_occurrences import aggregate
from alfs.seg.segment_docs import iter_chunks

PA_SCHEMA = pa.schema(
    [
        ("form", pa.string()),
        ("doc_id", pa.string()),
        ("byte_offset", pa.int64()),
    ]
)

_nlp = None


def _init_worker() -> None:
    global _nlp
    _nlp = spacy.load("en_core_web_sm")


def _segment_doc(args: tuple[str, str]) -> list[tuple[str, str, int]]:
    """Segment a single doc; returns list of (form, doc_id, byte_offset)."""
    text, doc_id = args
    rows: list[tuple[str, str, int]] = []
    for chunk, chunk_start_chars in iter_chunks(text):
        chunk_start_bytes = len(text[:chunk_start_chars].encode())
        assert _nlp is not None
        spacy_doc = _nlp(chunk)
        for token in spacy_doc:
            byte_offset = chunk_start_bytes + len(chunk[: token.idx].encode())
            rows.append((token.text, doc_id, byte_offset))
    return rows


def _get_segmented_doc_ids(seg_data_dir: Path) -> set[str]:
    """Collect all doc_ids already present in by_prefix parquets."""
    parquets = list(seg_data_dir.glob("*/occurrences.parquet"))
    if not parquets:
        return set()
    doc_ids: set[str] = set()
    for p in parquets:
        df = pl.read_parquet(p, columns=["doc_id"])
        doc_ids.update(df["doc_id"].to_list())
    return doc_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally segment new docs into by_prefix layout"
    )
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument(
        "--seg-data-dir",
        required=True,
        help="Output directory for by_prefix occurrences",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes"
    )
    args = parser.parse_args()

    seg_data_dir = Path(args.seg_data_dir)

    # 1. Find already-segmented doc_ids
    segmented_ids = _get_segmented_doc_ids(seg_data_dir)
    print(f"Already segmented: {len(segmented_ids)} docs")

    # 2. Load corpus and filter to new docs
    print(f"Loading docs from {args.docs}...")
    all_docs = pl.read_parquet(args.docs)
    new_docs = all_docs.filter(~pl.col("doc_id").is_in(list(segmented_ids)))
    print(f"New docs to segment: {len(new_docs)}")

    if len(new_docs) == 0:
        print("Nothing to do.")
        return

    # 3. Segment with multiprocessing pool
    tasks = [(row["text"], row["doc_id"]) for row in new_docs.iter_rows(named=True)]

    all_rows: list[tuple[str, str, int]] = []
    with multiprocessing.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(_segment_doc, tasks), 1):
            all_rows.extend(result)
            if i % 100 == 0 or i == len(tasks):
                print(f"  Segmented {i}/{len(tasks)} docs...", end="\r")
    print()

    # 4. Build DataFrame and merge into by_prefix layout
    df = pl.DataFrame(
        {
            "form": [r[0] for r in all_rows],
            "doc_id": [r[1] for r in all_rows],
            "byte_offset": [r[2] for r in all_rows],
        }
    )
    print(f"Total occurrences: {len(df):,}")
    aggregate(df, seg_data_dir, merge=True)


if __name__ == "__main__":
    main()
