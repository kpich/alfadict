# ETL: Piecemeal Corpus Augmentation

The corpus (`docs.parquet`) grows incrementally.  Each `make etl` run adds
`N_DOCS` new documents from a single source dump without re-processing anything
already in the corpus.

## Overview

```
make etl SOURCE=wikibooks N_DOCS=10000   # add up to 10k wikibooks docs
make etl SOURCE=wikisource N_DOCS=5000   # add up to 5k wikisource docs
make seg                                 # segment any new docs
```

Run `make etl` repeatedly until it prints "0 new docs" — the dump is exhausted.
Then switch sources or re-download a fresher dump.

## Doc ID Scheme

`doc_id = SHA256(plain_text)[:8]` (hex).  The same text always produces the
same ID regardless of source or run order.

## Deduplication

Two layers run on every candidate document:

1. **Exact dedup** — skip if `doc_id` already in corpus.
2. **8-gram near-dedup** — skip if ≥ 5% of the doc's word 8-grams match the
   ngram cache.  The cache (`ngram_cache.npy`) is updated after each accepted
   doc (every 10th gram is sampled for indexing; all grams are checked on
   query).  File size is ~24 MB for 100K docs.

## Cursor: Incremental Progress

Each source has a cursor file `{cache_dir}/{source}_cursor.json` storing
`{"pages_consumed": N}`.  The next run skips the first N pages with fast XML
iteration (no `mwparserfromhell` parsing), then picks up from page N+1.

This means the dump is consumed exactly once, chunk by chunk, until exhausted.

## Adding a New Source

Add an entry to `SOURCES` in `sources.py`:

```python
"mywiki": Source(
    name="mywiki",
    dump_url="https://dumps.wikimedia.org/enmywiki/latest/enmywiki-latest-pages-articles.xml.bz2",
    dump_filename="enmywiki-latest-pages-articles.xml.bz2",
    base_url="https://en.mywiki.org/wiki/",
),
```

No new files or Makefile targets are needed.

## Workflow

```
# 1. Download dump (idempotent — skips if already cached)
make download SOURCE=wikibooks

# 2. Augment corpus (run until "0 new docs")
make etl SOURCE=wikibooks N_DOCS=10000

# 3. Segment new docs
make seg
```

`make etl` calls `python -m alfs.etl.download` automatically if the dump is
missing, but pre-downloading is recommended for large dumps.

## Note on Segmentation Format

The current format uses word-level partitions with byte offsets.  A future
migration to overlapping or multi-token tokenization would require re-running
`make seg` on the full corpus.
