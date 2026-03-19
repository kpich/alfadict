"""Deduplication helpers for corpus augmentation."""

from alfs.data_models.doc import Doc
from alfs.etl.ngram_cache import NgramCache


def exact_dedup(docs: list[Doc], existing_ids: set[str]) -> list[Doc]:
    """Filter docs whose doc_id is already in existing_ids (mutates existing_ids)."""
    result = []
    for doc in docs:
        if doc.doc_id not in existing_ids:
            existing_ids.add(doc.doc_id)
            result.append(doc)
    return result


def ngram_dedup(docs: list[Doc], cache: NgramCache) -> list[Doc]:
    """Filter near-duplicate docs using the ngram cache (mutates cache)."""
    result = []
    for doc in docs:
        if not cache.is_near_duplicate(doc.text):
            cache.add_doc(doc.text)
            result.append(doc)
    return result
