from __future__ import annotations


def hit_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return 1.0 if relevant_ids.intersection(retrieved_ids[:k]) else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(relevant_ids.intersection(retrieved_ids[:k])) / len(relevant_ids)
