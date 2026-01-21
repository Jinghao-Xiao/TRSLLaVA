from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecallMetrics:
    recall_at: dict[int, float]

    @property
    def mean_recall(self) -> float:
        if not self.recall_at:
            return 0.0
        return sum(self.recall_at.values()) / float(len(self.recall_at))


def compute_recall_at_k(hits_at_k: dict[int, int], total: int) -> RecallMetrics:
    if total <= 0:
        return RecallMetrics(recall_at={k: 0.0 for k in hits_at_k})
    return RecallMetrics(recall_at={k: hits_at_k[k] / float(total) for k in hits_at_k})

