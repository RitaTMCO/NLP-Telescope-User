from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.result import MetricResult

from sklearn.metrics import recall_score


class Recall(Metric):

    name = "Recall"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        score = recall_score(ref, cand, average='macro', zero_division=0)

        return MetricResult(score, [], src, cand, ref, self.name)
