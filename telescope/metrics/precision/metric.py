from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.result import MetricResult

from sklearn.metrics import precision_score


class Precision(Metric):

    name = "Precision"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        if ref == []:
            score = 0
        else:
            score = precision_score(ref, cand, average='macro', zero_division=0)

        return MetricResult(score, [], src, cand, ref, self.name)
