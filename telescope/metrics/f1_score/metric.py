from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.result import MetricResult

from sklearn.metrics import f1_score


class F1Score(Metric):

    name = "F1-score"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        score = f1_score(ref, cand, average='macro')

        return MetricResult(score, [], src, cand, ref, self.name)
