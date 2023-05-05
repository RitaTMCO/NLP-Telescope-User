from typing import List

from telescope.metrics.result import MetricResult


class ROUGELResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        precision: float,
        recall: float
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.precision = precision
        self.recall = recall

    def __str__(self):
        return f"{self.metric}({self.sys_score}), Precision = {self.precision}, Recall = {self.recall}"
