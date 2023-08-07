from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.result import MetricResult
from fairlearn.metrics import demographic_parity_difference



class DemographicParity(Metric):

    name = "Demographic-Parity"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        num_instances = len(ref)
        get_group_right_true = []
        get_group_right_pred = []

        for i in range(num_instances):
            get_group_right_true.append(1)
            if ref[i] == cand[i]:
                get_group_right_pred.append(1)
            else:
                get_group_right_pred.append(0)
        
        score = demographic_parity_difference(get_group_right_true,get_group_right_pred, sensitive_features=ref)

        return MetricResult(score, [], src, cand, ref, self.name)
