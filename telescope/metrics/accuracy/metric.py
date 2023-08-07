from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.result import MetricResult

from sklearn.metrics import accuracy_score, multilabel_confusion_matrix


class Accuracy(Metric):

    name = "Accuracy"
    segment_level = True

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        if ref == []:
            score = 0
        else:
            score = accuracy_score(ref, cand)
        labels = self.labels
        number_of_labels = len(labels)
        label_scores = []

        matrix = multilabel_confusion_matrix(ref, cand, labels=labels)

        for i in range(number_of_labels):
            tn,fp,fn,tp = list(list(matrix[i][0]) + list(matrix[i][1]))
            sum = tn+fp+fn+tp
            if sum == 0:
                label_scores.append(sum)
            else:
                label_scores.append((tp+tn)/(sum))

        return MetricResult(score,label_scores, src, cand, ref, self.name)
