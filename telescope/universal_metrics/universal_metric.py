import abc
from typing import Dict
from telescope.metrics.metric import MultipleMetricResults
from telescope.testset import MultipleTestset
from telescope.universal_metrics.universal_metric_results import UniversalMetricResult, MultipleUniversalMetricResult


class UniversalMetric(metaclass=abc.ABCMeta):

    name = None
    title = None

    def __init__(self, multiple_metrics_results: Dict[str, MultipleMetricResults]):
        self.multiple_metrics_results = multiple_metrics_results #{metric:MultipleMetricResults}

    def ranking_systems(self, systems_universal_scores:Dict[str,float],reverse:bool=True):
        systems_ranks = {}
        r = 1
        sorted_sys_id = sorted(systems_universal_scores, key=systems_universal_scores.get, reverse=reverse)
        num_sys = len(sorted_sys_id)

        for i in range(num_sys):
            sys_id = sorted_sys_id[i]
            score = systems_universal_scores[sys_id]
            if i != 0 and systems_ranks[sorted_sys_id[i-1]]["score"] == score:
                systems_ranks[sys_id] = {"rank":systems_ranks[sorted_sys_id[i-1]]["rank"], "score":score}  
            else:
                systems_ranks[sys_id] = {"rank":r, "score":score} 
                r += 1
        return systems_ranks
        
    @abc.abstractmethod
    def universal_score(self,testset:MultipleTestset) -> Dict[str,float]:
        pass

    def universal_score_calculation_and_ranking(self,testset:MultipleTestset) -> MultipleUniversalMetricResult:
        ref = testset.ref
        systems_outputs = testset.systems_output
        metrics = list(self.multiple_metrics_results.keys())

        universal_scores = self.universal_score(testset)

        if self.name == "social-choice-theory":
            ranks = self.ranking_systems(universal_scores,False) 
        else:
            ranks = self.ranking_systems(universal_scores)


        sys_id_results = {sys_id:UniversalMetricResult(ref, systems_outputs[sys_id], metrics, self.name, self.title, description["rank"], description["score"]) 
                                              for sys_id,description in ranks.items()}
        return MultipleUniversalMetricResult(sys_id_results)


