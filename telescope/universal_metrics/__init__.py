import sys
from .average import Average
from .median import Median
from .weighted_mean import WeightedMean
from .pairwise_comparison import PairwiseComparison
from .social_choice_theory import SocialChoiceTheory

from .universal_metric_results import UniversalMetricResult

from telescope.utils import read_yaml_file

universal_metrics_yaml = read_yaml_file("universal_metrics.yaml")

AVAILABLE_UNIVERSAL_METRICS  = [ 
    Average,
    Median,
    WeightedMean,
    PairwiseComparison,
    SocialChoiceTheory
]

AVAILABLE_UNIVERSAL_METRICS_NAMES = {universal_metric.name:universal_metric for universal_metric in AVAILABLE_UNIVERSAL_METRICS if universal_metric!=WeightedMean}

for weighted_mean in universal_metrics_yaml["Weights"]:
    mean_name = list(weighted_mean.keys())[0]
    if mean_name not in AVAILABLE_UNIVERSAL_METRICS_NAMES:
        AVAILABLE_UNIVERSAL_METRICS_NAMES[mean_name] = WeightedMean

try:
    AVAILABLE_NLP_UNIVERSAL_METRICS = {
    universal_metric_name: AVAILABLE_UNIVERSAL_METRICS_NAMES[universal_metric_name] for universal_metric_name in universal_metrics_yaml["NLP universal metrics"]
    }

    AVAILABLE_NLG_UNIVERSAL_METRICS  = { 
    **{universal_metric_name: AVAILABLE_UNIVERSAL_METRICS_NAMES[universal_metric_name] for universal_metric_name in universal_metrics_yaml["NLG universal metrics"]} ,
    **AVAILABLE_NLP_UNIVERSAL_METRICS}

    AVAILABLE_MT_UNIVERSAL_METRICS = {
    **{universal_metric_name: AVAILABLE_UNIVERSAL_METRICS_NAMES[universal_metric_name] for universal_metric_name in universal_metrics_yaml["Machine Translation universal metrics"]},
    **AVAILABLE_NLG_UNIVERSAL_METRICS}

    AVAILABLE_SUMMARIZATION_UNIVERSAL_METRICS = {
    **{universal_metric_name: AVAILABLE_UNIVERSAL_METRICS_NAMES[universal_metric_name] for universal_metric_name in universal_metrics_yaml["Summarization universal metrics"]} ,
    **AVAILABLE_NLG_UNIVERSAL_METRICS}

    AVAILABLE_DIALOGUE_UNIVERSAL_METRICS= {
    **{universal_metric_name: AVAILABLE_UNIVERSAL_METRICS_NAMES[universal_metric_name] for universal_metric_name in universal_metrics_yaml["Dialogue System universal metrics"]},
    **AVAILABLE_NLG_UNIVERSAL_METRICS}

    AVAILABLE_CLASSIFICATION_UNIVERSAL_METRICS  = { 
    **{universal_metric_name: AVAILABLE_UNIVERSAL_METRICS_NAMES[universal_metric_name] for universal_metric_name in universal_metrics_yaml["Classification universal metrics"]} ,
    **AVAILABLE_NLP_UNIVERSAL_METRICS}

except KeyError as error:
    print("Error (yaml): " + str(error) + " as an universal metric is not available.")
    sys.exit(1)