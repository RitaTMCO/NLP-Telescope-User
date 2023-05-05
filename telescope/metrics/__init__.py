from .sacrebleu import sacreBLEU
from .chrf import chrF
from .zero_edit import ZeroEdit

# from .bleurt import BLEURT
from .bertscore import BERTScore
from .comet import COMET
from .ter import TER
# from .prism import Prism
from .gleu import GLEU

from .rouge_one import ROUGEOne
from .rouge_two import ROUGETwo
from .rouge_l import ROUGEL

from .accuracy import Accuracy
from .precision import Precision
from .recall import Recall
from .f1_score import F1Score

from .result import MetricResult, PairwiseResult, BootstrapResult


AVAILABLE_METRICS = [
    COMET,
    sacreBLEU,
    chrF,
    ZeroEdit,
    # BLEURT,
    BERTScore,
    TER,
    # Prism,
    GLEU,
    ROUGEOne, 
    ROUGETwo, 
    ROUGEL,
    Accuracy,
    Precision,
    Recall,
    F1Score
]

AVAILABLE_NLG_METRICS = [
    BERTScore,
    #BLEURT,
]

AVAILABLE_MT_METRICS = [COMET,sacreBLEU,chrF,TER,GLEU,ZeroEdit] + AVAILABLE_NLG_METRICS

AVAILABLE_SUMMARIZATION_METRICS = [ROUGEOne, ROUGETwo, ROUGEL] + AVAILABLE_NLG_METRICS

AVAILABLE_DIALOGUE_METRICS = [sacreBLEU, ROUGEOne, ROUGETwo, ROUGEL] + AVAILABLE_NLG_METRICS

AVAILABLE_CLASSIFICATION_METRICS = [Accuracy,Precision,Recall,F1Score]
