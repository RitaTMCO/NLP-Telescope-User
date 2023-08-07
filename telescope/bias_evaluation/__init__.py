import sys

from .gender_bias_evaluation import GenderBiasEvaluation

from .bias_result import BiasResult

from telescope.utils import read_yaml_file

bias_evaluations_yaml = bias_evaluations_yaml = read_yaml_file("bias_evaluations.yaml")

AVAILABLE_BIAS_EVALUATIONS  = [ 
    GenderBiasEvaluation,
]

names_availabels_bias_evaluations = {bias_evaluation.name:bias_evaluation for bias_evaluation in AVAILABLE_BIAS_EVALUATIONS}

try:
    AVAILABLE_NLP_BIAS_EVALUATIONS = [names_availabels_bias_evaluations[bias_evaluation_name] for bias_evaluation_name in bias_evaluations_yaml["NLP bias evaluations"]]

    AVAILABLE_NLG_BIAS_EVALUATIONS  = [names_availabels_bias_evaluations[bias_evaluation_name] for bias_evaluation_name in bias_evaluations_yaml["NLG bias evaluations"]]  + AVAILABLE_NLP_BIAS_EVALUATIONS

    AVAILABLE_MT_BIAS_EVALUATIONS = [names_availabels_bias_evaluations[bias_evaluation_name] for bias_evaluation_name in bias_evaluations_yaml["Machine Translation bias evaluations"]] + AVAILABLE_NLG_BIAS_EVALUATIONS 

    AVAILABLE_SUMMARIZATION_BIAS_EVALUATIONS = [names_availabels_bias_evaluations[bias_evaluation_name] for bias_evaluation_name in bias_evaluations_yaml["Summarization bias evaluations"]] + AVAILABLE_NLG_BIAS_EVALUATIONS 

    AVAILABLE_DIALOGUE_BIAS_EVALUATIONS  = [names_availabels_bias_evaluations[bias_evaluation_name] for bias_evaluation_name in bias_evaluations_yaml["Dialogue System bias evaluations"]] + AVAILABLE_NLG_BIAS_EVALUATIONS 

    AVAILABLE_CLASSIFICATION_BIAS_EVALUATIONS  = [names_availabels_bias_evaluations[bias_evaluation_name] for bias_evaluation_name in bias_evaluations_yaml["Classification bias evaluations"]] + AVAILABLE_NLP_BIAS_EVALUATIONS 

except KeyError as error:
    print("Error (yaml): " + str(error) + " as a bias evaluation is not available.")
    sys.exit(1)