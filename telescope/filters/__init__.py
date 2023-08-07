import sys
from .ner import NERFilter
from .length import LengthFilter
from .duplicates import DuplicatesFilter

from telescope.utils import read_yaml_file

filters_yaml = read_yaml_file("filters.yaml")

AVAILABLE_FILTERS = [
    NERFilter, 
    LengthFilter, 
    DuplicatesFilter
]

names_availabels_filters = {filter.name:filter for filter in AVAILABLE_FILTERS}

try:
    AVAILABLE_NLP_FILTERS = [names_availabels_filters[filter_name] for filter_name in filters_yaml["NLP filters"]]

    AVAILABLE_NLG_FILTERS = [names_availabels_filters[filter_name] for filter_name in filters_yaml["NLG filters"]] + AVAILABLE_NLP_FILTERS

    AVAILABLE_MT_FILTERS =  [names_availabels_filters[filter_name] for filter_name in filters_yaml["Machine Translation filters"]] + AVAILABLE_NLG_FILTERS

    AVAILABLE_SUMMARIZATION_FILTERS = [names_availabels_filters[filter_name] for filter_name in filters_yaml["Summarization filters"]] + AVAILABLE_NLG_FILTERS

    AVAILABLE_DIALOGUE_FILTERS = [names_availabels_filters[filter_name] for filter_name in filters_yaml["Dialogue System filters"]] + AVAILABLE_NLG_FILTERS

    AVAILABLE_CLASSIFICATION_FILTERS = [names_availabels_filters[filter_name] for filter_name in filters_yaml["Classification filters"]]+ AVAILABLE_NLP_FILTERS

except KeyError as error:
    print("Error (yaml): " + str(error) + " as a filter is not available.")
    sys.exit(1)