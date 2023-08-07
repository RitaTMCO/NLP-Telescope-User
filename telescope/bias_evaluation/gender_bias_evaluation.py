import nltk
import spacy
import time

from typing import Tuple, List, Dict
from nltk.tokenize import word_tokenize
from spacy.tokens import Token
from telescope.bias_evaluation.bias_result import BiasResult, MultipleBiasResults
from telescope.bias_evaluation.bias_evaluation import BiasEvaluation
from telescope.testset import MultipleTestset
from telescope.metrics.metric import Metric

class GenderBiasEvaluation(BiasEvaluation):
    name = "Gender"
    available_languages = ["en", "pt"]
    groups = ["neutral", "female", "male"]

    directory = 'telescope/bias_evaluation/data/'
    nltk_languages = {"en":"english", "pt":"portuguese"}
    models = {"en":"en_core_web_sm","pt":"pt_core_news_sm"}
    groups_nlp = {"Neut": "neutral", "Fem" : "female", "Masc":"male"}
    options_bias_evaluation = ["with dataset","with library","with datasets and library"]

    def __init__(self, language: str):
        super().__init__(language)
        self.options_bias_evaluation_funs = {"with dataset": self.find_extract_genders_identify_terms_with_dataset, 
                                             "with library": self.find_extract_genders_identify_terms_with_library, 
                                             "with datasets and library": self.find_extract_genders_identify_terms_combination
                                            }
        self.directory = self.directory + self.language + "/"
        #[{"they":"neutral", "she":"female", "he":"male"},...]
        self.sets_of_gender_terms = self.open_and_read_identify_terms(self.directory + 'gendered_terms.json')
        self.sets_of_occupations = self.open_and_read_identify_terms(self.directory + 'occupations.json')
        self.sets_of_prons_dets = self.open_and_read_identify_terms(self.directory + 'pronouns_determinants.json')
        self.sets_of_suffixes = self.open_and_read_identify_terms(self.directory + 'suffixes.json') 

        self.nlp = spacy.load(self.models[self.language])

        self.nltk_language = self.nltk_languages[self.language]
        nltk.download('punkt')
    
    def dep_language(self, token:Token):
        if self.language == "pt":
            return (token.dep_== "nsubj" or token.dep_== "nsubj:pass" or token.dep_== "obj" or token.dep_ == "iobj" or 
                    token.dep_== "obl" or token.dep_== "obl:agent" or token.dep_ == "nmod") 
        elif self.language == "en":
            return (token.dep_== "nsubj" or token.dep_== "nsubjpass" or token.dep_== "dobj"  or token.dep_ == "iobj" or
                      token.dep_== "pobj" or token.dep_== "agent" or token.dep_ == "nmod")

    def find_extract_genders_identify_terms(self, output_per_sys:Dict[str,List[str]], ref:List[str], option_bias_evaluation:str):
        num_segs = len(ref)
        genders_ref_per_seg = {}
        genders_per_sys_per_seg = {sys_id:{} for sys_id in list(output_per_sys.keys())}

        for seg_i in range(num_segs):
            genders_ref = list()
            genders_per_sys = {sys_id:list() for sys_id in list(output_per_sys.keys())}
            sys_seg_output = {sys_id:sys_output[seg_i] for sys_id, sys_output in output_per_sys.items()}

            _option_fun = self.options_bias_evaluation_funs[option_bias_evaluation]
            
            genders_ref, genders_per_sys = _option_fun(ref[seg_i], sys_seg_output, genders_ref, genders_per_sys)   
            num = len(genders_ref)
            if any(len(genders)!=num for genders in list(genders_per_sys.values())):
                genders_ref = []
                genders_per_sys = {sys_id:list() for sys_id in list(output_per_sys.keys())}

            genders_ref_per_seg[seg_i] = genders_ref
            for sys_id, groups in genders_per_sys.items():
                genders_per_sys_per_seg[sys_id].update({seg_i:groups})
        
        return genders_ref_per_seg,genders_per_sys_per_seg
    

    def score_with_metrics(self, ref:str, sys_output:str, genders_ref:List[str], genders_ref_per_seg:Dict[int,List[str]], text_genders_ref_per_seg:dict, 
                           text_genders_sys_per_seg:dict, init_metrics:List[Metric]) -> BiasResult:
        genders_sys_per_seg = {seg_i: [token_gen["gender"] for token_gen in text_genders] for seg_i, text_genders in text_genders_sys_per_seg.items()}
        genders_sys = [gender for genders in list(genders_sys_per_seg.values()) for gender in genders]
        metrics_results = {metric.name:metric.score([""], genders_sys, genders_ref) for metric in init_metrics}
        return BiasResult(self.groups,ref,sys_output,genders_ref,genders_ref_per_seg,genders_sys,genders_sys_per_seg, text_genders_ref_per_seg,
                          text_genders_sys_per_seg, metrics_results)
    

    def evaluation(self, testset: MultipleTestset, option_bias_evaluation:str) -> MultipleBiasResults:
        """ Gender Bias Evaluation."""
        start = time.time()
        ref = testset.ref
        output_per_sys = testset.systems_output

        text_genders_ref_per_seg, text_genders_per_sys_per_seg = self.find_extract_genders_identify_terms(output_per_sys,ref,option_bias_evaluation)
        genders_ref_per_seg = {seg_i: [token_gen["gender"] for token_gen in text_genders] for seg_i, text_genders in text_genders_ref_per_seg.items()}
        genders_ref = [gender for genders in list(genders_ref_per_seg.values()) for gender in genders]

        init_metrics = [metric(self.language,self.groups) for metric in self.metrics]

        systems_bias_results = {
            sys_id: self.score_with_metrics(ref,sys_output,genders_ref,genders_ref_per_seg,text_genders_ref_per_seg,text_genders_per_sys_per_seg[sys_id],init_metrics)
            for sys_id,sys_output in output_per_sys.items()
            }
        end = time.time()
        return MultipleBiasResults(ref, genders_ref, genders_ref_per_seg, self.groups, systems_bias_results, text_genders_ref_per_seg, self.metrics, (end-start))


#------------------------- Evaluation with datasets ---------------------------------
    def find_gender_from_dataset(self, set_words:List[str], word_k:int):
        
        def find_gender_from_set(word:str,set_of_words:dict):
         # [{"they":"neutral", "she":"female", "he":"male"},...]
            if word in set_of_words:
                gender = set_of_words.get(word)
            else:
                gender = ""
            return gender

        def find_next_word(set_words:List[str], word_k:int):
            if word_k + 1 < len(set_words): 
                return set_words[word_k + 1]
            else:
                return ""
        
        def is_word_in_set(word:str, next_word:str, set:Dict[str,str]):
            if word in set:
                return word, find_gender_from_set(word,set)
            elif word + " " + next_word in set:
                return word + " " + next_word, find_gender_from_set(word + " " + next_word,set)
            return word, ""
    
        # Exemple: suffixes --> woman and man
        def find_suffix(word: str, suffixes: Tuple[str]) -> str:
            bigger_suffix = ""
            for suffix in suffixes:
                if word.endswith(suffix) and len(bigger_suffix) < len(suffix):
                    bigger_suffix = suffix
            return bigger_suffix
        
        sets_of_terms = self.sets_of_prons_dets + self.sets_of_occupations + self.sets_of_gender_terms
        word = set_words[word_k]
        next_word = find_next_word(set_words, word_k)

        # sets_of_terms
        for set in sets_of_terms:
            word, gender = is_word_in_set(word, next_word, set) 
            if gender:    
                return word, gender, set

        # sets_of_suffixes
        for set in self.sets_of_suffixes:
            suffixes = tuple(set.keys())
            if word.endswith(suffixes):
                suffix = find_suffix(word,suffixes)
                gender = find_gender_from_set(suffix,set)
                if gender:    
                    return word, gender, set
        return word, "", {}

    def find_identify_terms_with_dataset(self, text:str):
        words = word_tokenize(text.lower(), language=self.nltk_language)
        num_words = len(words)
        tokens = []
        for word_k in range(num_words):
            word,gender,set = self.find_gender_from_dataset(words, word_k)
            if set:
                tokens.append({"text":word, "gender": gender, "set":set})
        return tokens
    
    def has_match_with_dataset(self, set:Dict[str,str], set_tokens_per_sys:Dict[str,List[list]]):
        return all(any(set == token_sys["set"] for token_sys in set_tokens_sys)
                   for set_tokens_sys in list(set_tokens_per_sys.values())) and set != {}
    
    def is_match_with_dataset(self,set_ref:Dict[str,str],set_sys: Dict[str,str]):
        return set_ref and set_ref == set_sys

    def match_identify_terms_with_dataset(self,set_tokens_ref, set_tokens_per_sys, genders_ref, genders_per_sys):
        for token_ref in set_tokens_ref:
            if self.has_match_with_dataset(token_ref["set"], set_tokens_per_sys):
                for sys_id in list(set_tokens_per_sys.keys()):
                    text_sys = [token_sys["text"] for token_sys in set_tokens_per_sys[sys_id]]
                    if token_ref["text"] in text_sys:
                        genders_per_sys[sys_id].append({"term":token_ref["text"], "gender":token_ref["gender"]})
                        i = text_sys.index(token_ref["text"])
                        set_tokens_per_sys[sys_id].pop(i)
                    else:
                        for token_sys in set_tokens_per_sys[sys_id]:
                            if self.is_match_with_dataset(token_ref["set"],token_sys["set"]):
                                genders_per_sys[sys_id].append({"term":token_sys["text"], "gender":token_sys["gender"]})
                                set_tokens_per_sys[sys_id].remove(token_sys)
                                break
                genders_ref.append({"term":token_ref["text"], "gender":token_ref["gender"]})
        return genders_ref, genders_per_sys

    
    def find_extract_genders_identify_terms_with_dataset(self, ref_seg:str, seg_per_sys:Dict[str,str], genders_ref:List[str], genders_per_sys: Dict[str,List[str]]):     
        set_tokens_ref = self.find_identify_terms_with_dataset(ref_seg)
        set_tokens_per_sys = {sys_id:self.find_identify_terms_with_dataset(seg_per_sys[sys_id]) for sys_id in list(genders_per_sys.keys())}
        genders_ref, genders_per_sys = self.match_identify_terms_with_dataset(set_tokens_ref, set_tokens_per_sys, genders_ref, genders_per_sys)
        return genders_ref, genders_per_sys

#------------------------- Evaluation with library ---------------------------------
    def find_gender_from_morph(self,token:Token):
        gen = token.morph.get("Gender")
        gender = self.groups_nlp[gen[0]]
        return gender
    
    def find_number_from_morph(self,token:Token):
        num = token.morph.get("Number")
        if num:
            return num[0]
        return ""
    
    def find_prontype_person_case_from_morph(self,token:Token):
        per = token.morph.get("Person")
        type = token.morph.get("PronType")
        case = token.morph.get("Case")
        if per and type and case:
            return [per[0],type[0],case[0]]
        return []
    
    def has_gender(self,token:Token):
        return (token.has_morph() and token.morph.get("Gender"))
    
    def find_identify_terms_with_library(self, text:str):
        doc = self.nlp(text.lower())  
        tokens = [{"token":token,"gender":self.find_gender_from_morph(token)} for token in doc if self.has_gender(token) and self.dep_language(token)]
        return tokens
        
    def has_match_with_library(self,token:Token, set_tokens_per_sys:Dict[str,List[dict]]):
        def is_any_pronoum(token:Token, set_tokens_per_sys:Dict[str,List[dict]]):
            if token.pos_ == "PRON":
                return all([self.find_prontype_person_case_from_morph(token), token.dep_]
                           in [[self.find_prontype_person_case_from_morph(token_sys["token"]), token_sys["token"].dep_] 
                                for token_sys in set_tokens_sys if token_sys["token"].pos_ == "PRON"] 
                           for set_tokens_sys in list(set_tokens_per_sys.values())) 
        
        lemma = all([token.lemma_ , self.find_number_from_morph(token), token.pos_, token.dep_] in 
                    [[token_sys["token"].lemma_ ,self.find_number_from_morph(token_sys["token"]), token_sys["token"].pos_, token_sys["token"].dep_] 
                     for token_sys in set_tokens_sys] 
                    for set_tokens_sys in list(set_tokens_per_sys.values()))  
        
        pron = is_any_pronoum(token, set_tokens_per_sys)
        return lemma or pron
    
    def is_match_with_library(self,token_ref:Token, token_sys:Token):
        lemma = ((token_ref.lemma_ == token_sys.lemma_) 
                 and self.find_number_from_morph(token_ref) == self.find_number_from_morph(token_sys)
                 and token_ref.pos_ == token_ref.pos_
                 and token_ref.dep_ == token_sys.dep_)
        pron = (token_ref.pos_ == "PRON" and token_sys.pos_ == "PRON" and 
                 self.find_prontype_person_case_from_morph(token_ref) == self.find_prontype_person_case_from_morph(token_sys))
        return lemma or pron
    
    def match_identify_terms_with_library(self, set_tokens_ref:List[dict], set_tokens_per_sys:Dict[str,List[dict]], genders_ref:List[str], 
                                          genders_per_sys:Dict[str,List[str]]):
        
        for token_ref in set_tokens_ref:
            if self.has_match_with_library(token_ref["token"], set_tokens_per_sys):
                for sys_id, set_tokens_sys in set_tokens_per_sys.items():
                    text_sys = [token_sys["token"].text for token_sys in set_tokens_per_sys[sys_id]]
                    if token_ref["token"].text in text_sys:
                        genders_per_sys[sys_id].append({"term":token_ref["token"].text, "gender":token_ref["gender"]})
                        i = text_sys.index(token_ref["token"].text)
                        set_tokens_per_sys[sys_id].pop(i)
                    else:
                        for token_sys in set_tokens_sys:
                            if self.is_match_with_library(token_ref["token"],token_sys["token"]):
                                genders_per_sys[sys_id].append({"term":token_sys["token"].text, "gender":token_sys["gender"]})
                                set_tokens_per_sys[sys_id].remove({"token":token_sys["token"],"gender":token_sys["gender"]})
                                break
                genders_ref.append({"term":token_ref["token"].text, "gender":token_ref["gender"]})
        return genders_ref, genders_per_sys

    
    def find_extract_genders_identify_terms_with_library(self, ref_seg:str, seg_per_sys:Dict[str,str], genders_ref:List[str], genders_per_sys: Dict[str,List[str]]):     
        set_tokens_ref = self.find_identify_terms_with_library(ref_seg)
        set_tokens_per_sys = {sys_id:self.find_identify_terms_with_library(seg_per_sys[sys_id]) for sys_id in list(genders_per_sys.keys())}
        genders_ref, genders_per_sys = self.match_identify_terms_with_library(set_tokens_ref, set_tokens_per_sys, genders_ref, genders_per_sys)
        return genders_ref, genders_per_sys
    
#------------------------- Evaluation with library and datasets ---------------------------------
    def find_identify_terms_combination(self, text:str):
        doc = self.nlp(text.lower())        
        num_tokens = len(doc)
        set_tokens = [token.text for token in doc]
        tokens_gender = []
        for token_k in range(num_tokens):
            token = doc[token_k]
            if self.dep_language(token):
                term,gender,set = self.find_gender_from_dataset(set_tokens, token_k)
                if not set and self.has_gender(token):
                    gender = self.find_gender_from_morph(token)
                if gender:
                    tokens_gender.append({"token":token, "term":term, "gender":gender, "set":set})
        return tokens_gender     
    
    def has_match_with_combination(self,token_ref:dict,set_tokens_per_sys:Dict[str,List[dict]]):
        return self.has_match_with_dataset(token_ref["set"], set_tokens_per_sys) or self.has_match_with_library(token_ref["token"], set_tokens_per_sys)
    
    def is_match_with_combination(self,token_ref:dict,token_sys:dict):
        return (token_ref["token"].dep_ == token_sys["token"].dep_ and 
                ((self.is_match_with_dataset(token_ref["set"],token_sys["set"]) and token_ref["set"] and token_sys["set"]) or 
                 (self.is_match_with_library(token_ref["token"],token_sys["token"]) and not token_ref["set"] and not token_sys["set"])))
    
    def match_identify_terms_combination(self, set_tokens_ref:List[dict], set_tokens_per_sys:Dict[str,List[dict]], genders_ref:List[str], 
                                          genders_per_sys:Dict[str,List[str]]):
        for token_ref in set_tokens_ref:
            if self.has_match_with_combination(token_ref,set_tokens_per_sys):
                for sys_id, set_tokens_sys in set_tokens_per_sys.items():
                    text_sys = [token_sys["term"] for token_sys in set_tokens_per_sys[sys_id]]
                    if token_ref["term"]  in text_sys:
                        genders_per_sys[sys_id].append({"term":token_ref["term"], "gender":token_ref["gender"]})
                        i = text_sys.index(token_ref["term"])
                        set_tokens_per_sys[sys_id].pop(i)
            
                    else:
                        for token_sys in set_tokens_sys:
                            if self.is_match_with_combination(token_ref, token_sys):
                                genders_per_sys[sys_id].append({"term":token_sys["token"].text, "gender":token_sys["gender"]})
                                set_tokens_per_sys[sys_id].remove(token_sys)
                                break
                genders_ref.append({"term":token_ref["term"], "gender":token_ref["gender"]})
        return genders_ref, genders_per_sys
    
    def find_extract_genders_identify_terms_combination(self, ref_seg:str, seg_per_sys:Dict[str,str], genders_ref:List[str], genders_per_sys: Dict[str,List[str]]):     
        set_tokens_ref = self.find_identify_terms_combination(ref_seg)
        set_tokens_per_sys = {sys_id:self.find_identify_terms_combination(seg_per_sys[sys_id]) for sys_id in list(genders_per_sys.keys())}
        genders_ref, genders_per_sys = self.match_identify_terms_combination(set_tokens_ref, set_tokens_per_sys, genders_ref, genders_per_sys)
        return genders_ref, genders_per_sys