from typing import List, Tuple, Dict
from telescope.testset import Testset, MultipleTestset
from telescope.utils import read_lines

import streamlit as st
import click
import abc

class CollectionTestsets:

    task = "nlp"
    title = "NLP"
    type_of_source = "Source"
    type_of_references = "References"
    type_of_output = "Systems Outputs"
    message_of_success = "Source, References and Outputs were successfully uploaded!"
    last_sys_id = 1
    last_ref_id = 1

    def __init__(
        self,
        src_name: str, #filename of source file 
        refs_names: List[str], #filenames of references files 
        refs_indexes: Dict[str, str],  #id of each reference file. {filename:ref_id} 
        systems_ids: Dict[str, str], #id of each system. {filename:sys_id} 
        systems_names: Dict[str, str], #name of each system. {sys_id:name} 
        filenames: List[str], # all filenames
        testsets: Dict[str, Testset], # set of testsets. Each testset refers to one reference. {reference_filename:testset}
    ) -> None:
        self.src_name = src_name
        self.refs_names = refs_names
        self.refs_indexes = refs_indexes
        self.systems_ids = systems_ids
        self.systems_names = systems_names
        self.filenames = filenames
        self.testsets = testsets
    
    @staticmethod
    def hash_func(collection):
        return " ".join(collection.filenames)

    def indexes_of_systems(self) -> List[str]:
        return list(self.systems_ids.values())
    
    def names_of_systems(self) -> List[str]:
        return list(self.systems_names.values())
    
    def system_name_id(self, name:str) -> str:
        for sys_id, sys_name in self.systems_names.items():
            if sys_name == name:
                return sys_id
        return ""
    
    def already_exists(self, name:str) -> bool:
        return name in self.names_of_systems()
    
    def display_systems(self) -> str:
        text = ""
        for sys_filename, sys_id in self.systems_ids.items():
            text += "--> " + sys_filename + " : " + self.systems_names[sys_id] + " \n"
        return text

    @staticmethod
    def validate_files(src,refs,systems_names,outputs) -> None:
        refs_name = list(refs.keys())
        refs_list = list(refs.values())
        systems_index = list(outputs.keys())
        output_list = list(outputs.values())

        ref_name = refs_name[0]
        ref = refs_list[0]
        x_name = systems_names[systems_index[0]]
        system_x = output_list[0]

        for name, text in zip(refs_name[1:], refs_list[1:]):
           assert len(ref) == len(text), "mismatch between reference {} and reference {} ({} > {})".format(
                ref_name, name, len(ref), len(text))
        assert len(ref) == len(src), "mismatch between references and sources ({} > {})".format(
            len(ref), len(src))
        for y_index, system_y in zip(systems_index[1:], output_list[1:]):
           assert len(system_x) == len(system_y), "mismatch between system {} and system {} ({} > {})".format(
                x_name, systems_names[y_index], len(system_x), len(system_y))
        assert len(system_x) == len(ref), "mismatch between systems and references ({} > {})".format(
            len(system_x), len(ref))
    

    @staticmethod
    def create_testsets(files:list) -> Dict[str, Testset]:
        source_file, sources, _, references, _, _, systems_ids, _, outputs = files
        testsets = {}
        for ref_filename, ref in references.items():
            filenames = [source_file.name] + [ref_filename] + list(systems_ids.keys())
            testsets[ref_filename] = MultipleTestset(sources, ref, outputs, filenames)
        return testsets
    
    @classmethod
    def upload_files(cls) -> list:
        st.subheader("Upload Files for :blue[" + cls.title + "] analysis:")
        
        source_file = st.file_uploader("Upload **one** file with the " + cls.type_of_source)
        sources = read_lines(source_file)

        ref_files = st.file_uploader("Upload **one** or **more** files with the " + cls.type_of_references, accept_multiple_files=True)
        references, refs_indexes = {}, {}
        for ref_file in ref_files:
            if ref_file.name not in references:
                data = read_lines(ref_file)
                ref_id = cls.create_ref_id(ref_file.name,data)
                references[ref_file.name] = data
                refs_indexes[ref_file.name] = ref_id

        outputs_files = st.file_uploader("Upload **one** or **more** files with the " + cls.type_of_output, accept_multiple_files=True)
        systems_ids, systems_names, outputs = {}, {}, {}
        for output_file in outputs_files:
            if output_file.name not in systems_ids:
                output = read_lines(output_file)
                sys_id = cls.create_sys_id(output_file.name,output)
                systems_ids[output_file.name] = sys_id
                if cls.task + "_" + sys_id + "_rename" not in st.session_state:
                    systems_names[sys_id] = sys_id
                else:
                    systems_names[sys_id] = st.session_state[cls.task + "_" + sys_id + "_rename" ]
                outputs[sys_id] = output

        return source_file,sources,ref_files,references,refs_indexes,outputs_files,systems_ids,systems_names,outputs
    
    @classmethod
    def upload_names(cls, systems_names:Dict[str,str]) -> list:
        load_name = st.checkbox('Upload systems names')
        name_file = None
        if load_name:
            name_file = st.file_uploader("Upload the file with systems names", accept_multiple_files=False)
            if name_file != None:
                list_new_names = []
                names = read_lines(name_file)
                for name in names:
                    if name not in list_new_names:
                        list_new_names.append(name)
                        
                if len(list_new_names) >= len(systems_names):
                    i = 0
                    for system_id, _ in systems_names.items():  
                        if cls.task + "_" + system_id + "_rename" not in st.session_state:
                            systems_names[system_id] = list_new_names[i]
                        i += 1
        return systems_names,load_name,name_file 
   

    @classmethod
    @st.cache
    def create_sys_id(cls,filename,output):
        sys_id = "Sys " + str(cls.last_sys_id)
        cls.last_sys_id += 1
        return sys_id
    
    @classmethod
    @st.cache
    def create_ref_id(cls,filename,output):
        ref_id = "Ref " + str(cls.last_ref_id)
        cls.last_ref_id += 1
        return ref_id
    
    @classmethod
    @abc.abstractmethod
    def read_data(cls):
        return NotImplementedError

    @classmethod
    def read_data_cli(cls, source:click.File, system_names_file:click.File, systems_output:Tuple[click.File], reference:Tuple[click.File], 
                      extra_info:str="", labels_file:click.File=None): 
    
        systems_ids, systems_names, outputs = {}, {}, {}
        
        if system_names_file:
            sys_names = [l.replace("\n", "") for l in system_names_file.readlines()]

            if len(sys_names) < len(systems_output):
                for i in range(len(systems_output)-len(sys_names)):
                    sys_names.append("Sys " + str(i+1))
            
            id = 1
            for sys_file,sys_name in zip(systems_output,sys_names):
                data = [l.strip() for l in sys_file.readlines()]
                sys_id = "Sys " + str(id)
                id += 1
                systems_ids[sys_file.name] = sys_id
                systems_names[sys_id] = sys_name
                outputs[sys_id] = data
            
        else:
            id = 1
            for sys_file in systems_output:
                data = [l.strip() for l in sys_file.readlines()]
                sys_id = "Sys " + str(id)
                id += 1
                systems_ids[sys_file.name] = sys_id
                systems_names[sys_id] = sys_id
                outputs[sys_id] = data
        
        id = 1
        references,refs_indexes = {},{}
        for ref in reference:
            if ref.name not in references:
                data = [l.strip() for l in ref.readlines()]
                ref_id = "Ref " + str(id)
                id += 1
                references[ref.name] = data
                refs_indexes[ref.name] = ref_id

        src = [l.strip() for l in source.readlines()]

        if labels_file:
            extra_info = [l.strip() for l in labels_file.readlines()]

        cls.validate_files(src,references,systems_names,outputs)

        files = [source,src,reference,references,refs_indexes,systems_output,systems_ids,systems_names,outputs] 

        testsets = cls.create_testsets(files)

        return cls(source.name, references.keys(), refs_indexes, systems_ids, systems_names,
                    [source.name] +  list(references.keys()) + list(systems_ids.values()), testsets, extra_info)
    

class NLGTestsets(CollectionTestsets):
    task = "nlg"
    title = "NLG"
    def __init__(
        self,
        src_name: str,
        refs_names: List[str],
        refs_indexes: Dict[str, str],
        systems_ids: Dict[str, str],
        systems_names: Dict[str, str],
        filenames: List[str],
        testsets: Dict[str, Testset],
        language_pair: str
    ) -> None:
        super().__init__(src_name, refs_names, refs_indexes, systems_ids, systems_names, filenames, testsets)
        self.language_pair = language_pair.lower()
    
    @property
    def source_language(self):
        return self.language_pair.split("-")[0]

    @property
    def target_language(self):
        return self.language_pair.split("-")[1]
    
    @staticmethod
    def upload_language() -> str:
        language_pair = ""
        language = st.text_input(
            "Please input the language of the files to analyse (e.g. 'en'):",
            "", 
            help=("If the language is indifferent and BERTScore metric is not used, then write X.")
        )
        if language != "":
            language_pair = language + "-" + language
        return language_pair
    
    @classmethod
    def read_data(cls):
        files = cls.upload_files()

        language = cls.upload_language()
        
        source_file,sources,ref_files,references, refs_indexes, outputs_files,systems_ids, systems_names, outputs = files

        systems_names,load_name, file_sys_names= cls.upload_names(systems_names)

        if ((ref_files != []) 
            and (source_file is not None) 
            and (outputs_files != []) 
            and (language != "")
            and ( not load_name or (load_name and file_sys_names is not None))):

            cls.validate_files(sources,references,systems_names,outputs)
            st.success(cls.message_of_success)

            testsets = cls.create_testsets(files)

            return cls(source_file.name, references.keys(), refs_indexes, systems_ids, systems_names,
                [source_file.name] +  list(references.keys()) + list(systems_ids.values()),
                testsets, language)


class MTTestsets(NLGTestsets):
    task = "machine-translation"
    title = "Machine Translation"
    type_of_output = "Systems Translations"
    message_of_success = "Source, References, Translations and LP were successfully uploaded!"
    def __init__(
        self,
        src_name: str,
        refs_names: List[str],
        refs_indexes: Dict[str, str],
        systems_ids: Dict[str, str],
        systems_names: Dict[str, str],
        filenames: List[str],
        testsets: Dict[str, MultipleTestset],
        language_pair: str,
    ) -> None:
        super().__init__(src_name, refs_names, refs_indexes, systems_ids, systems_names, filenames,
                testsets, language_pair)
    
    @staticmethod
    def upload_language():
        language_pair = st.text_input(
            "Please input the language pair of the files to analyse (e.g. 'en-ru'):",
            "", help=("If the language is indifferent and BERTScore metric is not used, then write X-X.")
        )
        return language_pair


class SummTestsets(NLGTestsets):
    task = "summarization"
    title = "Summarization"
    type_of_source = "Text to be summarized"
    type_of_output = "Systems Summaries"
    message_of_success = "Source, References, Summaries and Language were successfully uploaded!"
    def __init__(
        self,
        src_name: str,
        refs_names: List[str],
        refs_indexes: Dict[str, str],
        systems_ids: Dict[str, str],
        systems_names: Dict[str, str],
        filenames: List[str],
        testsets: Dict[str, MultipleTestset],
        language_pair: str,
    ) -> None:
        super().__init__(src_name, refs_names, refs_indexes, systems_ids, systems_names, filenames, 
                testsets, language_pair)
        
    @staticmethod
    def validate_files(src,refs,systems_names,outputs):
        refs_name = list(refs.keys())
        refs_list = list(refs.values())
        systems_index = list(outputs.keys())
        output_list = list(outputs.values())

        ref_name = refs_name[0]
        ref = refs_list[0]
        x_name = systems_names[systems_index[0]]
        system_x = output_list[0]

        for name, text in zip(refs_name[1:], refs_list[1:]):
           assert len(ref) == len(text), "mismatch between reference {} and reference {} ({} > {})".format(
                ref_name, name, len(ref), len(text))
        for y_index, system_y in zip(systems_index[1:], output_list[1:]):
           assert len(system_x) == len(system_y), "mismatch between system {} and system {} ({} > {})".format(
                x_name, systems_names[y_index], len(system_x), len(system_y))
        assert len(system_x) == len(ref), "mismatch between systems and references ({} > {})".format(
            len(system_x), len(ref))
    

class DialogueTestsets(NLGTestsets):
    task = "dialogue-system"
    title = "Dialogue System"
    type_of_source = "Context"
    type_of_references = "Truth Answers"
    type_of_output = "Systems Answers"
    message_of_success = "Source, References, Dialogues and Language were successfully uploaded!"
    def __init__(
        self,
        src_name: str,
        refs_names: List[str],
        refs_indexes: Dict[str, str],
        systems_ids: Dict[str, str],
        systems_names: Dict[str, str],
        filenames: List[str],
        testsets: Dict[str, MultipleTestset],
        language_pair: str
    ) -> None:
        super().__init__(src_name, refs_names, refs_indexes, systems_ids, systems_names, filenames, 
                testsets, language_pair)
   
    @staticmethod
    def validate_files(src,refs,systems_names,outputs):
        refs_name = list(refs.keys())
        refs_list = list(refs.values())
        systems_index = list(outputs.keys())
        output_list = list(outputs.values())
        ref_name = refs_name[0]
        ref = refs_list[0]
        x_name = systems_names[systems_index[0]]
        system_x = output_list[0]
        for name, text in zip(refs_name[1:], refs_list[1:]):
           assert len(ref) == len(text), "mismatch between reference {} and reference {} ({} > {})".format(
                ref_name, name, len(ref), len(text))
        for y_index, system_y in zip(systems_index[1:], output_list[1:]):
           assert len(system_x) == len(system_y), "mismatch between system {} and system {} ({} > {})".format(
                x_name, systems_names[y_index], len(system_x), len(system_y))
        assert len(system_x) == len(ref), "mismatch between systems and references ({} > {})".format(
            len(system_x), len(ref))



class ClassTestsets(CollectionTestsets):
    task = "classification"
    title = "Classification"
    type_of_source = "Samples"
    type_of_references = "True Labels"
    type_of_output = "Predicated Labels"
    message_of_success = "Source and Labels were successfully uploaded!"

    def __init__(
        self,
        src_name: str,
        refs_names: List[str],
        refs_indexes: Dict[str, str],
        systems_ids: Dict[str, str],
        systems_names: Dict[str, str],
        filenames: List[str],
        testsets: Dict[str, MultipleTestset],
        labels: List[str]
    ) -> None:
        super().__init__(src_name, refs_names, refs_indexes, systems_ids, systems_names, filenames, testsets)
        self.labels = labels
    
    @staticmethod
    def upload_labels() -> List[str]:
        labels_file = st.file_uploader("Upload the file with the existing labels separated by line", accept_multiple_files=False)
        return labels_file
    
    @classmethod
    def read_data(cls):
        files = cls.upload_files()
        source_file,sources,ref_files,references,refs_indexes,outputs_files,systems_ids,systems_names,outputs = files
        labels_file = cls.upload_labels()

        systems_names,load_name, file_sys_names= cls.upload_names(systems_names)

        if ((ref_files != []) 
            and (source_file is not None) 
            and (outputs_files != []) 
            and (labels_file is not None)
            and ( not load_name or (load_name and file_sys_names is not None))):
            
            cls.validate_files(sources,references,systems_names,outputs)
            st.success(cls.message_of_success)

            testsets = cls.create_testsets(files)

            labels = read_lines(labels_file)

            return cls(source_file.name, references.keys(), refs_indexes, systems_ids, systems_names,
                [source_file.name] +  list(references.keys()) + list(systems_ids.values()),
                testsets, labels)