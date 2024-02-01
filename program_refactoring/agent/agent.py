from pathlib import Path 
import pdb 
import os 
import json 
import subprocess 
import re 
import ast 
import time 
from typing import List, Any, Dict
from tqdm import tqdm

import openai
import chromadb 
from chromadb import Settings
from chromadb.utils import embedding_functions

from program_refactoring.model.model import Model
from program_refactoring.model.openai_model import OpenAIModel, TokenCounter
from program_refactoring.model.hf_model import HFModel, CodeLlamaModel, LemurModel
from program_refactoring.model.prompts import (gpt_logo_agent_prompt, 
                                               gpt_python_agent_prompt,
                                               gpt_textcraft_agent_prompt, 
                                               gpt_retrial_prompt) 
from program_refactoring.model.llama_prompts import (llama_logo_agent_completion_prompt, 
                                                     llama_python_agent_completion_prompt,
                                                     llama_textcraft_agent_completion_prompt,
                                                     llama_retrial_prompt)

from program_refactoring.model.lemur_prompts import (lemur_logo_agent_completion_prompt, 
                                                     lemur_python_agent_completion_prompt,
                                                     lemur_retrial_prompt,
                                                     lemur_textcraft_agent_completion_prompt) 

<<<<<<< Updated upstream
from program_refactoring.tree.node import Node, LogoNode, PythonNode
=======
from program_refactoring.tree.node import Node, LogoNode, PythonNode, LispNode, TextCraftNode  
>>>>>>> Stashed changes
from program_refactoring.domains.logos.utils import clean_import
from program_refactoring.domains.logos.utils import get_func_names as get_logo_func_names
from program_refactoring.domains.python.utils import get_func_names as get_python_func_names
from program_refactoring.domains.textcraft.utils import get_func_names as get_textcraft_func_names
from program_refactoring.codebank.codebank import CodeBank
from program_refactoring.paths import LEMUR_PATH

MODEL_DICT = {"gpt-3.5-turbo": OpenAIModel, 
              "gpt-4-turbo": OpenAIModel,
              "codellama/CodeLlama-7b-Python-hf": CodeLlamaModel, 
              "codellama/CodeLlama-7b-Instruct-hf": CodeLlamaModel,
              "codellama/CodeLlama-13b-Instruct-hf": CodeLlamaModel,
              "codellama/CodeLlama-34b-Instruct-hf": CodeLlamaModel,
              "lemur70b": LemurModel,
              LEMUR_PATH: LemurModel, # add your lemur path here
              "token_counter": TokenCounter} 


COMPLETION_PROMPTS = {"llama": {"logos": llama_logo_agent_completion_prompt,
                                "python": llama_python_agent_completion_prompt,
                                "textcraft": llama_textcraft_agent_completion_prompt},

                      "lemur": {"logos": lemur_logo_agent_completion_prompt,
                                "python": lemur_python_agent_completion_prompt,
                                "textcraft": lemur_textcraft_agent_completion_prompt},

                      "gpt":   {"logos": gpt_logo_agent_prompt,
                                "python": gpt_python_agent_prompt,
                                "textcraft": gpt_textcraft_agent_prompt}
                    }

RETRIAL_PROMPTS = {"gpt": {"textcraft": gpt_retrial_prompt}, 'llama':{'textcraft': llama_retrial_prompt}, "lemur":{'textcraft': lemur_retrial_prompt}}
NODE_DICT = {"logos": LogoNode,
             "python": PythonNode,
             "textcraft": TextCraftNode} 


class Example:
    def __init__(self, id, query, program = None, provenance = None, expected_answer = None):
        self.id = str(id)
        if 'textcraft' in self.id:
            self.idx = self.id.split('_')[-1]
        self.query = query
        self.program = program
        self.provenance  = provenance
        self.expected_answer = expected_answer
        if program is None and expected_answer is None:
            raise ValueError("Must provide either program or expected answer")

class Agent:
    def __init__(self, 
                 train_datas: Dict[str, List[Any]],
                 model: Model, 
                 save_path: Path, 
                 task: str = "logos",
                 dataset: str = "logos",
                 codebank: CodeBank = None,
                 use_docstring_exs: bool = False,
                 use_success: bool = False,
                 use_thought: bool = False,
                 max_budget: int = 5,
                 budget_split: float = 0.6,
                 infilling: bool = False):

        self.train_datas = train_datas

        self.train_data_by_id = {k: {ex.id: ex for ex in train_data} for k, train_data in train_datas.items()} 

        self.save_path = Path(save_path)
        self.codebank = codebank 

        self.max_budget = max_budget 
        self.budget_split = budget_split
        self.infilling = infilling

        # save codebank 
        if codebank is not None:
            self.codebank.temp_dir = self.save_path
            self.codebank.write_to_file()

        self.task = task 
        if task == "logos":
            self.get_func_names = get_logo_func_names
        elif task == "textcraft":
            self.get_func_names = get_textcraft_func_names

        else:
            self.get_func_names = get_python_func_names
        self.dataset = dataset 
        self.use_docstring_exs = use_docstring_exs
        self.use_success = use_success
        self.use_thought = use_thought
        self.repeat_query_in_thought = self.task == "logos"
        self.node_cls = NODE_DICT[self.task]

        # build collection for train data 
        self.model = model
        self.language = "PYTHON"
        self.comment_tok = "#"

        if model.model_name in ["gpt-3.5-turbo", "gpt-4-turbo"]:
            self.model_type = "gpt"
        elif "codellama" in model.model_name:
            self.model_type = "llama"
        elif "lemur" in model.model_name:
            self.model_type = "lemur"

        if self.model_type in ['llama', 'lemur']:
            self.prompt_builder = self.llama_prompt_builder
        elif self.model_type == "gpt":
            self.prompt_builder = self.gpt_prompt_builder
        else:
            raise NotImplementedError(f"Unrecognized model: {model.model_name}")

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ['OPENAI_API_KEY'],
                    model_name="text-embedding-ada-002"
                )
        
        persist_directory = save_path / f"chromadb/{self.task}_agent"
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.Client(Settings(persist_directory=str(persist_directory),
                                                chroma_db_impl="duckdb+parquet",))


        self.collections = {k: None for k in self.train_datas.keys()}
        for k, train_data in self.train_datas.items():

            collection = chroma_client.get_or_create_collection(name=f"{self.task}_agent_{k}" ,embedding_function=openai_ef)

            # add train data to collection
            # filter training data down to examples that use functions that are in the codebank!
            # needed because we don't want to use examples using functions removed from codebank

            docs, ids = self.get_docs_ids(train_data, codebank, do_filter = self.task == "logos")
            collection.add(documents=docs, ids=ids)
            self.collections[k] = collection

    def get_docs_ids(self, train_data, codebank, do_filter = True):
        docs, ids = [], []
        if codebank is not None:
            codebank_funcs = [x for x in codebank._codebank.keys()]
        for ex in train_data:
            if codebank is not None and ex.provenance == "log":
                # in the codebank setting, only keep training examples that use codebank functions and whose functions haven't been removed 
                # only do this for examples from the log; examples from the data always get kept 
                funcs_used = self.get_func_names(ex.program)
                skip = False
                offender = None
                for func in funcs_used:
                    if func not in codebank_funcs:
                        skip = True
                        offender = func
                        break

                if skip and do_filter:
                    print(f"removing {ex.id} because {offender} not in codebank")
                else:
                    if ex.id not in ids and ex.query not in docs:
                        docs.append(ex.query)
                        ids.append(ex.id)
            else:
                # if no codebank, just add all training examples
                if ex.id not in ids and ex.query not in docs:
                    docs.append(ex.query)
                    ids.append(ex.id)

        return docs, ids

    def retrieve_similar(self, data_key, example, k, attempts=0, exclude_current = True):
        # exclude_current: exclude any example that is an EXACT match to the example 
        # retrieve top k from the collection 
        try:
            results = self.collections[data_key].query(query_texts = [example.query], n_results = k) 
        except openai.error.APIError:
            if attempts < 3: 
                time.sleep(10)
                return self.retrieve_similar(example, k, attempts + 1, exclude_current)
        results = results['ids'][0]
        to_ret = [self.train_data_by_id[data_key][id] for id in results]
        if exclude_current:
            to_ret = [ex for ex in to_ret if ex.id != example.id]
        return to_ret

    def llama_prompt_builder(self, example, icl_examples):
        """build prompt for CodeLlama-style models"""
        icl_prompt = ""
        for ex in icl_examples:
            if self.use_thought:
                try:
                    functions_used = self.get_func_names(ex.program)
                except SyntaxError:
                    pdb.set_trace()
                if "embed" in ex.program: 
                    functions_used += ["embed"] 
                if len(functions_used) > 0: 
                    if self.repeat_query_in_thought:
                        thought = f"The query asked for {ex.query} so I will use the helper functions: {', '.join(functions_used)}"
                    else:
                        thought = f"I will use the helper functions: {', '.join(functions_used)}"
                    icl_prompt += f"\n# Query: {ex.query}\n# Thought: {thought}\n# Program:\n{clean_import(ex.program)}\n"
                else:
                    icl_prompt += f"\n# Query: {ex.query}\n# Program:\n{clean_import(ex.program)}\n"

            else:
                icl_prompt += f"\n# Query: {ex.query}\n# Program:\n{clean_import(ex.program)}\n"

        codebank_instr = ""
        if self.codebank is not None:
            codebank_ids = self.codebank.get_relevant(example.query, k = 20) 
            codebank_funcs = [self.codebank.get(id) for id in codebank_ids]

            # codebank_str = [f"{func._name} (description: {re.sub('_', ' ', func._description)})" for func in codebank_funcs if func is not None]
            codebank_str = [func.summarize(include_ex=self.use_docstring_exs, include_success=self.use_success) \
                            for func in codebank_funcs if func is not None]
            codebank_str = "\n".join(codebank_str)

            if len(codebank_str) > 0:
                codebank_instr = f"\nYou can also choose from the following helper functions: \n{codebank_str}\n." 

        llama_agent_prompt = COMPLETION_PROMPTS[self.model_type][self.task]
            # llama_logo_agent_completion_prompt

        prompt = llama_agent_prompt.format(codebank_str = codebank_instr,
                                         icl_string = icl_prompt, 
                                         query = example.query)
        
        return prompt


    def gpt_prompt_builder(self, example, icl_examples): 
        """build prompt for GPT-style models"""
        icl_prompt, thought_and, thought_str = "", "", ""
        for ex in icl_examples:
            if self.use_thought:
                thought_str = "Begin your program with a comment that explains your reasoning. For example, you might write:\n# Thought: the query asks for a line, so I will use the forward() function."
                thought_and = "Thought and "
                functions_used = self.get_func_names(ex.program)
                if "embed" in ex.program: 
                    functions_used += ["embed"] 
                if len(functions_used) > 0: 
                    if self.repeat_query_in_thought:
                        thought = f"The query asked for {ex.query} so I will use the helper functions: {', '.join(functions_used)}"
                    else:
                        thought = f"I will use the helper functions: {', '.join(functions_used)}"
                    icl_prompt += f"\nQuery: {ex.query}\nThought: {thought}\nProgram:\n{clean_import(ex.program)}\n"
                else:
                    icl_prompt += f"\nQuery: {ex.query}\nProgram:\n{clean_import(ex.program)}\n"

            else:
                thought_str = ""
                thought_and = ""
                icl_prompt += f"\nQuery: {ex.query}\nProgram:\n{clean_import(ex.program)}\n"

        codebank_instr = ""
        if self.codebank is not None:
            codebank_ids = self.codebank.get_relevant(ex.query, k = 10) 
            codebank_funcs = [self.codebank.get(id) for id in codebank_ids]

            # codebank_str = [f"{func._name} (description: {re.sub('_', ' ', func._description)})" for func in codebank_funcs if func is not None]
            codebank_str = [func.summarize(include_ex=self.use_docstring_exs, include_success=self.use_success) \
                            for func in codebank_funcs if func is not None]
            codebank_str = "\n".join(codebank_str)

            if len(codebank_str) > 0:
                codebank_instr = f"\nYou can also choose from the following helper functions: \n{codebank_str}\n." 

        gpt_agent_prompt = COMPLETION_PROMPTS['gpt'][self.task]
        # logo_agent_prompt
        prompt = gpt_agent_prompt.format(codebank_str = codebank_instr,
                                         icl_string = icl_prompt, 
                                         query = example.query,
                                         thought_str = thought_str,
                                         thought_and = thought_and)

        return prompt

  
    def build_prompt(self, example, icl_examples):
        return self.prompt_builder(example, icl_examples)


    def __call__(self, example, retry=False, craft_retrieve=False): 
        # retrieve example from function examples 
        # then retrieve example from train 
        if len(self.collections) == 1:
            icl_examples = self.retrieve_similar("train", example, k=self.max_budget, exclude_current=True)
        else:
            n_tc_exs = int(self.budget_split * self.max_budget)
            testcase_examples = self.retrieve_similar("test_cases", example, k = n_tc_exs, exclude_current=True)
            n_train_exs = self.max_budget - n_tc_exs
            train_examples = self.retrieve_similar("train", example, k = n_train_exs, exclude_current=True)
            icl_examples = testcase_examples + train_examples

        # produce program
        pdb.set_trace()
        prompt = self.build_prompt(example, icl_examples)
        output = self.model(prompt, infilling = self.infilling, agent=True, language=self.language, comment_tok = self.comment_tok) 

        if type(output) == int:
            # token counting model
            return None, output

        # execute program 
        try:
            node =  self.node_cls(example.query, output, type="pred", temp_dir=self.save_path, name=f"{example.id}", node_id=f"{example.id}") 
            fname = self.save_path / f"{example.id}_pred.py"
            if self.task == "textcraft" and retry:
              result, err = node.execute(fname, verbose=True)
            else:
              result = node.execute(fname)
            if not result and retry:
                retry_prompt = RETRIAL_PROMPTS[self.model_type][self.task].format(codebank_str=self.codebank_instr, crafting_commands=example.metadata, query=example.query, program=output, exec_trace=err, succ='False')
                output = self.model(retry_prompt, infilling = self.infilling, agent=True, language=self.language, comment_tok = self.comment_tok)
                node =  self.node_cls(example.query, output, type="pred", temp_dir=self.save_path, name=f"{example.id}", node_id=f"{example.id}") 
                fname = self.save_path / f"{example.id}_pred.py"
                result = node.execute(fname)
        except SyntaxError:
            result = None
        except AttributeError:
            result = None
        return result, output

    def do_multiple(self, examples, batch_size=5, rerun=False, exclude_current=True): 
        if not rerun: 
            # retrieve example from function examples 
            # then retrieve example from train 
            prompts = []
            for example in examples:
                if len(self.collections) == 1:
                    icl_examples = self.retrieve_similar("train", example, k=self.max_budget, exclude_current=exclude_current)
                else:
                    n_tc_exs = int(self.budget_split * self.max_budget)
                    testcase_examples = self.retrieve_similar("test_cases", example, k = n_tc_exs, exclude_current=exclude_current)
                    n_train_exs = self.max_budget - n_tc_exs
                    train_examples = self.retrieve_similar("train", example, k = n_train_exs, exclude_current=exclude_current)
                    icl_examples = testcase_examples + train_examples

                # produce program
                prompt = self.build_prompt(example, icl_examples)
                prompts.append(prompt)

            if hasattr(self.model, "run_multiple"):
                outputs = self.model.run_multiple(prompts, batch_size= batch_size, infilling = self.infilling, agent=True, language=self.language, comment_tok = self.comment_tok) 
            else:
                outputs = [self.model(prompt, 
                                    infilling = self.infilling, 
                                    agent = True, 
                                    language = self.language, 
                                    comment_tok = self.comment_tok) for prompt in prompts]
        else:
            outputs = ["skip" for _ in range(len(examples))]
        results = [] 
        print("Executing programs...")
        for example, output in tqdm(zip(examples, outputs), total=len(examples)):
            if type(output) == int:
                # token counting model
                results.append(None)

            # execute program 
            try:
                if output != "skip": 
                    node =  self.node_cls(example.query, 
                                      output, 
                                      type="pred", 
                                      temp_dir=self.save_path, 
                                      name=f"{example.id}", 
                                      node_id=f"{example.id}") 
                    fname = self.save_path / f"{example.id}_pred.py"
                    result = node.execute(fname)
                else:
                    # manually execute 
                    fname = self.save_path / f"{example.id}_pred.py"
                    out, errs = subprocess.Popen(["python", fname], 
                                                 stdout=subprocess.PIPE, 
                                                 stderr=subprocess.PIPE).communicate()
                    
                    errs = errs.decode("utf-8")
                    result = out.decode("utf-8").strip()

            except SyntaxError:
                result = None
            except AttributeError:
                result = None
            except json.decoder.JSONDecodeError: 
                result = None
            results.append(result)
        return results, outputs 



