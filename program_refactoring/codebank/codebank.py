import re 
import os 
import pdb 
import ast 
import json 
from pathlib import Path
from collections import defaultdict
import logging 
import copy
import time 
from typing import Any 

import openai
import numpy as np
import chromadb 
from chromadb import Settings
from chromadb.utils import embedding_functions

from program_refactoring.model.prompts import (logo_codebank_refactor_prompt, 
                                               logo_codebank_single_refactor_prompt,
                                               python_codebank_refactor_prompt,
                                               python_codebank_single_refactor_prompt,
                                               textcraft_codebank_refactor_prompt,
                                               textcraft_codebank_single_refactor_prompt,
                                               logo_codebank_failure_explanation_prompt,
                                               codebank_deduplication_prompt,
                                               codebank_comment_prompt)

from program_refactoring.model.model import Model
from program_refactoring.tree.node import LogoNode, TextCraftNode
from program_refactoring.domains.logos.utils import make_pass_fail_str as make_logo_pass_fail_str


from program_refactoring.domains.logos.utils import get_func_names as get_logo_func_names
from program_refactoring.domains.python.utils import get_func_names as get_python_func_names
from program_refactoring.domains.textcraft.utils import get_func_names as get_textcraft_func_names
from program_refactoring.codebank.test_case import LogoTestCase, PythonTestCase, TextCraftTestCase
from program_refactoring.codebank.function import Function
from program_refactoring.headers import LOGO_HEADER, SIMPLE_LOGO_HEADER, PYTHON_HEADER, TEXTCRAFT_HEADER
from program_refactoring.utils import clean_header

FUNC_NAME_BY_KEY = {"logos": get_logo_func_names, 
                    "python": get_python_func_names,
                    "textcraft": get_textcraft_func_names} 


logger = logging.getLogger(__name__)

np.random.seed(12)

SINGLE_PROMPTS = {"logos": logo_codebank_single_refactor_prompt,
                  "python": python_codebank_single_refactor_prompt,
                  "textcraft": textcraft_codebank_single_refactor_prompt} 

SINGLE_FEEDBACK_PROMPTS = {"logos": logo_codebank_failure_explanation_prompt}

PASS_FAIL_MAKERS = {"logos": make_logo_pass_fail_str,
                    "python": make_logo_pass_fail_str,
                    "textcraft": make_logo_pass_fail_str} 


class CodeBank:

    def __init__(self, 
                name: str, 
                model: Model,
                skip_collection: bool = False,
                run_dir: Path = None,
                temp_dir: Path = None,
                task: str = "logos",
                use_modular: bool = False,
                use_explanation: bool = False,
                wide_refactor: bool = False): 
        
        self.model = model
        # a dictionary of Functions, keyed by name
        self._name = name
        self._codebank = {}
        self._imports = []

        self.use_modular = use_modular
        self.single_func_refactor_prompt = SINGLE_PROMPTS[task] 

        self.use_explanation = use_explanation
        self.explanation_prompt = logo_codebank_failure_explanation_prompt
        self.make_pass_fail_str = PASS_FAIL_MAKERS[task]
        self.wide_refactor = wide_refactor

        self.deduplication_prompt = codebank_deduplication_prompt 
        self.comment_prompt = codebank_comment_prompt

        # a dictionary of Functions, keyed by description
        self._by_description = {}
        # a header added to the codebank file 
        if task == "logos":
            self._header = LOGO_HEADER
        elif task == "textcraft":
            self._header = TEXTCRAFT_HEADER
        else:
            self._header = ""

        if task == "python":
            self._imports = ["from datetime import *", "from dateutil.relativedelta import *"]


        if temp_dir is not None:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = "temp" 

        self._skip_collection = skip_collection

        self.task = task
        try: 
            self.get_func_names = FUNC_NAME_BY_KEY[task]
        except KeyError:
            raise ValueError(f"Invalid task type: {self.task}")

        if not skip_collection:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                        api_key=os.environ['OPENAI_API_KEY'],
                        model_name="text-embedding-ada-002"
                    )
            if run_dir is None and len(logger.manager.root.handlers) > 0:
                logdir = Path(logger.manager.root.handlers[0].baseFilename.split(".log")[0]) 
            else:
                # running at agent time 
                logdir = Path(run_dir) 

            persist_directory = logdir / f"chromadb/{name}"
            persist_directory.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.Client(Settings(persist_directory=str(persist_directory),
                                                    chroma_db_impl="duckdb+parquet",))

            self.collection = self.chroma_client.get_or_create_collection(name=name, embedding_function=openai_ef)

        self.removed = []
        self.winners = []


    def safe_upsert(self, documents, ids, attempts=0):
        try:
            self.collection.upsert(documents=documents, ids=ids)
        except openai.error.APIError:
            # automatically retry 3x 
            if attempts < 3:
                time.sleep(10)
                self.safe_upsert(documents, ids, attempts + 1)

    def safe_query(self, query_texts, n_results, attempts=0):
        try:
            results = self.collection.query(query_texts = query_texts, n_results = n_results) 
        except openai.error.APIError:
            # automatically retry 3x 
            if attempts < 3:
                time.sleep(10)
                return self.safe_query(query_texts, n_results, attempts + 1)
        except:
            # sometimes the query fails for unknown reasons, just return the best you can 
            return self.collection.peek()
        return results 
    
    def add(self, function_str, round_added): 
        """Add a function to the codebank"""
        func = Function.from_str(function_str, round_added)

        # don't add duplicate functions 
        if func._name not in self._codebank.keys():
            self._codebank[func._name] = func
            self._by_description[func._description] = func

            # add description to collection 
            docs = [func._description]
            ids = [func._name]

            if not self._skip_collection: 
                self.safe_upsert(documents=docs, ids=ids)

        return func._name

    def add_multiple(self, big_str, round_added = None): 
        """Add a python file with potentially multiple functions"""

        # extract imports 
        parsed = ast.parse(big_str)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                self._imports.append(ast.unparse(node))

        # remove imports 
        big_str = re.sub("^from .* import .*$", "", big_str, flags=re.MULTILINE)
        big_str = re.sub("^import .*$", "", big_str, flags=re.MULTILINE) 
        # split into functions using string so we keep our comments 
        functions = re.split(r"(def \w+\(.*\):[.\n]*?)", big_str, flags=re.MULTILINE)
        # remove empty strings
        functions = [x for x in functions if x.strip() != ""]
        # recombine into functions
        functions = ["".join(x) for x in zip(functions[::2], functions[1::2])]

        added = []
        for func_str in functions:
            try:
                parsed = ast.parse(func_str)
            except SyntaxError:
                continue

            # check if it is a function, if not skip
            if not isinstance(parsed.body[0], ast.FunctionDef):
                continue

            name = self.add(func_str, round_added) 
            added.append(name)

        return added 

    def get(self, key):
        return self._codebank.get(key, None) 

    def get_relevant(self, query_desc, k=20, attempt = 0): 
        """Return a list of relevant functions"""
        try:
            results = self.safe_query(query_texts = [query_desc], n_results = k) 
        except openai.error.APIError:
            # automatically retry 3x 
            if attempt < 3:
                time.sleep(10)
                return self.get_relevant(query_desc, k, attempt + 1)

        return set(results['ids'][0])
    
    def save(self, path):
        # save _codebank the header 

        with open(path, "w") as f1:
            for imp in set(self._imports):
                f1.write(imp + "\n")
            f1.write(self._header + "\n")
            for func in self._codebank.values(): 
                f1.write(func._original_code + "\n\n")
        # also write the success info, might be nice for agent later  
        path = Path(path)
        with open(path.parent / "success_info.json", "w") as f1:
            to_write = []
            for func in self._codebank.values():
                line = {"name": func._name, "was_success": func.was_success, "num_programs_used": func.num_programs_used, "round_added": func.round_added}
                to_write.append(line)
            json.dump(to_write, f1)

        with open(path.parent / "test_cases.jsonl", "w") as f1:
            to_write = []
            for func in self._codebank.values():
                line = [x.to_json() for x in func.test_cases]
                try:
                    f1.write(json.dumps(line) + "\n")
                except TypeError:
                    pdb.set_trace()
                

    @classmethod
    def load(cls, pypath, succpath, testcase_path, name, model, run_dir = None, temp_dir = None, tc_class=LogoTestCase, task="logos"):
        # load from a checkpointed file 
        codebank = cls(name, model, run_dir = run_dir, temp_dir = temp_dir, task = task) 
        with open(pypath) as f1:
            data = f1.read()
        data = clean_header(codebank._header, data)
        codebank.add_multiple(data)

        # read success info
        with open(succpath) as f1:
            succ_data = json.load(f1)

        # read test cases 
        if testcase_path.exists():
            with open(testcase_path) as f1:
                testcase_data = [json.loads(x) for x in f1.readlines()]
        else:
            testcase_data = [[None] for _ in range(len(succ_data))]
        

        # update function counts 
        for line, tc_line in zip(succ_data, testcase_data): 
            try:
                codebank._codebank[line['name']].round_added = line['round_added']
                codebank._codebank[line['name']].was_success = line['was_success']
                codebank._codebank[line['name']].num_programs_used = line['num_programs_used']
                # add test cases
                test_cases = []
                for tc in tc_line:
                    if tc is not None:
                        tc = tc_class.from_json(tc)
                        test_cases.append(tc) 
                codebank._codebank[line['name']].test_cases = test_cases
            except KeyError:
                # when I edit the codebank for oracle numbers, it can break this process
                # we can skip for now 
                continue
        return codebank 

    @classmethod
    def clone(cls, codebank): 
        new_codebank = cls(codebank._name, codebank.model, skip_collection=True) 
        new_codebank._header = codebank._header
        new_codebank._codebank = {k:v for k,v in codebank._codebank.items()} 
        for k,v in new_codebank._codebank.items():
            assert(v.was_success == codebank._codebank[k].was_success)

        new_codebank._by_description = {k:v for k,v in codebank._by_description.items()}
        new_codebank.removed = codebank.removed
        new_codebank.temp_dir = codebank.temp_dir 
        return new_codebank 

    def write_to_file(self): 

        with open(f"{self.temp_dir}/codebank.py", "w") as f1:
            f1.write(self._header + "\n") 
            imports = set(self._imports)
            for imp in imports:
                f1.write(imp + "\n")
            for func in self._codebank.values(): 
                f1.write(func._original_code + "\n\n")

    def filter(self, round_idx, success_thresh=0.0, min_usage = 4.0, keep_low_usage=True, max_round_delta = 40):
        # compute success percentages for each function 
        funcs_to_keep = []
        for func in self._codebank.values():
            round_added = func.round_added
            if round_added is None:
                round_added = round_idx
            round_delta = round_idx - round_added
            success, n_used = func.compute_success()
            if success > success_thresh and n_used >= min_usage:
                logger.info(f"\t{func._name} has success score of {success} and was used {n_used} times, keeping")
                funcs_to_keep.append(func._name)
            elif n_used < min_usage and n_used > 0 and round_delta > max_round_delta:
                logger.info(f"\t{func._name} has success score of {success} and was used {n_used} times. It has been {round_delta} since it was introduced, removing")
            elif n_used < min_usage and n_used > 0 and keep_low_usage:
                logger.info(f"\t{func._name} has success score of {success} and was used {n_used} times, but was used too little to know if good, keeping")
                funcs_to_keep.append(func._name)
            else:
                logger.info(f"\t{func._name} has success score of {success} and was used {n_used} times, removing")

        # remove anything that doesn't succeed more than half the time (thresh = 0.0)
        removed = [func for func in self._codebank.values() if func._name not in funcs_to_keep]
        self._codebank = {k:v for k,v in self._codebank.items() if k in funcs_to_keep}
        # remove from the collection 
        self.collection.delete(ids=[func._name for func in removed])
        # add to removed to use as negative examples in refactor and later on 
        self.removed = removed 
        return removed 


    def parse_refactor_output(self, output, round_added):

        # clean up output, only take inside of ```...```
        try:
            output = re.search(r"```(.*)```", output, re.DOTALL).group(1)
        except AttributeError:
            # no ``` found so take the whole thing
            pass 

        output = output.strip()
        # remove any extra ``` 
        output = re.sub(r"```", "", output) 
        # remove bullet points because model includes them sometimes
        output = re.sub(r"^\- .+$", "", output, flags=re.MULTILINE)

        mapping = {}

        # split into functions 
        try:
            new_parsed = ast.parse(output) 
        except SyntaxError:
            return mapping

        for func in new_parsed.body:
            if isinstance(func, ast.FunctionDef):
                name = func.name 
                func = Function.from_str(ast.unparse(func), round_added)
            else:
                continue 
            
            mapping[name] = func

        return mapping 


    def single_func_refactor(self, func, passing_cases, failing_cases, header=SIMPLE_LOGO_HEADER, round_added = None):
        prompt = self.single_func_refactor_prompt

        library_str = header

        codebank_str = ""
        query = func._description
        codebank_ids = self.get_relevant(query, k = 10) 
        codebank_funcs = [self.get(id) for id in codebank_ids]

        # codebank_str = [f"{func._name} (description: {re.sub('_', ' ', func._description)})" for func in codebank_funcs if func is not None]
        codebank_str = [func.summarize() for func in codebank_funcs if func is not None]
        codebank_str = "\n".join(codebank_str)

        pass_fail_str = self.make_pass_fail_str(func, passing_cases, failing_cases)
        if self.use_modular:
            modular_str = """Common refactoring operations:
- add_parameter: add a parameter to a function
- adjust_loop: adjust the range of a loop in the function 
- rename_function: rename a function
- edit_body: edit the body of a function
- merge_functions: merge two functions
Before refactoring, please describe the operations you will perform in the following format, followed by the new program:"""    
        else:
            modular_str = ""

        def extract_output(output):
            # search for NEW PROGRAM
            match = re.search(r"NEW PROGRAM:?\s*```(.*?)```", output, re.DOTALL)
            if match is not None:
                output = match.group(1)
                output = re.sub(r"```", "", output)
                output = output.strip()
                return output 
            match = re.search(r"NEW PROGRAM:?(.*?)", output, re.DOTALL)
            if match is not None:
                output = match.group(1)
                output = re.sub(r"```", "", output)
                output = output.strip()
                return output 
            match = re.search(r"```python(.*?)```", output, re.DOTALL)
            if match is not None:
                return match.group(1)
            match = re.search(r"```(.*?)```", output, re.DOTALL)
            if match is not None:
                return match.group(1)
            return output 
        
        prompt = prompt.format(library_str = library_str, 
                               codebank_str=codebank_str, 
                               func_str=func._original_code, 
                               pass_fail_str=pass_fail_str,
                               modular_str=modular_str)
        logger.info(f"Single function refactor prompt:\n{prompt}")
        if self.wide_refactor:
            outputs = self.model.wide_call(prompt, agent=False)
        else: 
            output = self.model(prompt, agent=False)
            output = output.strip()
            outputs = [output]
            logger.info(f"Single function refactor output:\n{output}")
        outputs = [extract_output(output) for output in outputs]
        new_func_mappings = []
        for i, output in enumerate(outputs):
            logger.info(f"EXTRACTED OUTPUT {i}:\n{output}")

            try:
                new_func_mapping = self.parse_refactor_output(output, round_added)
                # new_func = Function.from_str(output)
            except (SyntaxError, ValueError, IndexError) as e:
                logger.info(f"Unable to parse:\n{output}")
                continue
            new_func_mappings.append(new_func_mapping)
            # return None
        return new_func_mappings


    def refactor_one_function(self, func_name, func, n_iter=1, header=SIMPLE_LOGO_HEADER, task = "logos", round_added = None):
        def check_args(func1, func2):
            return func1._args != func2._args or func1._name != func2._name

        to_update = []
        try:
            orig_func = self._codebank[func_name]
        except KeyError:
            # if func is new, just add it 
            logger.info(f"Adding new function: {func_name}")
            to_update.append(func_name)
            called = [func_name]
            return to_update, called


        test_cases = self._codebank[func_name].test_cases
        # for now, only refactor one at a time 

        new_test_cases = []
        new_passing = []
        old_passing_perc = np.mean(self._codebank[func_name].was_success)

        # split into passing and failing cases
        passing_cases = [tc for i, tc in enumerate(test_cases) if self._codebank[func_name].was_success[i]]
        failing_cases = [tc for i, tc in enumerate(test_cases) if not self._codebank[func_name].was_success[i]]

        # dedup test cases: remove any test cases where the query, program, and outcome are identical 
        # doing this based on query AND program since sometimes the same query has multiple programs 
        deduped_passing_cases, deduped_failing_cases = [], []
        for pc in passing_cases:
            all_queries = [tc.pred_node.query for tc in deduped_passing_cases]
            all_progs = [tc.pred_node.program for tc in deduped_passing_cases]
            if pc.pred_node.query not in all_queries and pc.pred_node.program not in all_progs:
                deduped_passing_cases.append(pc)
        for fc in failing_cases:
            all_queries = [tc.pred_node.query for tc in deduped_failing_cases]
            all_progs = [tc.pred_node.program for tc in deduped_failing_cases]
            if fc.pred_node.query not in all_queries and fc.pred_node.program not in all_progs:
                deduped_failing_cases.append(fc)
        passing_cases = deduped_passing_cases
        failing_cases = deduped_failing_cases

        if len(failing_cases) == 0:
            logger.info(f"Function {func._name} is passing all tests, no refactor needed for now")
            called = [x for tc in passing_cases for x in self.get_func_names(tc.pred_node.program)] + [func_name]
            return None, called

        logger.info(f"Before refactor, {func._name} is passing {old_passing_perc * 100:.1f}% of cases")

        func_to_refactor = func  
        updated = False
        old_called = []

        for iteration in range(n_iter):
            # refactor each function separately n times as long as the test acc improves  

            # refactor program 
            new_func_mappings = self.single_func_refactor(func_to_refactor, passing_cases, failing_cases, header, round_added = round_added)
            if len(new_func_mappings) == 0:
                # misparsed! 
                logger.info(f"Could not parse any outputs")
                break
                
            # iterate through all mappings
            for new_func_mapping in new_func_mappings:

                new_passing_cases, new_failing_cases = [], []
                for i, test_case in enumerate(test_cases):
                    test_case_copy_before = copy.deepcopy(test_case)
                    test_case_copy_after = copy.deepcopy(test_case)
                    logger.info(f"Test case: {test_case_copy_before.pred_node.query}")
                    logger.info(f"Code before:\n{test_case_copy_before.pred_node.program}")
                    passing_before = test_case_copy_before.get_acc(task = task, overwrites = [func_to_refactor])
                    logger.info(f"Passing: {passing_before}")
                    if iteration == 0:
                        # get functions originally called in all test cases 
                        old_called += self.get_func_names(test_case_copy_before.pred_node.program)

                    # refactor test case if needed 
                    old_to_new_mapping = []
                    sigs_differ = []
                    for new_fn in new_func_mapping.keys():
                        try:
                            new_code = new_func_mapping[func_name]._original_code
                            try:
                                signature_differs = check_args(self._codebank[new_fn], new_func_mapping[new_fn])
                                original_code = self._codebank[new_fn]._original_code
                            except KeyError:
                                # new function 
                                signature_differs = True
                                original_code = "None"
                        except KeyError:
                            # renamed an existing function 
                            new_code = new_func_mapping[new_fn]._original_code
                            original_code = self._codebank[func_name]._original_code
                            signature_differs = True

                        sigs_differ.append(signature_differs)
                        if signature_differs:
                            old_to_new_mapping.append((original_code, new_code)) 
                    if any(sigs_differ):
                        test_case_copy_after.refactor(old_to_new_mapping)

                    logger.info(f"Code after:\n{test_case_copy_after.pred_node.program}")
                    # check if passing while using the new codebank 
                    passing_after = test_case_copy_after.get_acc(task = task, overwrites = list(new_func_mapping.values()))
                    logger.info(f"Passing after: {passing_after}")
                    if passing_after: 
                        logger.info("Updating test case...")
                        new_passing_cases.append(test_case_copy_after)
                    else:
                        new_failing_cases.append(test_case_copy_after)

                if len(passing_cases) >= len(new_passing_cases): 
                    # if fewer cases pass after, don't continue refactoring
                    # backtrack to previous function 
                    logger.info(f"Old function had more/same passing cases: {len(passing_cases)} > {len(new_passing_cases)}")
                    new_func = func_to_refactor
                    continue
                else:
                    # otherwise, update and repeat 
                    logger.info(f"New version is better, updating!")
                    try:
                        func_to_refactor = new_func_mapping[func_name]
                    except KeyError:
                        # if the function isn't there, break
                        continue

                    updated = True
                    test_cases = new_passing_cases + new_failing_cases
                    # keep iterating over mappings until max is reached 

        called = []
        if updated: 
            logger.info(f"Replacing {func_name} with refactored version!")
            # update test cases
            for i, test_case_copy in enumerate(test_cases):
                self._codebank[func_name].test_cases[i] = test_case_copy
                called += self.get_func_names(test_case_copy.pred_node.program)


            # get success data
            # NOTE (elias): right now, not updating success data since the function has changed
            # so old success/failure is not as informative 

            # update function 
            names = []
            for new_func_name, new_func in new_func_mapping.items():
                names.append(new_func_name)
                try:
                    before = self._codebank[new_func_name]._original_code
                except KeyError:
                    # adding a new helper
                    logger.info(f"adding new helper: {new_func_name}")
                    before = "NONE"
                after = new_func._original_code
                logger.info(f"{func_name} was \n{before}\n Now: \n{after}\n========================")
                self._codebank[new_func_name] = new_func

            return names, called
        else:
            logger.info(f"Refactored version of {func_name} was worse")
            return [], old_called



    def refactor(self, success_thresh=0.0, min_usage = 4.0, double_refactor = False, do_filter=True, header = SIMPLE_LOGO_HEADER, task = "logos", round_added=None):
        """Refactor the codebank to improve generalization"""

        if do_filter:
            self.filter(success_thresh=success_thresh, min_usage=min_usage)


        # refactor each function separately
        mapping = {func._name: func for func in self._codebank.values()}

        to_update = []
        passing_funcs_called = []
        for func_name, func in mapping.items():
            result, called = self.refactor_one_function(func_name, func, header = header, task = task, round_added = round_added) 
            passing_funcs_called += called
            if result is None:
                continue
            to_update.extend(result) 


        passing_funcs_called = set(passing_funcs_called)


        # update codebank, adding new functions 
        for fn in to_update: 
            if fn not in self._codebank.keys():
                logger.info(f"Adding {fn}")
                self.add(mapping[fn]._original_code)

        # delete anything that is now obsolete, i.e. it is never called in the passing test cases or the codebank 
        funcs_called_in_passing_tests = set(passing_funcs_called)

        funcs_called_in_codebank = []
        for func in self._codebank.values():
            code = func._original_code
            funcs_called_in_codebank.extend(self.get_func_names(code))

        funcs_called_in_codebank = set(funcs_called_in_codebank)

        codebank_keys = [x for x in self._codebank.keys()]
        for fn in codebank_keys: 
            if fn not in to_update and fn not in funcs_called_in_passing_tests and fn not in funcs_called_in_codebank:
                logger.info(f"{fn} is now obsolete, removing...")
                del self._codebank[fn]
                self.collection.delete(ids=[fn])
                self.removed.append(fn)

    def add_comments(self):

        def parse_output(output):
            output = re.split("DOCSTRING:", output)[1].strip()
            return f"    # {output}"

        def comment_code(code_block):
            # split codeblock into lines 
            # output each line in comment format, i.e. 
            # \t# >>> <line> 
            lines = code_block.split("\n")
            commented_lines = [f"    # >>> {line}" for line in lines]
            return "\n".join(commented_lines)

        def remove_imports(code_block):
            code_block = re.sub(r"from .* import .*", "", code_block, flags=re.MULTILINE)
            code_block = re.sub(r"import .*", "", code_block, flags=re.MULTILINE)
            return code_block.strip()

        for func in self._codebank.keys(): 
            test_cases = self._codebank[func].test_cases
            passing_cases = [tc for i, tc in enumerate(test_cases) if self._codebank[func].was_success[i]]
            comment_prompt = self.comment_prompt.format(function = self._codebank[func]._original_code) 
            comment_output = self.model(comment_prompt, agent=False)
            comment_output = parse_output(comment_output)
            tc = np.random.choice(passing_cases, 1)[0]
            tc_query = tc.pred_node.query
            tc_code = remove_imports(tc.pred_node.program)
            tc_query = f"    # Query: {tc_query.strip()}" 
            tc_code = comment_code(tc_code)
            comment = f"{comment_output}\n    # Example:\n{tc_query}\n{tc_code}"

            code = self._codebank[func]._original_code
            code = code.split("\n")
            code = [code[0]] + [comment] + code[1:]
            code = "\n".join(code)
            self._codebank[func]._original_code = code


    
 








