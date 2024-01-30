import pdb 
import re 
import ast 
import json 
from collections import defaultdict
from copy import deepcopy 
import logging 
from pathlib import Path 
from typing import List, Dict

import numpy as np
np.random.seed(12)

from program_refactoring.headers import LOGO_HEADER
from program_refactoring.tree.node import Node, PythonNode
from program_refactoring.tree.tuple import Tuple
from program_refactoring.codebank.codebank import CodeBank
from program_refactoring.codebank.test_case import TestCase, LogoTestCase, PythonTestCase

from program_refactoring.domains.python.utils import get_func_names   

from program_refactoring.model.prompts import python_tuple_refactor_prompt, python_self_consistency_prompt

import sys
sys.path.insert(0, "third_party/Faithful-COT/source")
from evaluate.evaluate_answer_acc import is_correct, extract_gold_answer, extract_pred_answer

logger = logging.getLogger(__name__)

class PythonTuple(Tuple):

    def __init__(self, 
                 nodes: List[Node],
                 use_removed = False,
                 task: str = "python",
                 temp_dir: str = "temp"):
        super().__init__(nodes, task) 

        self.use_removed = use_removed
        self.temp_dir = temp_dir 
        if self.task == "scan": 
            self.tuple_refactor_prompt = scan_tuple_refactor_prompt
        else:
            self.tuple_refactor_prompt = python_tuple_refactor_prompt

    def import_codebank(self, program):
        program = f"""from {self.temp_dir}.codebank import *\n\n{program}"""
        return program 

    def remove_import(self, x):
        return re.sub("from .*\.codebank import \*", "", x).strip()

    def retry_merge(self, 
                    codebank, 
                    model, 
                    helper_functions,
                    nodes_before, 
                    nodes_after, 
                    nodes_succeed,
                    functions_used,
                    results_before,
                    results_after): 
        """Retry the merge with a new prompt, including some feedback"""

        # get relevant info from the codebank 
        codebank_ids = []
        for idx, node in nodes_before.items():
            # no more than 20 total examples 
            codebank_ids += codebank.get_relevant(node.query, k = 20 // len(nodes_before))
        codebank_ids = set(codebank_ids)
        codebank_funcs = [codebank.get(id) for id in codebank_ids]

        # format codebank, adding in the body of the functions 
        codebank_str = [func.summarize() for func in codebank_funcs if func is not None]
        codebank_str = "\n".join(codebank_str)


        # all_funcs_used = [x for y in functions_used.values() for x in y]

        # NOTE (elias): Moving to retrying individual examples 
        new_helpers = ""
        new_programs = {idx: None for idx in nodes_after.keys()}
        for idx, node_before in nodes_before.items():
            try:
                node_after = nodes_after[idx]
            except KeyError:
                logger.info(f"ERROR: node {idx} not in nod")
                continue
            # pdb.set_trace()
            funcs_used = functions_used[idx]
            
            if len(funcs_used) > 0:
                funcs_used = set(funcs_used)
                funcs_used = ", ".join(funcs_used)
                functions_used_str = f"Pay special attention to the following functions: {funcs_used} and refactor them if needed."
            else:
                functions_used_str = ""

            prompt_internal_str = ""
            prompt_final_str = ""

            node_success = nodes_succeed[idx]
            if not node_success: 
                prompt_internal_str += f"Query {idx}: {node_before.query}\n" +\
                f"PROGRAM {idx} (before):\n{self.remove_import(node_before.program)}\n" + \
                f"RESULT {idx} (before):\n{results_before[idx].strip()}\n" + \
                f"PROGRAM {idx} (after):\n{self.remove_import(node_after.program)}\n" + \
                f"RESULT {idx} (after):\n{str(results_after[idx]).strip()}\n"

                prompt_final_str += f"NEW PROGRAM {idx}: # Thought: 1. The difference between the outputs is <difference>.\n# 2. the previous version failed because <reason>.\n<code for program {idx}>\n" 

            # pdb.set_trace()
            if len(prompt_internal_str) > 0:
                prompt = f"""ERROR: the following rewritten programs failed to execute to the same result.
Helper functions:\n{helper_functions}
{prompt_internal_str}

Please rewrite the programs so that the are equivalent to the programs before the merge. Do not simply reproduce program 1 -- try to use helper functions and fix them if you have to.
{functions_used_str}
Do not include anything in your output that is not valid Python code. Your programs MUST be formatted in the following fashion: 
{prompt_final_str}
"""
            else:
                continue
                # return None, {idx: None for idx in nodes_before.keys()}
            logger.info(f"Running retry with prompt: {prompt}")

            result_text = model(prompt, agent=False)
            logger.info(f"Response:\n{result_text}")
            try:
                local_new_helpers, new_programs  = self.parse_result(result_text)
                # sometimes model hallucinates additional programs
                new_program = list(new_programs.values())[0]
                # new_programs = {k:v for k, v in new_programs.items() if k in results_before.keys()}
                # return new_helpers, new_programs
                # if local_new_helpers is not None and len(local_new_helpers) > 0:
                #     pdb.set_trace()
                new_helpers += "\n"+local_new_helpers + "\n"
                new_programs[idx] = new_program
            except ValueError:
                logger.info(f"ERROR: could not parse result text: {result_text}")
                # return None, {idx: None for idx in nodes_before.keys()}
        if len(new_helpers) == 0:
            new_helpers = None
        # pdb.set_trace()

        return new_helpers, new_programs

    # def run_self_consistency(self, model, prompt, n_variants=3):
    #     def parse_option_result(result_text):
    #         try:
    #             index = re.search("BEST OPTION: (\d+)", result_text).group(1)
    #             index = int(index)
    #         except (ValueError, AttributeError) as e:
    #             return 0
    #         return index
    #     helpers_by_variant = {}
    #     programs_variants_by_idx = defaultdict(list)

    #     all_new_helpers = [] 
    #     new_programs_by_idx = {}

    #     # get variants from model
    #     for i in range(n_variants):
    #         result_text = model(prompt, agent=False)
    #         new_helpers, new_programs = self.parse_result(result_text)
    #         helpers_by_variant[i] = new_helpers
    #         for idx, prog in new_programs.items():
    #             programs_variants_by_idx[idx].append(prog)
    #         helpers_by_variant[i] = new_helpers

    #     # vote on variants 
    #     for idx in programs_variants_by_idx.keys():
    #         options = programs_variants_by_idx[idx]
    #         program_str = []
    #         for i, option in enumerate(options):
    #             # helpers = "\n".join(helpers_by_variant[i]) 
    #             helpers = helpers_by_variant[i]
    #             program_str.append(f"OPTION {i}:\n{helpers}\n{option}")

    #         program_str = "\n".join(program_str)
    #         prompt = python_self_consistency_prompt.format(query = self.nodes[idx].query,
    #                                                         orig_function = self.nodes[idx].program,
    #                                                         option_str = program_str)
    #         result = model(prompt, agent=False)

    #         best_idx = parse_option_result(result)
    #         all_new_helpers = helpers_by_variant[best_idx]
    #         new_programs_by_idx[idx] = options[best_idx]

    #     return all_new_helpers, new_programs_by_idx


    def merge(self, codebank, model, done = [], do_retry = True, round_added: int = None, 
              helpers_first: bool = False, use_self_consistency: bool = False,
              self_consistency_width: int = 3):
        """Merge nodes using a merge prompt"""
        logger.info("====================================\n\n")
        logging_filename = logger.manager.root.handlers[0].baseFilename
        logging_path = Path(logging_filename.split(".log")[0])  
        logging_path.mkdir(parents=True, exist_ok=True)  

        # get original results 
        node_names = {i: re.sub(" ","_", node.name) for i, node in self.nodes.items()} 

        results_before = {}
        for i, node in self.nodes.items():
            try:
                results_before[i] = node.execute(logging_path / f"{node.name}_before.py")
            except json.decoder.JSONDecodeError:
                pdb.set_trace()

        # get relevant info from the codebank 
        codebank_ids = []
        for idx, node in self.nodes.items():
            # no more than 20 total examples 
            codebank_ids += codebank.get_relevant(node.query, k = 20 // len(self.nodes))
        codebank_ids = set(codebank_ids)
        codebank_funcs = [codebank.get(id) for id in codebank_ids]

        # format codebank, adding in the body of the functions 
        codebank_str = [func.summarize() for func in codebank_funcs if func is not None]
        codebank_str = "\n".join(codebank_str)

        if len(codebank_str) > 0:
            codebank_instr = f"\nYou can also choose from the following helper functions:\n{codebank_str}"
        else:
            codebank_instr = ""


        # NOTE (elias): removed part about removal for now since it didn't help for pairs 
        queries_and_code = []
        queries_only = []
        if self.task == "scan":
            # answer_format_short = ["HELPER SCRATCHPAD:\nNEW HELPERS:\n"]
            # answer_format_long = ["HELPER SCRATCHPAD:\n# Reproduce existing helper functions that you can use here\nNEW HELPERS:\n# Thoughts:\n# 1. The following functions are shared by multiple programs: <function names>\n<code for helper functions>\n"]
            answer_format_short = ["NEW HELPERS (NEVER REDEFINE jump, run, walk, look, turn_left, turn_right! NEVER DEFINE perform_actions):\n"]
            answer_format_long = ["NEW HELPERS (NEVER REDEFINE jump, run, walk, look, turn_left, turn_right! NEVER DEFINE perform_actions):\n# Thoughts:\n# 1. The following functions are shared by multiple programs: <function names>\n<code for helper functions>\n"]
        else:
            answer_format_short = []
            answer_format_long = []
        for i, node in self.nodes.items():
            # skip things already done 
            if node.node_id in done:
                continue
            queries_and_code.append(f"QUERY {i}: {node.query}\nPROGRAM {i}:\n{self.remove_import(node.program)}")
            queries_only.append(f"QUERY {i}: {node.query}")
            answer_format_short.append(f"NEW PROGRAM {i}:")
            # answer_format_long.append(f"NEW PROGRAM {i}:\n# Thoughts:\n# 1. The query asks for: <query intention>\n 2. <query> can be solved by <components>.\n# 3. I will use helper function <function> to <goal>.\n<code for program {i}>\n")
            answer_format_long.append(f"NEW PROGRAM {i}:\n# Thoughts:\n# 1. The query asks for: <query intention>\n 2. <query> can be solved by <components>.\n# 3. I will use/define helper function <function> to <goal>.\n<code for program {i}>\n")


        queries_and_code = "\n".join(queries_and_code)
        answer_format_short = "\n".join(answer_format_short)
        answer_format_long = "\n".join(answer_format_long)

        # create the merge prompt
        prompt = self.tuple_refactor_prompt.format(codebank_instr=codebank_instr,
                                            queries_and_code = queries_and_code,
                                            queries_only = queries_only,
                                            answer_format_short = answer_format_short,
                                            answer_format_long = answer_format_long) 
        
        logger.info(f"Running {prompt}")
        logger.info(f"Using MODEL: {model.model_name}")
        logger.info(f"program names: {[node.name for i, node in self.nodes.items()]}")

        if use_self_consistency:
            logger.info(f"Running self-consistency with n = 3...")
            new_helpers, new_programs = self.run_self_consistency(model, prompt, n_variants=self_consistency_width)
        else:
            result_text = model(prompt, agent=False)
            logger.info(f"Raw result from {model.model_name}:\n{result_text}") 
            # parse into helpers and programs 
            new_helpers, new_programs = self.parse_result(result_text)
        # try:
        #     new_helpers, new_programs = PythonTuple.parse_result(result_text)
        # except ValueError:
        #     logger.info(f"ERROR: could not parse result text: {result_text}")
        #     return {i: False for i, _ in self.nodes.items()}, codebank


        (new_codebank,
         success_by_idx,
         new_nodes_by_idx,
         funcs_called_by_idx,
         results_after) = self.test_programs(
                                          codebank, 
                                          new_helpers, 
                                          new_programs,
                                          node_names,
                                          results_before,
                                          logging_path)
        logger.info(f"success_by_idx coming out of test: {success_by_idx}")
        # pdb.set_trace()
        if do_retry: 
            logger.info(f"doing retry")
            retry_helpers, retry_programs = self.retry_merge(codebank,
                                                            model,
                                                            new_helpers,
                                                            self.nodes,
                                                            new_nodes_by_idx,
                                                            success_by_idx,
                                                            funcs_called_by_idx,
                                                            results_before,
                                                            results_after)
            # if retry_helpers is not None:
            #     pdb.set_trace()
            (new_codebank,
             retry_success_by_idx,
             retry_nodes_by_idx,
             retry_funcs_called_by_idx,
             retry_results_after) = self.test_programs(
                                                codebank,
                                                retry_helpers,
                                                retry_programs,
                                                node_names,
                                                results_before,
                                                logging_path)
            logger.info(f"retry success_by_idx: {retry_success_by_idx}")
            for idx, success in retry_success_by_idx.items():
                if success and not success_by_idx[idx]:
                    logger.info(f"Retry succeeded for program {idx}, overwriting...")
                    new_programs[idx] = retry_programs[idx]
                    success_by_idx[idx] = success
                    new_nodes_by_idx[idx] = retry_nodes_by_idx[idx]
                    new_nodes_by_idx[idx].is_success = True
                    funcs_called_by_idx[idx] = retry_funcs_called_by_idx[idx]
        else:
            retry_helpers = None
            retry_programs = None

        for idx, new_program in new_programs.items():
            if success_by_idx[idx]:
                logger.info(f"Merge succeeded! New programs:\n{new_program}") 
                logger.info(f"success_by_idx: {success_by_idx}")
                logger.info("Adding new codebank to collection")

                helpers_split, __ = self.split_helpers(new_helpers)
                if retry_helpers is not None and idx in retry_success_by_idx.keys() and retry_success_by_idx[idx]:
                    logger.info(f"Retry succeeded for program {idx} -- adding helper functions")
                    split_retry_helpers, __ = self.split_helpers(retry_helpers)
                    helpers_split += split_retry_helpers 
                
                final_helpers = []
                for helper in helpers_split:
                    # get function name 
                    name = re.search("(?<=def )([\w_\d]+)\(", helper).group(1)
                    if name in funcs_called_by_idx[idx]:
                        final_helpers.append(helper)
                final_helpers = "\n".join(final_helpers)
                # if the merge succeeded we want to add the helper functions to the collection for retrieval 
                # just re-add all the helpers to the collection, which will trigger adding to the collection
                __ = codebank.add_multiple(final_helpers)
            
        for idx in new_nodes_by_idx.keys():
            # update node with the refactored version
            new_nodes_by_idx[idx].is_success = success_by_idx[idx]
            if success_by_idx[idx]:
                self.nodes[idx] = new_nodes_by_idx[idx]

        # get function names from all programs
        # rules here: 
        # 1. if function already exists, and it is used in a failed program, assign a failure
        # 2. if function doesn't already exist and is used in a failed program, don't add it or assign failure
        # 3. if function doesn't exist and program succeeded, then it will have been added above, assign success
        for idx, funcs in funcs_called_by_idx.items():
            # add success info

            for f in funcs: 
                # if function already exists, assign a failure 
                if f in codebank._codebank.keys():
                    codebank._codebank[f].was_success.append(success_by_idx[idx])
                    codebank._codebank[f].num_programs_used.append(len(funcs_called_by_idx[idx]))
                    # add test case 
                    left_tc = PythonTestCase(new_nodes_by_idx[idx], 
                                             self.original_nodes[idx], 
                                             model,
                                             is_correct = success_by_idx[idx])
                    codebank._codebank[f].test_cases.append(left_tc)

        return success_by_idx, codebank
    
    def get_imports(self, program):
        parsed = ast.parse(program) 
        imports = [node for node in parsed.body if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom)]
        return ast.unparse(imports)

    def test_programs(self, 
                    codebank: CodeBank, 
                    new_helpers: List[str], 
                    new_programs: Dict[int, str],
                    node_names: Dict[int, str], 
                    results_before: Dict[int, np.array],
                    logging_path: Path,
                    round_added: int = None):
        # add helper functiosn to the codebank and import the codebank 
        new_codebank = CodeBank.clone(codebank) 

        # skip making the collection so as to not confuse the retriever 
        try:
            new_imports = ""
            for k, new_program in new_programs.items():
                new_imports += "\n"+ self.get_imports(new_program)
            if type(new_helpers) == list:
                new_helpers = "\n".join(new_helpers)
            new_helpers = new_imports + "\n" + new_helpers
            func_names = new_codebank.add_multiple(new_helpers) 
        except TypeError:
            # no new helpers, new_helpers = None
            pass

        new_codebank.write_to_file()


        success_by_idx = {}
        new_nodes_by_idx = {}
        funcs_called_by_idx = {}
        results_after = {}
        for idx, prog in new_programs.items():
            new_program = self.import_codebank(prog)

            # only occurs at k=1 ablation 
            if idx not in self.nodes.keys():
                continue
            # make new node objects 
            prog_dict = {k:v for k,v in self.nodes[idx].__dict__.items() if k in PythonNode.__init__.__code__.co_varnames}
            prog_dict['type'] = "pred"
            prog_dict['program'] = new_program
            node_copy = self.nodes[idx].__class__(**prog_dict)

            # check that the new programs execute to the same result 
            try:
                result_after = node_copy.execute(logging_path / f"{node_names[idx]}_after.py")
                results_after[idx] = result_after
            except Exception as e:
                logger.info(f"ERROR: left program failed to execute")
                logger.info(f"traceback: {e}")
                result_after = None
                results_after[idx] = e

            success = False
            if result_after is not None:
                try:
                    if self.task == "scan":
                        answer_before = extract_scan_answer(results_before[idx])
                        answer_after = extract_scan_answer(result_after)
                        assert(answer_before == answer_after)
                    else:
                        answer_before = extract_pred_answer(self.task, results_before[idx])
                        answer_after = extract_pred_answer(self.task, result_after) 
                        assert(is_correct(self.task, answer_before, answer_after))
                    success = True
                except AssertionError:
                    logger.info(f"ERROR: program {idx} does not execute to the same result")
            success_by_idx[idx] = success
            new_nodes_by_idx[idx] = node_copy
            funcs_called = get_func_names(node_copy.program)
            funcs_called_by_idx[idx] = funcs_called

        return new_codebank, success_by_idx, new_nodes_by_idx, funcs_called_by_idx, results_after
    
    def parse_result(self, result_text):

        def clean(prog):
            prog = re.sub("```", "", prog)
            prog = re.sub("^[pP]ython", "", prog)
            prog = re.sub("^[pP]ython$", "", prog, flags=re.MULTILINE)
            prog = prog.strip() 
            return prog

        # add trailing program for regex to work 
        result_text += "\nNEW PROGRAM"
        has_helpers = False 

        if "NEW HELPERS" in result_text: 
            helper_gex = re.compile(r"NEW HELPERS\s+(.*?)NEW PROGRAM", flags=re.DOTALL)
            has_helpers = True
        search_gex = re.compile(r"NEW PROGRAM (\d+):(.*?)(?=((NEW PROGRAM)|(^\s+$)))", flags=re.DOTALL|re.MULTILINE)
        programs_and_idxs = search_gex.findall(result_text)

        if len(programs_and_idxs) == 0:
            search_gex = re.compile(r"Program (\d+):(.*?)(?=((Program)|(^\s+$)))", flags=re.DOTALL|re.MULTILINE)
            programs_and_idxs = search_gex.findall(result_text)
            if len(programs_and_idxs) == 0:
                raise ValueError(f"Could not parse result text: {result_text}")

        # helper_fxns = match.group(1)
        if has_helpers: 
            try:
                helpers = helper_gex.search(result_text).group(1)
                helpers = re.sub("\(NEVER REDEFINE.*?\):\n", "", helpers, flags=re.MULTILINE)
            except AttributeError:
                pdb.set_trace()

        clean_progs_and_idxs = []
        for prog_tup in programs_and_idxs:
            try:
                idx = int(prog_tup[0])
            except ValueError:
                logger.info(f"Skipping malflormed program: {prog_tup}")
                continue
            try:
                program = clean(prog_tup[1])
            except IndexError:
                logger.info(f"Skipping malflormed program: {prog_tup}")
                continue
            clean_progs_and_idxs.append((idx, program))


        pre_helpers = []
        if has_helpers:
            pre_helpers, __ = Tuple.split_helpers(helpers)

        all_phelpers = []
        calls = {}
        # separate out helper functions
        for i, prog in clean_progs_and_idxs:
            phelpers, pcalls = Tuple.split_helpers(prog)
            all_phelpers += phelpers
            calls[i] = "\n".join(pcalls)


        helpers = list(set(pre_helpers + all_phelpers))
        helpers = "\n".join(helpers)

        return helpers, calls
