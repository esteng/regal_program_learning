import pdb 
import re 
import ast 
from collections import defaultdict
from copy import deepcopy 
import logging 
from pathlib import Path 
from typing import List, Dict

import numpy as np
np.random.seed(12)

from program_refactoring.headers import TEXTCRAFT_HEADER
from program_refactoring.tree.node import Node, TextCraftNode
from program_refactoring.tree.tuple import Tuple
from program_refactoring.domains.textcraft.utils import get_func_names
from program_refactoring.codebank.codebank import CodeBank
from program_refactoring.codebank.test_case import TestCase, TextCraftTestCase

from program_refactoring.model.prompts import textcraft_tuple_refactor_prompt


logger = logging.getLogger(__name__)

class TextCraftTuple(Tuple):

    def __init__(self, 
                 nodes: Dict[int, Node],
                 task: str = "textcraft",
                 temp_dir: str = "temp"):
        super().__init__(nodes, "textcraft") 
        self.temp_dir = temp_dir 

    def import_codebank(self, program):
        program = f"""from {self.temp_dir}.codebank import *\n\n{program}"""
        return program 


    def retry_merge(self, 
                    codebank_str, 
                    model, 
                    helper_functions,
                    nodes_before, 
                    nodes_after, 
                    nodes_succeed,
                    functions_used,
                    results_before_by_idx,
                    results_after_by_idx,
                    helpers_first=True): 
        """Retry the merge with a new prompt"""

        all_funcs_used = [x for y in functions_used.values() for x in y]
        if len(all_funcs_used) > 0:
            functions_used = set(all_funcs_used)
            functions_used = ", ".join(functions_used)
            functions_used_str = f"Pay special attention to the following functions: embed, {functions_used} and refactor them if needed."
        else:
            functions_used_str = ""

        # TODO (elias): Expected here: 
        # iterate through self.nodes and determine if they failed
        # only show the failed ones 

        prompt_internal_str = ""
        prompt_final_str = ""
        for idx, node_before in nodes_before.items():
            try:
                node_after = nodes_after[idx]
            except KeyError:
                logger.info(f"ERROR: node {idx} not in nodes_after")
                continue
            node_success = nodes_succeed[idx]
            if not node_success:
                _, err = node_after.execute(verbose=True)
                err = '\n'.join(err.split('\n')[:-1])
                prompt_internal_str += f"Query {idx}: {node_before.query}\n" +\
                f"PROGRAM {idx} (before):\n{node_before.program}\n" + \
                f"PROGRAM {idx} (after):\n{node_after.program}\n" + \
                f"EXECUTION TRACE {idx} (after):\n{err}\n" + \
                f"SUCCEEDED: {node_success}\n\n"

                prompt_final_str += f"NEW PROGRAM {idx}: <code for program {idx}>\n" 

        if len(prompt_internal_str) > 0:
            if helpers_first:
                prompt = f"""ERROR: the following rewritten programs failed to execute to the same result.
Helper functions:\n{helper_functions}
{prompt_internal_str}

Please rewrite the programs so that the are equivalent to the programs before the merge. Do not simply reproduce program 1 -- try to use helper functions and fix them if you have to.
{functions_used_str}
Do not include anything in your output that is not valid Python code. Your programs MUST be formatted in the following fashion: 
NEW HELPERS: <code for helper functions or refactored functions>
{prompt_final_str}
"""
            else:
                prompt = f"""ERROR: the following rewritten programs failed to execute to the same result.
Helper functions:\n{helper_functions}
{prompt_internal_str}

Please rewrite the programs so that the are equivalent to the programs before the merge. Do not simply reproduce program 1 -- try to use helper functions and fix them if you have to.
{functions_used_str}
Do not include anything in your output that is not valid Python code. Your programs MUST be formatted in the following fashion: 
{prompt_final_str}
NEW HELPERS: <code for helper functions or refactored functions>
"""
        else:
            return None, {idx: None for idx in nodes_before.keys()}
        logger.info(f"Running retry with prompt: {prompt}")

        result_text = model(prompt, agent=False)

        logger.info(f"Response:\n{result_text}")
        try:
            new_helpers, new_programs  = TextCraftTuple.parse_result(result_text)
            return new_helpers, new_programs
        except ValueError:
            logger.info(f"ERROR: could not parse result text: {result_text}")
            return None, {idx: None for idx in nodes_before.keys()}

    def merge(self, codebank, model, done = [], do_retry = True, round_added: int = None, helpers_first: bool = True, craft_retrieve=False):
        """Merge nodes using a merge prompt"""
        logger.info("====================================\n\n")
        logging_filename = logger.manager.root.handlers[0].baseFilename
        logging_path = Path(logging_filename.split(".log")[0])  
        logging_path.mkdir(parents=True, exist_ok=True)  

        # get original results 
        node_names = {i: re.sub(" ","_", node.name) for i, node in self.nodes.items()} 

        results_before = {}
        for i, node in self.nodes.items():
            results_before[i] = node.execute(logging_path / f"{node.name}_before.py")
        # get relevant info from the codebank 
        codebank_ids = []
        for idx, node in self.nodes.items():
            # no more than 20 total examples 
            if not craft_retrieve:
                codebank_ids += codebank.get_relevant(node.query, k = 20 // len(self.nodes))
            else:
                query = node.metadata + 'Goal: ' + node.query
                codebank_ids += codebank.get_relevant(query, k = 20 // len(self.nodes))
        codebank_ids = set(codebank_ids)
        codebank_funcs = [codebank.get(id) for id in codebank_ids]

        # format codebank, adding in the body of the functions 
        codebank_str = [func.summarize() for func in codebank_funcs if func is not None]
        codebank_str = "\n".join(codebank_str)

        if len(codebank_str) > 0:
            codebank_instr = f"\nYou can also choose from the following helper functions:\n{codebank_str}"
        else:
            codebank_instr = ""


        queries_and_code = []
        if helpers_first:
            answer_format_short = ["NEW HELPERS:\n"]
            answer_format_long = ["NEW HELPERS:\n# Thoughts:\n# 1. The following functions are shared by multiple programs: <function names>\n<code for helper functions>\n"]
            for i, node in self.nodes.items():
                # skip things already done 
                if node.node_id in done:
                    continue
                queries_and_code.append(f"QUERY {i}: {node.query}\n PROGRAM {i}:\n{node.program}")
                answer_format_short.append(f"NEW PROGRAM {i}:")
                answer_format_long.append(f"NEW PROGRAM {i}: <code for program {i}>")
        else:
            answer_format_short = []
            answer_format_long = []
            for i, node in self.nodes.items():
                # skip things already done 
                if node.node_id in done:
                    continue
                queries_and_code.append(f"QUERY {i}: {node.query}\n PROGRAM {i}:\n{node.program}")
                answer_format_short.append(f"NEW PROGRAM {i}:")
                answer_format_long.append(f"NEW PROGRAM {i}: <code for program {i}>")
            answer_format_short.append("NEW HELPERS:\n") 
            answer_format_long.append("NEW HELPERS:\n# Thoughts:\n# 1. The following functions are shared by multiple programs: <function names>\n<code for helper functions>\n") 


        queries_and_code = "\n".join(queries_and_code)
        answer_format_short = "\n".join(answer_format_short)
        answer_format_long = "\n".join(answer_format_long)

        # create the merge prompt
        prompt = textcraft_tuple_refactor_prompt.format(codebank_instr=codebank_instr,
                                            queries_and_code = queries_and_code,
                                            answer_format_short = answer_format_short,
                                            answer_format_long = answer_format_long) 
        
        logger.info(f"Running {prompt}")
        logger.info(f"Using MODEL: {model.model_name}")
        logger.info(f"program names: {[node.name for i, node in self.nodes.items()]}")
        result_text = model(prompt, agent=False)        
        logger.info(f"Raw result from {model.model_name}:\n{result_text}") 

        # parse into helpers and programs 
        try:
            new_helpers, new_programs = TextCraftTuple.parse_result(result_text)
        except ValueError:
            logger.info(f"ERROR: could not parse result text: {result_text}")
            return {i: False for i, _ in self.nodes.items()}, codebank
        (new_codebank,
         success_by_idx,
         new_nodes_by_idx,
         funcs_called_by_idx,
         results_before_by_idx,
         results_after_by_idx) = self.test_programs(
                                          codebank, 
                                          new_helpers, 
                                          new_programs,
                                          node_names,
                                          results_before,
                                          logging_path)

        if do_retry: 
            retry_helpers, retry_programs = self.retry_merge(codebank_str,
                                                            model,
                                                            new_helpers,
                                                            self.nodes,
                                                            new_nodes_by_idx,
                                                            success_by_idx,
                                                            funcs_called_by_idx,
                                                            results_before_by_idx,
                                                            results_after_by_idx)
            (new_codebank,
             retry_success_by_idx,
             retry_nodes_by_idx,
             retry_funcs_called_by_idx,
             results_before_by_idx,
             results_after_by_idx) = self.test_programs(
                                                codebank,
                                                retry_helpers,
                                                retry_programs,
                                                node_names,
                                                results_before,
                                                logging_path)

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
                __ = codebank.add_multiple(final_helpers, round_added)

        # get function names from left and right program 
        # rules here: 
        # 1. if function already exists, and it is used in a failed program, assign a failure
        # 2. if function doesn't already exist and is used in a failed program, don't add it or assign failure
        # 3. if function doesn't exist and program succeeded, then it will have been added above, assign success
        for idx, funcs in funcs_called_by_idx.items():
            # add success info
            new_nodes_by_idx[idx].is_success = success_by_idx[idx]
            for f in funcs: 
                # if function already exists, assign a failure 
                if f in codebank._codebank.keys():
                    codebank._codebank[f].was_success.append(success_by_idx[idx])
                    codebank._codebank[f].num_programs_used.append(len(funcs_called_by_idx[idx]))
                    # add test case 
                    left_tc = TextCraftTestCase(new_nodes_by_idx[idx], 
                                           self.original_nodes[idx], 
                                           model,
                                           is_correct=success_by_idx[idx])
                    codebank._codebank[f].test_cases.append(left_tc)

        return success_by_idx, codebank

    def test_programs(self, 
                    codebank: CodeBank, 
                    new_helpers: List[str], 
                    new_programs: Dict[int, str],
                    node_names: Dict[int, str], 
                    results_before: Dict[int, np.array],
                    logging_path: Path):
        # add helper functiosn to the codebank and import the codebank 
        new_codebank = CodeBank.clone(codebank) 

        # skip making the collection so as to not confuse the retriever 
        try:
            if type(new_helpers) == list:
                new_helpers = "\n".join(new_helpers)
            func_names = new_codebank.add_multiple(new_helpers) 
        except TypeError:
            # no new helpers, new_helpers = None
            pass
        new_codebank.write_to_file()


        success_by_idx = {}
        new_nodes_by_idx = {}
        funcs_called_by_idx = {}
        results_after_by_idx = {}
        for idx, prog in new_programs.items():
            new_program = self.import_codebank(prog)

            # make new node objects 
            prog_dict = {k:v for k,v in self.nodes[idx].__dict__.items() if k in TextCraftNode.__init__.__code__.co_varnames}
            prog_dict['type'] = "pred"
            prog_dict['program'] = new_program
            node_copy = self.nodes[idx].__class__(**prog_dict)
            # check that the new programs execute to the same result 
            try:
                result_after = node_copy.execute(logging_path / f"{node_names[idx]}_after.py")
            except Exception as e:
                logger.info(f"ERROR: left program failed to execute")
                logger.info(f"traceback: {e}")
                result_after = None

            results_after_by_idx[idx] = result_after
            success = False
            if result_after is not None:
                try:
                    assert(result_after)
                    success = True
                except AssertionError:
                    logger.info(f"ERROR: program {idx} does not execute to the same result")

            success_by_idx[idx] = success
            new_nodes_by_idx[idx] = node_copy
            funcs_called = get_func_names(node_copy.program)
            funcs_called_by_idx[idx] = funcs_called

        return (new_codebank, 
                success_by_idx,    
                new_nodes_by_idx, 
                funcs_called_by_idx, 
                results_before, 
                results_after_by_idx)


    @staticmethod 
    def parse_result(result_text):

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
            helper_gex = re.compile(r"NEW HELPERS:\s+(.*?)NEW PROGRAM", flags=re.DOTALL)
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
            helpers = helper_gex.search(result_text).group(1)
            helpers = clean(helpers)

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