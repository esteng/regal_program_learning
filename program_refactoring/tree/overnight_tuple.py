import pdb 
import re 
import ast 
from collections import defaultdict
import logging 
from pathlib import Path 
from typing import List, Dict

import numpy as np
np.random.seed(12)

from program_refactoring.tree.node import Node, OvernightNode
from program_refactoring.codebank.codebank import CodeBank
from program_refactoring.model.prompts import overnight_tuple_refactor_prompt
from program_refactoring.domains.python.utils import get_func_names   
from program_refactoring.tree.python_tuple import PythonTuple

logger = logging.getLogger(__name__)

class OvernightTuple(PythonTuple):

    def __init__(self, 
                 nodes: List[Node],
                 use_removed = False,
                 task: str = "overnight",
                 temp_dir: str = "temp"):
        super().__init__(nodes, use_removed, task, temp_dir) 

        self.tuple_refactor_prompt = overnight_tuple_refactor_prompt

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

            # make new node objects 
            prog_dict = {k:v for k,v in self.nodes[idx].__dict__.items() if k in OvernightNode.__init__.__code__.co_varnames}
            prog_dict['type'] = "pred"
            prog_dict['program'] = new_program
            try:
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
            except (ValueError, IndexError) as e: 
                result_after = None
                results_after[idx] = "Parsing error"
                node_copy = None

            success = False
            if result_after is not None and results_before[idx] == result_after:

                success = True
            else:
                logger.info(f"ERROR: program {idx} does not execute to the same result")
                logger.info(f"success_by_idx: {success_by_idx}")
            success_by_idx[idx] = success
            logger.info(f"set success at {idx} to {success}")

            if node_copy is not None:
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
            pre_helpers, __ = OvernightTuple.split_helpers(helpers)

        all_phelpers = []
        calls = {}
        # separate out helper functions
        for i, prog in clean_progs_and_idxs:
            phelpers, pcalls = OvernightTuple.split_helpers(prog)
            all_phelpers += phelpers
            calls[i] = "\n".join(pcalls)


        helpers = list(set(pre_helpers + all_phelpers))
        helpers = "\n".join(helpers)

        return helpers, calls

    @staticmethod
    def split_helpers(content):

        name_gex = re.compile("(?<=def )([\w_\d]+)\(")
        # pull out any initial comments for each function 
        prev_was_fxn = False
        prev_was_com = False
        comments = defaultdict(str) 
        curr_comment = []
        name=None 
        split_content = re.split("\n", content)
        for line in split_content:
            if re.match("^def .*$", line.strip()): 
                prev_was_fxn = True
                try:
                    name = name_gex.search(line.strip()).group(1)
                    if name == "answer": 
                        continue
                except AttributeError:
                    import pdb 
                    pdb.set_trace()
                    logger.info(f"ERROR: could not parse content:\n{content}")
                    return [], []
            else:
                if len(curr_comment) > 0:
                    comments[name] = "\n".join(curr_comment) 
                    curr_comment = []

                prev_was_fxn = False
            if prev_was_fxn or prev_was_com:
                if re.match("^#.*$", line.strip()): 
                    curr_comment.append(line)
                    prev_was_com = True
                prev_was_fxn = False


        # turn into ast 
        try:
            parsed = ast.parse(content)
        except SyntaxError:
            logger.info(f"ERROR: could not parse content:\n{content}")
            return [], []

        # split into functions and calls 
        def split_into_calls(parsed):
            fxns, calls = [], []
            for element in parsed:
                if isinstance(element, ast.FunctionDef): 
                    if element.name == "answer":
                        calls.append(element)
                    else:
                        fxns.append(element)
                else:
                    calls.append(element)
            return [ast.unparse(x) for x in fxns], [ast.unparse(x) for x in calls]

        functions, calls = split_into_calls(parsed.body)

        # add initial comments back into functions 
        def add_comments(function):
            split_fxn = function.split("\n")
            fxn_name = name_gex.search(split_fxn[0]).group(1)
            comment = comments[fxn_name]
            if comment is not None:
                new_fxn_content = [split_fxn[0]] + [comment] + split_fxn[1:]
            return "\n".join(new_fxn_content) 
        
        functions = [add_comments(fxn) for fxn in functions]


        return functions, calls