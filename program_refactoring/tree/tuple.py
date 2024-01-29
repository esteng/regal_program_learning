import logging 
import re 
import ast 
from collections import defaultdict
from typing import List, Dict

from program_refactoring.tree.node import Node
from program_refactoring.model.prompts import python_self_consistency_prompt

logger = logging.getLogger(__name__)

class Tuple:
    def __init__(self, 
                 nodes: Dict[int, Node],
                 task: str = "logos"
    ):
        self.nodes = nodes 
        self.task = task
        self.original_nodes = {k: v.__class__.from_json(v.to_json()) for k, v in nodes.items()}

    
    def merge(self, codebank, model):
        raise NotImplementedError


    def run_self_consistency(self, model, prompt, n_variants=3):
        def parse_option_result(result_text):
            try:
                index = re.search("BEST OPTION: (\d+)", result_text).group(1)
                index = int(index)
            except (ValueError, AttributeError) as e:
                return 0
            return index
        helpers_by_variant = {}
        programs_variants_by_idx = defaultdict(list)

        all_new_helpers = [] 
        new_programs_by_idx = {}

        # get variants from model
        for i in range(n_variants):
            result_text = model(prompt, agent=False)
            new_helpers, new_programs = self.parse_result(result_text)
            helpers_by_variant[i] = new_helpers
            for idx, prog in new_programs.items():
                programs_variants_by_idx[idx].append(prog)
            helpers_by_variant[i] = new_helpers

        # vote on variants 
        votes = defaultdict(int)
        for i in range(n_variants):
            for idx in programs_variants_by_idx.keys():
                options = programs_variants_by_idx[idx]
                program_str = []
                for i, option in enumerate(options):
                    # helpers = "\n".join(helpers_by_variant[i]) 
                    helpers = helpers_by_variant[i]
                    program_str.append(f"OPTION {i}:\n{helpers}\n{option}")

                program_str = "\n".join(program_str)
                prompt = python_self_consistency_prompt.format(query = self.nodes[idx].query,
                                                                orig_function = self.nodes[idx].program,
                                                                option_str = program_str)
                result = model(prompt, agent=False)

                best_idx = parse_option_result(result)
                votes[best_idx] += 1
        # get max
        best_idx = max(votes, key=votes.get)
        # best_idx = 
        all_new_helpers = helpers_by_variant[best_idx]
        new_programs_by_idx[idx] = options[best_idx]

        return all_new_helpers, new_programs_by_idx


    @staticmethod
    def parse_result(result_text):
        raise NotImplementedError
    
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
    




        