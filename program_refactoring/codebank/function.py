import re 
import ast 
import numpy as np
import pdb 

from program_refactoring.domains.logos.utils import clean_import
from dataflow.core.lispress import (parse_lispress,
                                    render_compact,
                                    render_pretty)

class Function:
    def __init__(self, name, args, content, description, original_code, round_added):
        self._name = name
        self._args = args
        self._content = content
        self._description = description
        self._original_code = original_code
        self.round_added = round_added

        self.was_success = []
        self.num_programs_used = []
        # query: test case
        self.test_cases = []

    @staticmethod
    def parse_name_args(line):
        line = line.strip()
        # line is like: def function_name(arg1, arg2, arg3):
        name_gex = re.compile(r"def\s+(\w+)\s*\((.*)\):")
        match = name_gex.match(line)
        if match is None:
            raise ValueError(f"Could not parse function name from {line}")
        name = match.group(1)
        args = match.group(2)
        args = args.split(",")
        args = [x.strip() for x in args]
        return name, args  


    @classmethod
    def from_str(cls, function_str, round_added = None):
        """Create a function from a string"""
        function_str = function_str.strip()

        try:
            parsed = ast.parse(function_str)
        except SyntaxError:
            raise SyntaxError(f"Could not parse function from {function_str} because of SyntaxError")

        # don't add non-functions to the codebank 
        if len(parsed.body) > 1:
            raise ValueError(f"Function class cannot parse multiple functions at once")
        function = parsed.body[0]

        # get name, args, and content from function
        name = function.name 
        args = [arg.arg for arg in function.args.args]
        content = ast.unparse(function.body)

        # get description (comments stripped by ast so we need these separately)
        split_str = [x.strip() for x in function_str.split("\n")]
        # parse out the docstring description 
        desc_lines = []
        i = 1
        if split_str[1].startswith("#"): 
            # comment-style
            for j, line in enumerate(split_str[1:]): 
                if line.startswith("#"): 
                    desc_lines.append(line) 
                    i += 1
                else:
                    # break on first non-match 
                    break
        elif split_str[1].startswith('"""') or split_str[1].startswith("''''"): 
            # string-style, continue until we find the end of the string
            
            desc_lines.append(split_str[1])
            line = split_str[1]
            while ('"""') not in line and ("''''") not in line: 
                desc_lines.append(line)
                i += 1
                line = split_str[i]

        # what if there's no comment 
        if len(desc_lines) == 0: 
            
            # use the name 
            description = name 
        else:
            description = "\n".join(desc_lines)

        return cls(name, args, content, description, function_str, round_added)
    
    def as_str(self):
        """Return the function as a string"""
        return self._original_code

    def summarize(self, include_ex = False, include_success = True):
        successes = np.sum(self.was_success)
        total = len(self.was_success)
        success_string = f"    # Success rate: {successes}/{total}" 

        ex_str = ""
        if include_ex:
            # get shortest successful test case 
            successful_tcs = [x for i, x in enumerate(self.test_cases) if self.was_success[i]]
            if len(successful_tcs) > 0:
                successful_tcs = sorted(successful_tcs, key=lambda x: len(x.pred_node.program.split("\n")))
                ex_query = successful_tcs[0].pred_node.query
                ex_prog = clean_import(successful_tcs[0].pred_node.program)
                # insert >>> at the start of each line
                ex_prog = "\n".join([f"    >>> {x.strip()}" for x in ex_prog.split("\n")])
                ex_str = f"    \"\"\"Example: {ex_query}\n{ex_prog}\n    \"\"\""

        code_string = self._original_code
        splitstring = code_string.split("\n")
        def_str, rest = splitstring[0], splitstring[1:]
        rest = "\n".join(rest)
        # rest = f"\t{rest}"
        if include_ex and include_success:
            return f"{def_str}\n{success_string}\n{ex_str}\n{rest}\n"
        elif include_ex and not include_success:
            return f"{def_str}\n{ex_str}\n{rest}\n"
        elif not include_ex and include_success:
            return f"{def_str}\n{success_string}\n{rest}\n"
        else:
            return f"{def_str}\n{rest}\n"
    
    def compute_success(self):
        """"Compute how successful a function is, based on how often programs that use it succeed. 
        Success measured on [-1, 1].
        Normalized by the number of other functions in the program to normalize blame.
        Return:
        - success_coef: int in [-1, 1] representing how successful the function is"""

        tot = 0 
        for was_s, num_other in zip(self.was_success, self.num_programs_used):
            blame = 1 / num_other 
            if was_s: 
                # if success, gets full credit 
                tot += 1 
            else:
                # if failure, only gets part of the blame 
                tot += -1 * blame
        try:
            return tot / len(self.was_success), len(self.was_success)
        except ZeroDivisionError:
            return -1, 0 
    


class LispFunction(Function):
    def __init__(self, name, args, content, description, original_code, round_added):
        super().__init__(name, args, content, description, original_code, round_added)

    @staticmethod
    def parse_name_args(function):
        raise NotImplementedError
        return name, args  


    @classmethod
    def from_str(cls, function_str, round_added=None):
        """Create a function from a string"""
        function_str = function_str.strip()

        try:
            parsed = parse_lispress(function_str)
        except:
            raise SyntaxError(f"Could not parse function from {function_str} because of SyntaxError")

        # get name, args, and content from function
        name = parsed[1]
        args = parsed[2]
        content = parsed[3:]

        description = name 

        return cls(name, args, content, description, function_str, round_added)
    
    def as_str(self):
        """Return the function as a string"""
        return self._original_code
    
    def summarize(self, include_ex = False, include_success = True):
        return self.as_str()