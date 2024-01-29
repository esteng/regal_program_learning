import re 
import ast 
import pdb 
import json 
import pickle as pkl
import subprocess
from pathlib import Path

from sexpdata import Symbol
from dataflow.core.lispress import (parse_lispress,
                                    render_compact,
                                    render_pretty)

from program_refactoring.domains.logos.visual_sim import load_img
from program_refactoring.domains.calflow.utils import Macro, replace_macro, split_macros
from program_refactoring.headers import LOGO_HEADER

class Node: 
    def __init__(self, 
                 query, 
                 program, 
                 type="gold",
                 name = None,
                 description = None,
                 metadata = None,
                 node_id = None,
                 temp_dir = None,
                 is_success = False,
                 is_done = False):
        self.query = query
        self.query_tok = re.split("\s+", query)
        self.program = program
        self.description = description
        self.metadata = metadata
        self.name = name 
        self.node_id = node_id
        self.is_done = is_done
        self.is_success = is_success
        self.type = type
        self.temp_dir = temp_dir

        self.is_leaf = (self.query is not None and 
                        self.program is not None and 
                        len(self.query) > 0 and 
                        len(self.program) > 0) 

    def to_json(self):
        toret = self.__dict__
        try:
            toret['temp_dir'] = str(toret['temp_dir'])
        except KeyError:
            pass
        return toret

    @classmethod 
    def from_json(cls, json):
        # filter json 
        jsond = {k:v for k, v in json.items() if k in cls.__init__.__code__.co_varnames}
        return cls(**jsond) 
    
    def is_returning(self, program):
        raise NotImplementedError

    def wrap_program(self, program):
        raise NotImplementedError
    
    def execute(self, additional_path):
        raise NotImplementedError
        

class PythonNode(Node):
    def __init__(self, 
                 query, 
                 program, 
                 type="gold",
                 name = None,
                 description = None,
                 metadata = None,
                 node_id = None,
                 temp_dir = None,
                 is_success = False,
                 is_done = False):
        super().__init__(query, program, type=type, name=name,  description=description, metadata=metadata, node_id=node_id, temp_dir=temp_dir, is_success = is_success, is_done = is_done)

        # check to see if it executes 
        if not self.is_returning(program.strip()):
            self.exec_program = self.wrap_program(program.strip())
        else:
            self.exec_program = program 

        if temp_dir is None:
            self.temp_dir = "temp"
        else:
            self.temp_dir = Path(temp_dir)

    def is_returning(self, program):
        # check if final part of program is an invocation 
        try:
            last_expr = ast.parse(program).body[-1]
        except (SyntaxError, IndexError) as e:
            return False
        if isinstance(last_expr, ast.Expr):
            return True
        return False


    def wrap_program(self, program):
        # remove all punc and replace space with underscore
        func_name = re.sub(r"\s+", "_", self.name)
        func_name = re.sub(r"[^\w\s]", "", func_name)

        # add a def to the start 
        header = f"def {func_name}():\n    _=0\n"
        # identify all code that isn't part of a function 
        # and add it to the function
        header_parsed = ast.parse(header) 

        parsed = ast.parse(program)
        helpers = []
        imports = [ast.parse("import json"), ast.parse("from program_refactoring.domains.scan.scan_helpers import *")]
        for node in parsed.body:
            if isinstance(node, ast.FunctionDef):
                helpers.append(node)
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                imports.append(node)
            else:
                header_parsed.body[0].body.append(node)
                
        temp_dir_safe = re.sub("\/+", ".", str(self.temp_dir)) 
        if Path(f"{self.temp_dir}/codebank.py").exists():
            # create an __init__.py file and write codebank
            with open(f"{self.temp_dir}/__init__.py", "w") as f1:
                f1.write("\n")
            import_stmt = f"from {temp_dir_safe}.codebank import *"
            imports.append(ast.parse(import_stmt))
        try:
            final_line = parsed.body[-1]
        except IndexError:
            return ""
        # if it's a print statement, take the thing before (what will be printed )
        if isinstance(final_line, ast.Expr) and hasattr(final_line.value, "func") and final_line.value.func.id == "print":
            assignee = ast.unparse(final_line.value.args[0])
        else:
            try:
                assignee = final_line.targets[0].id
            except AttributeError:
                return ""
            
        # instead of return we will write to file 
        return_stmt = ast.parse(f"with open('{self.temp_dir}/result_{self.type}.json', 'w') as f1:\n    json.dump({assignee}, f1)")
        header_parsed.body[0].body.append(return_stmt.body[0])
        # remove placeholder _=0
        header_parsed.body[0].body = header_parsed.body[0].body[1:]
        exec_stmt = ast.parse(f"{func_name}()")
        header_parsed.body.append(exec_stmt)

        # add the helper functions into the LOCAL scope 
        header_parsed.body[0].body = helpers  + header_parsed.body[0].body
        # add imports before everything
        header_parsed.body = imports + header_parsed.body

        new_program = ast.unparse(header_parsed)

        return new_program 

         

    def execute(self, additional_path=None):
        """Execute the program to obtain a result"""
        self.exec_program = self.wrap_program(self.program)
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(self.exec_program)
        try:
            with open(f"{self.temp_dir}/prog_{self.type}.py", "w") as f1:
                f1.write(self.exec_program)
            p = subprocess.Popen(["python", f"{self.temp_dir}/prog_{self.type}.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, errs = p.communicate()
            out, errs, = out.decode(), errs.decode()
            
        except Exception as e:
            print(f"Error executing program: {e}")
            raise e
        # read from file 
        try:
            with open(f"{self.temp_dir}/result_{self.type}.json") as f1:
                result = json.load(f1)
            # delete file
            Path(f"{self.temp_dir}/result_{self.type}.json").unlink()
        except FileNotFoundError:
            result = "error"
        return result
    

class LogoNode(Node):
    def __init__(self, 
                 query, 
                 program, 
                 type = "gold",
                 name = None,
                 description = None,
                 metadata = None,
                 node_id = None,
                 temp_dir=None,
                 is_success=False,
                 is_done=False):
        super().__init__(query=query, program=program, type=type, name=name, description=description, metadata=metadata, node_id=node_id, is_success=is_success, is_done=is_done)
        # check to see if it executes 
        # if not self.is_returning(program.strip()):
        #     self.exec_program = program
        # else:
        if temp_dir is None:
            self.temp_dir = "temp"
        else:
            self.temp_dir = Path(temp_dir)
        self.exec_program = self.wrap_program(program) 
        self.name = re.sub(" ", "_", self.name)


    def wrap_program(self, program): 
        header = LOGO_HEADER 
        temp_dir_safe = re.sub("\/+", ".", str(self.temp_dir)) 

        if Path(f"{self.temp_dir}/codebank.py").exists():
            # create an __init__.py file and write codebank
            with open(f"{self.temp_dir}/__init__.py", "w") as f1:
                f1.write("\n")
            program = f"{header}\nfrom {temp_dir_safe}.codebank import *\n\n{program}"
        else:
            program = f"{header}\n\n\n{program}"

        program = f"{program}\nturtle.save('{self.temp_dir}/result_{self.type}.jpg')"
        return program
         

    def execute(self, additional_path = None):
        """Execute the program to obtain a result"""
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(self.exec_program)
        try:
            with open(f"{self.temp_dir}/prog_{self.type}.py", "w") as f1:
                f1.write(self.exec_program)
            p = subprocess.Popen(["python", f"{self.temp_dir}/prog_{self.type}.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            out, errs = p.communicate()
            out, errs, = out.decode(), errs.decode()
            # print(out)
            # print(errs)
            result = load_img(f"{self.temp_dir}/result_{self.type}.jpg")
        except Exception as e:
            # pdb.set_trace()
            print(f"Error executing program: {e}")
            result = None
        # read from file 
        # fname = get_fname(self.exec_program)

        return result
    

    
class LispNode(Node):
    def __init__(self, 
                 query, 
                 program, 
                 type = "gold",
                 name = None,
                 description = None,
                 metadata = None,
                 node_id = None,
                 temp_dir=None,
                 is_success=False,
                 is_done=False):
        super().__init__(query=query, program=program, type=type, name=name, description=description, metadata=metadata, node_id=node_id, is_success=is_success, is_done=is_done)
        # check to see if it executes 
        # if not self.is_returning(program.strip()):
        #     self.exec_program = program
        # else:
        if temp_dir is None:
            self.temp_dir = "temp"
        else:
            self.temp_dir = Path(temp_dir)
        self.exec_program = self.wrap_program(program) 

        self.name = re.sub(" ", "_", self.name)




    def wrap_program(self, program):
        # do nothing 
        return program 
    
    def execute(self, additional_path: str = None, codebank: str = None):
        # create a macro dict 
        # read in the macros 
        if codebank is None:
            with open(f"{self.temp_dir}/codebank.lisp") as f1:
                macros = f1.read()
        else:
            macros = codebank
        # make LUT for macros 
        macro_dict, __ = split_macros(macros)
        # parse program
        parsed = parse_lispress(self.program)

        macros_used = []

        # walk program and replace with macros 
        def execute_helper(sexp):
            if type(sexp) != list:
                return sexp
            
            else:
                # include lower() because of pre-processing needed for grammar, raised the macro functions to uppercase
                if type(sexp[0]) == str and (sexp[0] in macro_dict.keys() or sexp[0].lower() in macro_dict.keys()):
                    # is a macro, then replace
                    sexp[0] = sexp[0].lower()
                    macros_used.append(sexp[0])
                    return execute_helper(replace_macro(sexp, macro_dict)[0])
                else:
                    # not a macro, add to program as-is  
                    return [execute_helper(x) for  x in sexp]

        new_parsed = execute_helper(parsed)
        new_program = render_pretty(new_parsed)
        if additional_path is not None:
            with open(additional_path, "w") as f1:
                f1.write(new_program) 


        return new_program, macros_used
    

