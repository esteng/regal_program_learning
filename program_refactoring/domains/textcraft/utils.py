import ast 
import re 
import pdb 

from program_refactoring.headers import TEXTCRAFT_HEADER  
def get_func_names(program):
    """Get all function calls that are not part of the header"""
    parsed = ast.parse(program) 
    # get all expressions
    func_names = []
    header_func_names =  ['abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
    'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
    'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float',
    'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
    'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed',
    'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip']

    for node in ast.walk(parsed): 
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            try:
                func_names.append(node.value.func.id)
            except AttributeError:
                pass 
        elif isinstance(node, ast.Call):
            try:
                func_names.append(node.func.id)
            except AttributeError:
                try:
                    func_names.append(node.func.value.func.id)
                except AttributeError:
                    try:
                        func_names.append(node.func.value.id)
                    except AttributeError:
                        pass

        else:
            pass
        # ignore all imported functions 
        if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
            for name in node.names:
                if name.asname is None:
                    header_func_names.append(name.name)
                else:
                    header_func_names.append(name.asname)

    # get header names 
    header = ast.parse(TEXTCRAFT_HEADER)
    for node in ast.walk(header):
        if isinstance(node, ast.FunctionDef):
            header_func_names.append(node.name)

    

    func_names = list(set(func_names) - set(header_func_names))
    
    return func_names 


def clean_import(program):
    return re.sub("from .*\.codebank import \*", "", program).strip()