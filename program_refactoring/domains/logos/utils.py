import ast 
import re 
import pdb 
import numpy as np 
from scipy import ndimage


from program_refactoring.headers import LOGO_HEADER  
def get_func_names(program):
    """Get all function calls that are not part of the header"""
    try:
        parsed = ast.parse(program) 
    except (IndentationError, SyntaxError):
        return []

    # get all expressions
    func_names = []
    for node in ast.walk(parsed): 
        try:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.func.id == "embed":
                try:
                    embed_arg = node.value.args[0]
                except IndexError:
                    continue
                try:
                    func_names = func_names + get_func_names(embed_arg.value)
                except AttributeError:
                    continue
        except AttributeError:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func_names.append(node.value.func.id)

    # get header names 
    header = ast.parse(LOGO_HEADER)
    header_func_names = []
    for node in ast.walk(header):
        if isinstance(node, ast.FunctionDef):
            header_func_names.append(node.name)

    func_names = list(set(func_names) - set(header_func_names))


    
    return func_names 


def clean_import(program):
    return re.sub("from .*\.codebank import \*", "", program).strip()


if __name__ == "__main__": 
    program = """for j in range(4):
    embed('draw_semicircle()', locals())
    penup()
    forward(2)
    left(0.0)
    pendown()
"""
    print(get_func_names(program))


def convert_to_ascii(image):
    if image is None:
        return "ERROR"

    def block_mean(ar, fact):
        # https://stackoverflow.com/questions/18666014/downsample-array-in-python
        assert isinstance(fact, int), type(fact)
        sx, sy = ar.shape
        X, Y = np.ogrid[0:sx, 0:sy]
        regions = sy//fact * (X//fact) + Y//fact
        res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
        res.shape = (sx//fact, sy//fact)
        return res

    # downsample
    image = block_mean(image, fact = 40)

    ascii_str = []
    for row in image:
        row_str = []
        for pixel in row:
            if pixel == 255:
                row_str.append(".")
            else:
                row_str.append("#")
        ascii_str.append(" ".join(row_str))
    ascii_str = "\n".join(ascii_str)
    return ascii_str

def make_pass_fail_str(func, passing_cases, failing_cases): 
    def make_tc_str(cases, success=True, max=2):
        if success:
            sf_str = "success"
            sf_str_big = "SUCCEEDED"
        else:
            sf_str = "failure"
            sf_str_big = "FAILED"

        progs = []
        # get cases that the function is failing on 
        for tc in cases:
            pred_prog = tc.pred_node.program
            pred_prog = re.sub(f"from (.*)\.codebank import \*", "", pred_prog).strip()
            query = tc.pred_node.query
            progs.append(f"Query: {query}\nProgram ({sf_str}):\n```\n{pred_prog}\n```")
        if len(progs) > 0:
            # take up to the max, try to take the shortest ones first since longer programs have more of a chance of having other reasons for failure
            progs = sorted(progs, key = lambda x: len(x.split("\n")))
            progs = progs[0:max]

            progs = "\n\n".join(progs)
            sf_str = f"The function is used in the following functions which {sf_str_big}:\n{progs}"
            return sf_str
        else:
            return ""
    pass_str = make_tc_str(passing_cases, success=True)
    fail_str = make_tc_str(failing_cases, success=False)

    if pass_str != "" and fail_str != "":
        # get percentages
        pass_perc = len(passing_cases) / len(passing_cases + failing_cases)
        fail_perc = len(failing_cases) / len(passing_cases + failing_cases)
        pass_fail_str = f"""Try to increase the number of passing programs. Try to make programs general. For example, you can add parameters instead of hardcoded values or call other helper functions. First, for each failing query, explain why the programs do not accomplish the query's goal. Output this reasoning as: 
Thoughts:
1. The function passes some tests and fails others because <reason>. 
2. The failing queries <repeat queries here> asked for <intent>. 
3. The program failed because <reason>. 
4. This can be addressed by <change>. 
Then output your program so that all test cases pass, using the following format: NEW PROGRAM: <program>
Currently, {func._name} passes in {pass_perc * 100:.1f}% of cases and fails in {fail_perc*100:.1f}%.\n{pass_str}\n{fail_str}"""

    else:
        pass_fail_str = ""

    return pass_fail_str