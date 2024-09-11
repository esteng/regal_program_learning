python_tuple_refactor_prompt = """Please rewrite the following two programs to be more efficient. 
The resulting programs MUST execute to the same result as the original programs.
Start by writing helper functions that can reduce the size of the code.  
{codebank_instr}

{queries_and_code}

Please format your answer as:
{answer_format_short}

Do not include any text that is not valid Python code.
Recall that no matter what, your program MUST be formatted in the following fashion: 
{answer_format_long}
Try to make your new programs as short as possible by introducing shared helper functions. Helper function parameters should be as general as possible and helper functions should be informatively named."""

textcraft_tuple_refactor_prompt = """Please rewrite the following programs to be more efficient. 
The code uses the a custom library, similar to the built-in library, which is sufficient for all tasks. 
Here's a description of the custom library: 
- check_inventory(): returns the inventory of the agent at the current step
- get_object(target): obtain target object directly from the environment
- craft_object(target): using the crafting commands, craft a target object using its ingredients which MUST already be in the inventory. Mention both quantity and exact name, e.g. "2 dark oak logs"

The resulting programs MUST execute exactly the same actions as the original programs.
Start by writing helper functions that can reduce the size of the code.  
{codebank_instr}

{queries_and_code}

Please format your answer as:
{answer_format_short}

Do not include any text that is not valid Python code.
Recall that no matter what, your program MUST be formatted in the following fashion: 
{answer_format_long}
Try to make your new programs as short as possible by introducing shared helper functions that can be re-used across multiple queries. Helper function parameters should be as general as possible and helper functions should be informatively named.
"""

logo_decompose_prompt = """You are an expert coder. For each query below, decompose it into its parts. 
Example: 
Query: Do some action 5 times and then do another action
Query (decomposed): 
The query asks: Do some action and then do another action
This can be decomposed into: 
1. repeat an action 
2. some action
3. another action

Query: {query}
Query (decomposed):"""

python_decompose_prompt = logo_decompose_prompt
textcraft_decompose_prompt = python_decompose_prompt


logo_comment_prompt = """Please add comments to the following program to explain what each chunk of code does with respect to the query. 
First, decompose the query into parts. Then comment the code with the query parts. 
Example: 
Query: Do some action and then do another action
Code: 
do_some_action()
do_another_action()

Query: Do some action 5 times and then do another action
Query (decomposed): 
The query asks: Do some action and then do another action
This can be decomposed into: 
1. repeat an action 
2. some action
3. another action
Commented code:
# repeat an action
for i in range(5): 
    # do some action
    do_some_action()
# do another action
do_another_action()

Here's a description of the custom library used in the code: 
- forward(x): move forward x pixels
- left(theta): rotate left by theta degrees
- right(theta): rotate right by theta degrees
- penup(): stop drawing
- pendown(): start drawing
- teleport(x, y, theta): move to position (x, y) with angle theta
- heading(): get the current angle of the turtle 
- isdown(): check if the pen is down
- embed(program, local_vars): runs the code in program using the current context and teleports back to the original position. Allows you to nest programs. Implementationally, embed gets the turtle state (is_down, x, y, heading), executes program, then returns to the original state.
- save(path): save the picture to file 

Query: {query}
Code:
{program}

Query (decomposed):
{decomposed_query}"""

textcraft_comment_prompt = """Please add comments to the following program to explain what each chunk of code does with respect to the query. 
First, decompose the query into parts. Then comment the code with the query parts. 
Example: 
Query: Do some action 5 times and then do another action
Code: 
for i in range(5): 
    do_some_action()
do_another_action()
Commented code:
# repeat an action 5 times
for i in range(5): 
    # do some action
    do_some_action()
# do another action
do_another_action()

Here's a description of the custom library used in the code: 
- check_inventory(): returns the inventory of the agent at the current step
- get_object(target): obtain target object directly from the environment
- craft_object(target): using the crafting commands, craft a target object using its ingredients which MUST already be in the inventory. Mention both quantity and exact name, e.g. "2 dark oak logs"

Do not output any text that is not valid Python code.

Query: {query}
Code:
{program}
"""

python_comment_prompt = """Please add comments to the following program to explain what each chunk of code does with respect to the query. 
First, decompose the query into parts. Then comment the code with the query parts. 
Example: 
Query: Do some action and then do another action
Code: 
do_some_action()
do_another_action()

Query: Do some action 5 times and then do another action
Query (decomposed): 
The query asks: Do some action and then do another action
This can be decomposed into: 
1. repeat an action 
2. some action
3. another action
Commented code:
# repeat an action
for i in range(5): 
    # do some action
    do_some_action()
# do another action
do_another_action()

Query: {query}
Code:
{program}

Query (decomposed):
{decomposed_query}"""




logo_tuple_refactor_prompt = """Please rewrite the following programs to be more efficient. 
The code uses the a custom python turtle library, similar to the built-in library, which is sufficient for all tasks. 
Here's a description of the custom library: 
- forward(x): move forward x pixels
- left(theta): rotate left by theta degrees
- right(theta): rotate right by theta degrees
- penup(): stop drawing
- pendown(): start drawing
- teleport(x, y, theta): move to position (x, y) with angle theta
- heading(): get the current angle of the turtle 
- isdown(): check if the pen is down
- embed(program, local_vars): runs the code in program using the current context and teleports back to the original position. Allows you to nest programs. Implementationally, embed gets the turtle state (is_down, x, y, heading), executes program, then returns to the original state.
- save(path): save the picture to file 

The resulting programs MUST produce exactly the same image as the original programs.
Start by writing helper functions that can reduce the size of the code.  
{codebank_instr}

{queries_and_code}

Please format your answer as:
{answer_format_short}

Do not include any text that is not valid Python code.
Recall that no matter what, your program MUST be formatted in the following fashion: 
{answer_format_long}
Try to make your new programs as short as possible by introducing shared helper functions. Helper function parameters should be as general as possible and helper functions should be informatively named.
If the original function uses `embed`, you will likely need to use `embed` in your version. All code to be repeated needs to be included within the triple quotes passed to embed.
"""

gpt_textcraft_agent_prompt = """Your task is to craft MineCraft objects in a simplified environment using python programs. 
You will use a custom library, similar to the built-in library, which is sufficient for all tasks. 

Here's a description of the custom library: 
- check_inventory(): returns the inventory of the agent at the current step
- get_object(target): obtain target object directly from the environment
- craft_object(target): using the crafting commands, craft a target object using its ingredients which MUST already be in the inventory. Mention both quantity and exact name, e.g. "2 dark oak logs"
{codebank_str}

You will be given a query and have to produce a program. {thought_str} 
Examples:
{icl_string}

Please generate ONLY the code to produce the answer and nothing else.
{crafting_commands}
Query: {query} 
{thought_and}Program: 
"""

gpt_feedback_prompt = """Your task is to craft MineCraft objects in a simplified environment using python programs. 
You will use a custom library, similar to the built-in library, which is sufficient for all tasks. 

Here's a description of the custom library: 
- check_inventory(): returns the inventory of the agent at the current step
- get_object(target): obtain target object directly from the environment
- craft_object(target): using the crafting commands, craft a target object using its ingredients which MUST already be in the inventory. Mention both quantity and exact name, e.g. "2 dark oak logs"
{codebank_str}

The following program failed to execute correctly as shown in the execution trace. Based on the execution trace, generate feedback on what went wrong and should be fixed in subsequent attempts. Please generate ONLY the code to produce the answer and nothing else.
{crafting_commands}
Query: {query} 
Generated Program:
{program}
Execution Trace:
{exec_trace}
Success: {succ}
Feedback: 
"""

gpt_retrial_prompt = """Your task is to craft MineCraft objects in a simplified environment using python programs. 
You will use a custom library, similar to the built-in library, which is sufficient for all tasks. 

Here's a description of the custom library: 
- check_inventory(): returns the inventory of the agent at the current step
- get_object(target): obtain target object directly from the environment
- craft_object(target): using the crafting commands, craft a target object using its ingredients which MUST already be in the inventory. Mention both quantity and exact name, e.g. "2 dark oak logs"
{codebank_str}

The following program failed to execute correctly as shown in the execution trace. Re-write the program incorporating feedback from the execution trace to ensure it executes correctly. For each re-written program output format your answer as
```
# Thought 1: Based on the execution trace, the issue is <explanation>
# Thought 2: Based on this explanation, I should change <things to change>
<re-written code>
```
{crafting_commands}
Query: {query} 
Generated Program:
{program}
Execution Trace:
{exec_trace}
Success: {succ}
Re-written Program: 
"""



gpt_logo_agent_prompt = """Your task is to draw simple figures using python Turtle graphics. 
You will use a custom turtle library, similar to the built-in library, which is sufficient for all tasks. 

Here's a description of the custom library: 
- forward(x): move forward x pixels
- left(theta): rotate left by theta degrees
- right(theta): rotate right by theta degrees
- penup(): stop drawing
- pendown(): start drawing
- teleport(x, y, theta): move to position (x, y) with angle theta
- heading(): get the current angle of the turtle 
- isdown(): check if the pen is down
- embed(program, local_vars): runs the code in program using the current context and teleports back to the original position. Allows you to nest programs. Implementationally, embed gets the turtle state (is_down, x, y, heading), executes program, then returns to the original state.
- save(path): save the picture to file 
{codebank_str}

You will be given a query and have to produce a program. {thought_str} 
Examples:
{icl_string}

Please generate ONLY the code to produce the answer and nothing else.
Query: {query} 
{thought_and}Program: 
"""



logo_codebank_refactor_prompt = """Please rewrite the following programs to be more efficient and re-usable. Many of the programs you will see are very specific. Try to make these more general-purpose. 
Rewriting rules:
- If you can, please reduce any redundant programs into a single helper function. Try to make programs more general by including parameters. Example:
```
INPUT:
def some_function():
    for _ in range(10):
        action()
def other_function():
    for _ in range(9):
        action()

OUTPUT:
# MERGED: some_function into better_function
# MERGED: other_function into better_function
def better_function(n):
    for _ in range(n):
        action()
```
- If you can, please replace any repeated code with a call to a helper function. Example:
```
INPUT:
def better_function(n):
    for _ in range(n):
        action()

def multi_action():
    a, b = get_params()
    for _ in range(10):
        action()
    result = a + b

OUTPUT: 
def better_function(n):
    for _ in range(n):
        action()

# EDITED: multi_action
def multi_action():
    a, b = get_params()
    better_function(10)
    result = a + b
```
- Please replace any hard-coded values with parameters, wherever it makes sense to do so. 
- You MUST follow the following commenting conventions:
    - If you edit a function, please add a comment that follows this template: # EDITED: <function name>
    - If you rename a function, please add a comment that follows this template: # RENAMED: <function name> to <new function name>
    - If you merge or compress several functions, please add a comment for EACH function following this template: # MERGED: <function name> into <new function name>
- Do not include anything in your response that is not code or a comment 

PROGRAMS:
```
{program}
```

Think through the problem step-by-step and then give your final answer as a block starting with the words: REWRITTEN PROGRAMS"""

python_codebank_refactor_prompt = logo_codebank_refactor_prompt
textcraft_codebank_refactor_prompt = logo_codebank_refactor_prompt


test_case_refactor_prompt = """The following functions have been changed. Please modify the following program accordingly to use the new function.
Do not include anything in your response that is not code or a comment. 
{func_change_str}
Query: {query}
Old test case: 
{program}
New test case:"""


# codebank_single_refactor_prompt = """Refactor the following function to improve generalization. You may use the PyTurtle library described below: 
# {library_str}
# You may also use the following helper functions: 
# {codebank_str}. Formulate your answer as NEW_FUNCTION:\n<funcion>\n and do not include anything in your answer that is not Python code after writing NEW_FUNCTION.
# If the function is unneccessary, please produce the following output:
# NEW_FUNCTION: DELETED
# {pass_fail_str}

# First, think through why the example programs succeeded or failed, step-by-step. Write these thoughts down as follows:
# Thought: The successful programs succeeded because...
# Thought: The successful programs (are or are not) as efficient as they can be because...
# Thought: The failed programs failed because...
# Thought: The most similar existing helper functions are...
# Thought: This function (does or does not) achieve its goals because...
# Thought: I can make this program better and more efficient by...
# Based on that information, refactor the program and give your output as NEW_FUNCTION.
# FUNCTION: 
# ```
# {func_str}
# ```"""

logo_codebank_failure_explanation_prompt = """The following program using function {func_name} failed to execute correctly. Please explain why it failed.
You will see one example of {func_name} being used correctly followed by an example of the program with {func_name} in it that failed.
Try to contrast the two programs in your explanation.

CORRECT EXAMPLE:
Query: {correct_query}
Program: {correct_program}

INCORRECT EXAMPLE:
Query: {incorrect_query}
Program: {incorrect_program}

Explanation:"""


logo_codebank_single_refactor_prompt = """Refactor the following function to improve performance. 
FUNCTION: 
```
{func_str}
```

You may use the PyTurtle library described below: 
```
{library_str}
```
You may also use the following helper functions: 
{codebank_str} 
{pass_fail_str}
{modular_str}
Thoughts:"""

textcraft_codebank_single_refactor_prompt = """Refactor the following function to improve performance. 
FUNCTION: 
```
{func_str}
```
You may use the custom library described below: 
```
{library_str}
```
You may use the following helper functions: 
{codebank_str} 
{pass_fail_str}
Thoughts:"""

python_codebank_single_refactor_prompt = """Refactor the following function to improve performance. 
FUNCTION: 
```
{func_str}
```

You may use the following helper functions: 
```
{codebank_str} 
```
{pass_fail_str}
Thoughts:"""



gpt_python_agent_prompt = '''Your task is to solve simple word problems by creating Python programs.
{codebank_str}

You will be given a query and have to produce a program. {thought_str} 
Examples:
{icl_string}

Please generate ONLY the code to produce the answer and nothing else.
Query: {query}
{thought_and}Program:'''



codebank_deduplication_prompt = """For the following functions, please determine which functions are duplicates or can be merged into a single function. 
Please also remove hardcoded values and make them parameters. 
You will be given a list of functions. Your output should have the following format: 

Thoughts: <your thoughts for each function>
NEW FUNCTIONS: <functions that are not duplicates>

FUNCTION_MAPPING: <function name> -> <function name>

For example, given the input: 

```
def add():
    return 2 + 3

def add_numbers(a, b):
    return a + b

def add_cats_dogs(cats, dogs):
    sum = cats + dogs
    return sum

def subtraction(num1, num2):
    return num1 - num2

def subtract_b_from_a(a, b)
    diff = a - b
    return diff
``` 
you would output:
Thoughts:
`add` adds 2 and 3. It hardcodes 2 and 3, which is bad practice.  
`add_numbers` adds two numbers, a and b.
`add_cats_dogs` adds the number of cats and dogs. The variables have specific names, which is bad practice.
`subtraction` subtracts two numbers, num1 and num2.
`subtract_b_from_a` subtracts b from a.
The following functions are performing the same task:
`add` and `add_numbers` and `add_cats_dogs` all add two numbers.
`subtraction` and `subtract_b_from_a` both subtract two numbers.
I want to remove all hardcoded values and make them parameters.
I can create a more general version adding called `add_numbers`. I will replace `add` with `add_numbers` and `add_cats_dogs` with `add_numbers`. 
I can create a more general version subtraction called `subtract_numbers`. I will replace `subtraction` with `subtract_numbers` and `subtract_b_from_a` with `subtract_numbers`.

NEW FUNCTIONS:
# I have removed all hardcoded values and made them parameters.
def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

FUNCTION_MAPPING:
# I will cover all functions: `add`, `add_numbers`, `add_cats_dogs`, `subtraction`, `subtract_b_from_a`
add -> add_numbers
add_numbers -> add_numbers
add_cats_dogs -> add_numbers
subtraction -> subtract_numbers
subtract_b_from_a -> subtract_numbers

You should not include anything in your response that is not code or a comment.
Note that you should make very specific functions more general. 
For example, the above example converts 
```
def add():
    return 2 + 3
```
to 
``` 
def add_numbers(a, b):
    return a + b
```

FUNCTIONS:
```
{functions}
```

"""

codebank_comment_prompt = """Please add a docstring-style comment to the following function describing what it does.
Format your output as:
DOCSTRING: <docstring>
Do not include any other output. 

FUNCTION: 
```
{function}
```
"""


python_self_consistency_prompt = """The following function (a response to the query) has been rewritten to be more efficient. 
Here are several options; please decide which one is the best. 

Query: {query}
PROGRAM:
{orig_function}

OPTIONS:
{option_str}

Format your output as:
BEST OPTION: <option number>

Do not include any other output."""
