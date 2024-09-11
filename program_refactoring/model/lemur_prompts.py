


lemur_logo_agent_completion_prompt = '''<|im_start|> user:
Your task is to draw simple figures using python Turtle graphics. 
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

You will be given a query and have to produce a program. Begin with a comment that explains your reasoning. For example, you might write
# Thought: the query asks for a line, so I will use the forward() function. 
# Examples:
{icl_string}
Please generate ONLY the code to produce the answer and nothing else.
<|im_end|>
<|im_start|> agent: 
# Query: {query}
# Thought:'''

lemur_python_agent_completion_prompt = '''<|im_start|> user:
Your task is to solve simple word problems by creating Python programs.
{codebank_str}

You will be given a query and have to produce a program. Begin with a comment that explains your reasoning. For example, you might write
# Thought: the query asks for an action, so I will use the action() function. 
<|im_end|>
<|im_start|> agent:
{icl_string}
<|im_end|>
<|im_start|> user:
Please generate ONLY the code to produce the answer and nothing else.
<|im_end|>
<|im_start|> agent:
# Query: {query}
# Thought:'''

lemur_textcraft_agent_completion_prompt = '''<|im_start|> user: 
Your task is to craft MineCraft objects in a simplified environment using python programs. 
You will use a custom library, similar to the built-in library, which is sufficient for all tasks. 

Here's a description of the custom library: 
- check_inventory(): returns the inventory of the agent at the current step
- get_object(target): obtain target object directly from the environment
- craft_object(target): using the crafting commands, craft a target object using its ingredients which MUST already be in the inventory. Mention both quantity and exact name, e.g. "2 dark oak logs"
{codebank_str}

You will be given a query and have to produce a program. Begin with a comment that explains your reasoning. For example, you might write
# Thought: the query asks to get <item>, so I will use the get_object() function. 
# Examples:
<|im_end|>
<|im_start|> agent: 
# Examples:
{icl_string}
<|im_end|>
<|im_start|> user:
Please generate ONLY the code to produce the answer and nothing else.
<|im_end|>
<|im_start|> agent:
# Query: {query}
# Thought:'''

lemur_retrial_prompt = """<|im_start|> user: Your task is to craft MineCraft objects in a simplified environment using python programs. 
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
<|im_end|>
<|im_start|> agent:
# Re-written Program: 
"""

