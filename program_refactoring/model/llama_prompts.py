
llama_logo_agent_completion_prompt = '''[INST] Your task is to draw simple figures using python Turtle graphics. 
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
[/INST]
[PYTHON]
# Examples:
{icl_string}
[/PYTHON]
[INST]
Please generate ONLY the code to produce the answer and nothing else.
[/INST]
[PYTHON]
# Query: {query}
# Thought:'''

llama_python_agent_completion_prompt = '''[INST] Your task is to solve simple word problems by creating Python programs.
{codebank_str}

You will be given a query and have to produce a program. Begin with a comment that explains your reasoning. For example, you might write
# Thought: the query asks for an action, so I will use the action() function. 
[/INST]
[PYTHON]
# Examples:
{icl_string}
[/PYTHON]
[INST]
Please generate ONLY the code to produce the answer and nothing else.
[/INST]
[PYTHON]
# Query: {query}
# Thought:'''
