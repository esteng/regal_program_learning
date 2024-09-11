LOGO_HEADER = """from program_refactoring.domains.logos.pyturtle import PyTurtle
from program_refactoring.domains.logos.pyturtle import HALF_INF, INF, EPS_DIST, EPS_ANGLE

turtle = PyTurtle()
def forward(dist):
    turtle.forward(dist)
def left(angle):
    turtle.left(angle)
def right(angle):   
    turtle.right(angle)
def teleport(x, y, theta):
    turtle.teleport(x, y, theta)
def penup():
    turtle.penup()
def pendown():
    turtle.pendown()
def position():
    return turtle.x, turtle.y
def heading():
    return turtle.heading
def isdown():
    return turtle.is_down
def embed(program, local_vars):
    # NOTE: Program must be a string, and locals() must be provided as local_vars
    # expected usage: embed("function(arg)", locals())
    return turtle.embed(program, local_vars)"""

SIMPLE_LOGO_HEADER = """from program_refactoring.domains.logos.pyturtle import HALF_INF, INF, EPS_DIST, EPS_ANGLE

# available functions: 
# forward(dist), left(angle), right(angle), teleport(x, y, theta), penup(), pendown(), position(), heading(), isdown(), embed(program, local_vars)
# NOTE: For embed, program must be a string, and locals() must be provided as local_vars
# expected usage: embed("function(arg)", locals())"""



PYTHON_HEADER = """from datetime import *
from dateutil.relativedelta import *"""

TEXTCRAFT_HEADER="""import sys
from typing import List
sys.path.append("/nas-ssd2/archiki/program_refactoring/third_party/EnvironmentWebs/environments/")
from textcraft.env import TextCraft

env = TextCraft(minecraft_dir="/nas-ssd2/archiki/program_refactoring/third_party/EnvironmentWebs/environments/textcraft/")

global done
done = False

def step(command: str) -> str:
    global done
    obs, _, local, _, _ = env.step(command)
    if local:
        done = True
    return obs

def check_inventory() -> str:
    obs = step('inventory')
    print(obs)
    # return the inventory present in the observation
    # Example output: Inventory: [oak planks] (2)
    return obs

def get_object(target: str) -> None:
    obs = step("get " + target)
    print(obs)

def craft_object(target: str, ingredients: List[str]) -> None:
    obs = step("craft " + target + " using " + ", ".join(ingredients))
    print(obs)

"""
