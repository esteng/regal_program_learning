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
