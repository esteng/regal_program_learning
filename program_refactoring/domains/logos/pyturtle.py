import re 
import os 
import pdb 
from PIL import Image
from matplotlib import pyplot as plt 
import matplotlib as mpl
import matplotlib.style as mplstyle
from pathlib import Path
import ast 
mpl.use('Agg')

mpl.rcParams["path.simplify_threshold"] = 0.0
mpl.rcParams['agg.path.chunksize'] = 10000
mplstyle.use('fast')
import numpy as np

from program_refactoring.domains.logos.utils import get_func_names
from program_refactoring.codebank import CodeBank

HALF_INF = 63
INF = 126
EPS_DIST = 1/20
EPS_ANGLE = 2.86

class PyTurtle:
    """
    Wrapper class that makes matplotlib look like Turtle 
    """

    def __init__(self, ax=None):
        self.x = 0
        self.y = 0
        self.heading = 0
        if ax is None: 
            self.fig, self.ax = plt.subplots(1,1, figsize=(20,20))
        else:
            self.ax = ax
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(-50, 50)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # remove frame
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.is_down = True


    def forward(self, dist):
        x0, y0 = self.x, self.y
        x1 = x0 + dist * np.cos(self.heading)
        y1 = y0 + dist * np.sin(self.heading)
        if self.is_down:
            # print(f"plotting from {x0, y0} to {x1, y1}")

            line_distance = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            # print(f"line is {line_distance} long")
            self.ax.plot([x0, x1], [y0, y1], color='black', linewidth=3)
        self.x = x1
        self.y = y1

    def left(self, angle_deg):
        angle_rad = angle_deg * np.pi / 180
        self.heading += angle_rad
        # print(f"rotated by {angle_deg} degrees, heading is now {self.heading} rad")

    def right(self, angle_deg):
        angle_rad = angle_deg * np.pi / 180
        self.heading -= angle_rad

    def teleport(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading

    def penup(self):
        self.is_down = False

    def pendown(self):
        self.is_down = True

    def heading(self):
        return self.heading

    def save(self, path):
        # save with fixed dimensions of the whole figure
        self.fig.canvas.draw()

        pil_img = Image.frombytes('RGB', 
            self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        
        pil_img.save(path)
        # plt.imsave(path, dpi=100) 
        # print(f"saved to {path}")


    def embed(self, program, local_vars):
        # do substitution in program
        for fxn in ["forward", "left", "right", "penup", "pendown", "position", "heading", "isdown", "teleport"]:
            program = re.sub(f"^{fxn}\(", f"self.{fxn}(", program)

        x, y = self.x, self.y
        theta = self.heading
        pen_was_down = self.is_down
        # merge
        for k, v in locals().items():
            local_vars[k] = v
        try:
            exec(program, globals(), local_vars )
        except Exception as e:
            print(f"Error executing program {program}")
            raise e

        self.teleport(x, y, theta)
        self.pendown() if pen_was_down else self.penup()
