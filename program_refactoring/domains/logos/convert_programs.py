import json
import argparse
from collections import namedtuple 
import re 
from sexpdata import loads, dumps, Symbol

import pdb 
MOVE_UNIT = 2
ROT_UNIT = 2 * 3.14
EPSILON_ANGLE = 2.86
EPSILON_DIST = 1/20
INF = 63

def read_jsonl(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def parse_program(program_str): 
    """
    Convert a Lisp-style program from dreamcoder data into
    a declarative python version of the same program 
    """

    # split program into nested () lists and convert 
    nested = loads(program_str)

    def convert_int(s):
        if s == "infinity":
            return INF
        try:
            return int(s)
        except ValueError:
            if s in ['epsionLength', 'epsilonLength']: 
                return "EPS_DIST"
            elif s in ['epsionAngle', 'epsilonAngle']:
                return "EPS_ANGLE"
            elif s == "1d": 
                return 1 * MOVE_UNIT 
            elif s == "0d":
                return 0 * MOVE_UNIT
            elif s == "1a": 
                return 1 * ROT_UNIT
            elif s == "0a":
                return 0 * ROT_UNIT
            elif s in ['i','j']: 
                # is a var: 
                return s 
            else:
                print(program_str)
                pdb.set_trace()
                raise ValueError(f"unknown int: {s}")

    def conversion_helper(chunk, d = 0, use_self=False):
        """
        Recursively convert each program chunk, maintaining indentation depth
        """

        prefix = "    "*d
        if use_self:
            # don't put self into the converted code, just use it at execution time
            fxn_prefix = ""
        else: 
            fxn_prefix = ""
        # base case: chunk is a convertable chunk, convert to python
        if isinstance(chunk, Symbol):
            # is epsilonAngle 
            return convert_int(chunk._val)

        if isinstance(chunk[0], Symbol): 
            if chunk[0]._val == "loop":
                # for loop: (loop, var, top, body)
                __, var, top = chunk[0:3]
                body = chunk[3:]
                var = var._val

                if str(top) == "inf":
                    top = "HALF_INF"
                return f"{prefix}for {var} in range({top}):\n{conversion_helper(body, d+1, use_self=use_self)}"
            
            elif chunk[0]._val == "for":
                # for loop: (loop, var, top, body)
                __, var, top, body = chunk
                var = var._val
                
                if str(top) == "inf": 
                    top = "HALF_INF"
                return f"{prefix}for {var} in range({top}):\n{conversion_helper(body, d+1, use_self=use_self)}"

            elif chunk[0]._val == "move": 
                # move command: (move, line, angle)
                __, line, angle = chunk
                if type(line) == list:
                    return f"{prefix}{fxn_prefix}forward({conversion_helper(line, d, use_self=use_self)})\n{prefix}{fxn_prefix}left({conversion_helper(angle, d, use_self=use_self)})"
                else:
                    try:
                        return f"{prefix}{fxn_prefix}forward({convert_int(line._val)})\n{prefix}{fxn_prefix}left({convert_int(angle._val)})\n"
                    except AttributeError:
                        return f"{prefix}{fxn_prefix}forward({convert_int(line._val)})\n{prefix}{fxn_prefix}left({conversion_helper(angle)})\n"

            elif chunk[0]._val in ["*d", "*l"]:
                # distance command
                __, unit, num = chunk
                unit = unit._val 
                unit = convert_int(re.sub("l", "", unit))
                orig_num = num
                try:
                    num = int(num) * MOVE_UNIT
                except TypeError:
                    # is a variable, i, j, etc
                    num = num._val
                if re.match("\d+", str(num)) is not None:
                    if type(unit) != str:
                        try:
                            dist = unit*num
                        except:
                            pdb.set_trace()
                    else:
                        return f"{unit}*{orig_num}"
                else:
                    dist = f"{unit}*{num}"

                # scale down dist 
                # max_dist = 50
                # if type(dist) != str:
                #     return dist/125 * max_dist 
                return dist 

            elif chunk[0]._val == "/a":
                # divide angle 
                __, __, angle_denom = chunk
                angle_rads = ROT_UNIT / angle_denom
                angle_degs = angle_rads * 180 / 3.14
                return angle_degs
            
            elif chunk[0]._val == "-a": 
                # subtract angle 1 from angle 2 
                __, angle1, angle2 = chunk
                return f"{conversion_helper(angle1, d)} - {conversion_helper(angle2, d)}"
            
            elif chunk[0]._val == "+a": 
                # add angle 1 and angle 2 
                __, angle1, angle2 = chunk
                return f"{conversion_helper(angle1, d)} + {conversion_helper(angle2, d)}"

            elif chunk[0]._val == "p": 
                return f"{prefix}{fxn_prefix}penup()\n{conversion_helper(chunk[1:], d, use_self=use_self)}\n{prefix}{fxn_prefix}pendown()"

            elif chunk[0]._val == "embed":
                # save current position, run embedded program, return
                # embed takes a string and evaluates it, I think this is the only way to do it in python 
                return f"""{prefix}{fxn_prefix}embed(\"\"\"{conversion_helper(chunk[1:], 0, use_self=True)}\"\"\", locals())"""

#                 return f"""{prefix}x, y = position()
# {prefix}theta = heading()
# {prefix}pen_was_down = isdown()\n
# {conversion_helper(chunk[1:], d)}
# {prefix}teleport(x, y, theta)\n
# {prefix}pendown() if pen_was_down else penup()\n"""
            else:
                print(program_str)
                pdb.set_trace()
                raise ValueError(f"Unknown symbol: {chunk[0]}")
            
        # recursion case: chunk is a list, convert each element
        else:
            return "\n".join([conversion_helper(c, d, use_self=use_self) for c in chunk])

    return conversion_helper(nested, d=0)


def main(args):
    data = read_jsonl(args.in_path)
    if args.human_path is not None:
        with open(args.human_path) as f1:
            human_language_lut = json.load(f1)
    else:
        human_language_lut = None

    new_data = []
    for line in data:
        program = parse_program(line['program'])
        new_line = {k:v for k,v in line.items()}
        new_line['program'] = program

        if human_language_lut is None:
            new_data.append(new_line)
        else:
            new_languages = human_language_lut[line['language'][0]]
            for lang in new_languages:
                new_line['language'] = [lang]
                new_data.append({k:v for k,v in new_line.items()})

    with open(args.out_path, 'w') as f:
        for nl in new_data: 
            f.write(json.dumps(nl) + "\n") 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True, help="JSONL file to convert") 
    parser.add_argument("--out_path", type=str, required=True, help="JSONL file destination") 
    parser.add_argument("--human_path", type=str, required=False, help="path to human language. If provided, human language will be used!")
    args = parser.parse_args()
    main(args)