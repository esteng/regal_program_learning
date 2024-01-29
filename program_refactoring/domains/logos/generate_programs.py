import json 
import argparse 
from pathlib import Path 
import re 
import subprocess
import pdb 

from program_refactoring.domains.logos.convert_programs import read_jsonl
from program_refactoring.headers import LOGO_HEADER





def main(args):
    """
    Read in converted python from JSONL file and run the code using the custom turtle module 
    """

    data = read_jsonl(args.in_path)
    for i, line in enumerate(data): 
        content = f"{LOGO_HEADER}\n{line['program']}"
        name = re.sub(" ","_", line['language'][0])
        fname = f"{args.out_path}/logo_{i}_{name}.jpg"
        tailer = f"turtle.save('{fname}')"
        content = f"{content}\n{tailer}"

        with open("temp.py", 'w') as f: 
            f.write(content)


        p = subprocess.Popen(["python", "temp.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, errs = p.communicate()
        out, errs, = out.decode(), errs.decode()
        print(out)
        if len(errs) > 0: 
            print(errs)
            pdb.set_trace()





if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True, help="JSONL file to convert") 
    parser.add_argument("--out_path", type=str, required=True, help="JSONL file destination") 
    args = parser.parse_args()
    main(args)