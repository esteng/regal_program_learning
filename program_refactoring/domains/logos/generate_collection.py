import json 
import os 
import re 
from pathlib import Path
import argparse 

from num2words import num2words

from program_refactoring.utils import get_and_save_embeddings 

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


def sub_nums(s):
    # replace number digits with string version of number 
    # e.g. 10 becomes ten 
    nums = re.findall(r'\d+', s)
    for n in nums:
        s = s.replace(n, num2words(n))
    return s 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, 
                        help="path to generated images (obtained from `program_refactoring/domains/logos/generate_programs.py`)",
                        default="logo_data/images/train_200.jsonl") 
    parser.add_argument("--json_file", 
                        type=str, 
                        help="path to the json file containing the programs (obtained from `program_refactoring/domains/logos/convert_programs.py`)",
                        default="logo_data/python/train_200.jsonl")
    parser.add_argument("--output_dir",
                         type=str, 
                        help="path to directory to save the chromadb files",
                        default="logo_data/my_vectordb")
    parser.add_argument("--name", type=str, default="logos") 
    args = parser.parse_args()

    logo_docs, logo_ids = [], []
    logo_codes = []

    for fname in Path(args.image_dir).glob("*jpg"): 
        fonly = fname.name.split("_")
        __, idx = fonly[0], fonly[1]
        desc = " ".join(fonly[2:]) 
        desc = re.sub(".jpg", "", desc)
        logo_docs.append(desc)

        logo_ids.append(sub_nums(f"{idx}_{desc}")) 


    code_data = [json.loads(l) for l in open(args.json_file).readlines()]
    code_by_lang = {x['language'][0]: x['program'] for x in code_data}
    logo_codes = [code_by_lang[x] for x in logo_docs]


    get_and_save_embeddings(logo_docs, logo_codes, logo_ids, persist_directory=args.output_dir, name=args.name)
        
