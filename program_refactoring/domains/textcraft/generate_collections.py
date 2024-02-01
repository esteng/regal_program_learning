import json 
import os 
import re 
from pathlib import Path
import argparse 

# from num2words import num2words

from program_refactoring.utils import get_and_save_embeddings 

os.environ['OPENAI_API_KEY']


# def sub_nums(s):
#     # replace number digits with string version of number 
#     # e.g. 10 becomes ten 
#     nums = re.findall(r'\d+', s)
#     for n in nums:
#         s = s.replace(n, num2words(n))
#     return s 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="path to the jsonl file containing the dataset", default="python_data/textcraft/gpt-4/train.json")
    parser.add_argument("--output_dir", type=str, help="path to the output directory", default="python_data/textcraft/gpt-4/my_vectordb/")
    parser.add_argument("--name", type=str, help="name of the dataset", default="textcraft")
    args = parser.parse_args()

    python_docs, python_ids = [], []
    python_codes = []
    metadata = []


    with open(args.data_file) as f1:
        data = json.load(open(args.data_file))
   

    for d in data:
        entry = data[d]
        id = f"{args.name}_{d}"
        query = entry['query']
        code = entry['gold program']
        metadata.append(entry['commands'])
        python_docs.append(query)
        python_ids.append(id)
        python_codes.append(code)


    get_and_save_embeddings(python_docs, python_codes, python_ids, persist_directory=args.output_dir, metadata=metadata, name="textcraft")
        
