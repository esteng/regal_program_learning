import re 
import os 
import pdb 
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch

from program_refactoring.model.model import Model


class HFModel(Model):

    def __init__(self, model_name: str):
        super().__init__(model_name)

        bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name) #, padding_side="left")
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True) 

        self.gen_pipeline = pipeline(
            "text-generation",
            batch_size=1,
            model=self.model,
            tokenizer=self.tokenizer,   
            trust_remote_code=True,
            device_map="auto"
        )
        self.gen_pipeline.tokenizer.pad_token_id = self.model.config.eos_token_id

    def run(self, prompt, infilling=False, agent=False, language="PYTHON", comment_tok="#", pipeline_kwargs={}):
        """Run the model on a prompt"""

        output = self.gen_pipeline(
            prompt,
            max_new_tokens=500,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            num_return_sequences=1,
            **pipeline_kwargs) 

        # output = self.tokenizer.decode(outputs[0])
        return output[0]['generated_text']


    def __call__(self, x, infilling=False, agent=False, language="PYTHON", comment_tok="#", pipeline_kwargs={}): 
        return self.run(x, infilling, agent, language, comment_tok, pipeline_kwargs)


class CodeLlamaModel(HFModel):
    def __init__(self, model_name):
        super().__init__(model_name) 

    def clean_result(self, prompt, output, infilling=False, agent=False, language="PYTHON", comment_tok="#"):
        output = output[0]['generated_text']
        if infilling:
            # everything after prompt 
            gen_text = re.split(f"\[\/{language}\]", output)[-1]

            # everything before it starts making up new prompts
            program = re.split(f"{comment_tok} Query:", gen_text)[0].strip()
            return program 
        else:
            # model often generates a bunch of additional queries and programs
            # try to find the one that matches the query 
            my_query = re.findall(".*Query:(.*)", prompt)[-1].strip()
            # pdb.set_trace()
            try:
                query_m = re.search(f"({comment_tok} Query: {my_query}\n(.*?))(({comment_tok} Query)|($))", output, flags=re.DOTALL)
            except:
                pdb.set_trace()
            if query_m is not None:
                prog = query_m.group(1)
                # take everything before the close 
                prog = re.split(f"\[\/?{language}\]", prog)[0]
                prog = re.sub(f"\[\/?{language}\]", "", prog)
                return prog
            else:
                if my_query in output:
                    after_query_idx = output.index(my_query) + len(my_query)
                    prog = output[after_query_idx:]
                    prog = re.split(f"{comment_tok} Query", prog)[0]
                    prog = re.sub(f"\[\/?{language}\]", "", prog)
                    return prog 
                else:

                    # extract final [PYTHON] text 

                    matches = re.findall(f"\[{language}\](.*?)\[\/{language}\]", output, flags=re.DOTALL)
                    try:
                        final_match = matches[-1]
                    except IndexError:

                        pdb.set_trace()
                    prog = re.sub(f"\[\/?{language}\]", "", final_match)

                    # get query 
                    test_query = re.findall(".*Query:.*", prompt)[-1].strip()

                    try:
                        # often the model generates multiple programs, so we take the first one
                        all_progs = re.split(f"({comment_tok} Query:.*)", prog)
                        progs_by_query = {}
                        for i, row in enumerate(all_progs):
                            if row.startswith(f"{comment_tok} Query:"): 
                                try:
                                    body = all_progs[i+1]
                                except IndexError:
                                    body = "NONE"
                                progs_by_query[row.strip()] = body
                        try:
                            prog_for_query = progs_by_query[test_query]
                        except KeyError:
                            prog_for_query = "".join(all_progs[0:3])

                    except IndexError:
                        prog_for_query = prog

                    # split off any remaining [INST] things
                    if "[INST" in prog_for_query:
                        prog_for_query = re.split(f"\[\/?INST.*?\]?", prog_for_query)[0]

                return prog_for_query

    def run(self, prompt, infilling = False, agent=False, language = "PYTHON", comment_tok = "#", pipeline_kwargs={}):
        text =  super().run(prompt, pipeline_kwargs=pipeline_kwargs)
        return self.clean_result(prompt, text, infilling, agent, language, comment_tok) 
        
    def run_multiple(self, prompts, batch_size = 5, infilling = False, agent=False, language = "PYTHON", comment_tok = "#"):

        def dataset():
            print("Running prompts...")
            for prompt in tqdm(prompts, total=len(prompts)):
                yield prompt

        texts = self.gen_pipeline(dataset(),
                max_new_tokens=500,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                num_return_sequences=1,
                batch_size=batch_size) 

        texts = [x for x in texts]
        assert(len(prompts) == len(texts))
        results = []
        print("Cleaning results...")
        for prompt, text in tqdm(zip(prompts, texts), total=len(prompts)):
            res = self.clean_result(prompt, text, infilling, agent, language, comment_tok) 
            results.append(res)
        return results

class LemurModel(HFModel):
    def __init__(self, model_name):
        super().__init__(model_name) 

    def run(self, prompt, infilling = False, agent=False, language = "PYTHON", comment_tok = "#", pipeline_kwargs={}):
        text =  super().run(prompt)
        return self.clean_result(prompt, text, infilling, agent, language, comment_tok)
    
    def clean_result(self, prompt, output, infilling=False, agent=False, language="PYTHON", comment_tok="#"):
        # model often generates a bunch of additional queries and programs
        # try to find the one that matches the query 
        output = output[0]['generated_text']
        my_query = re.findall(".*Query:(.*)", prompt)[-1].strip()

        # query_m = re.search(f"({comment_tok} Query: {my_query}(.*?))(({comment_tok} Query)|($))", output, flags=re.DOTALL)
        query = re.sub("\?", "\?", my_query)
        try:
            query_m = re.findall(f"(?<=({comment_tok} Query: {query}))(.*?)(?=({comment_tok} Query))", output, flags=re.DOTALL)
        except re.error:
            query_m = None
        if len(query_m) == 0:
            # only 1 query so take the last one 
            query_m = re.findall(f"(?<=({comment_tok} Query: {query}))(.*)", output, flags=re.DOTALL)

        for match_obj in query_m:
            body = match_obj[1]
            # replace all comment lines 
            body = re.sub(f"{comment_tok}.*", "", body)
            body = body.strip()
            # if len(body) > 0:
                # prog = match_obj[1]
            prog = body
            prog = re.sub(f"\[\/?{language}\]", "", prog)
            prog = re.sub("<\|im_end\|>", "", prog)
            prog = re.sub("<\|im_start\|> .*:", "", prog)
            prog = re.sub("Please generate.*","",prog)
            prog = prog.strip()
            if len(prog)>0:
                return prog

        # extract first [PYTHON] text 

        matches = re.findall(f"<\|im_start\|> user:(.*?)<\|im_end\|>", output, flags=re.DOTALL)
        try:
            final_match = matches[-1]
        except IndexError:
            if my_query in output:
                after_query_idx = output.index(my_query) + len(my_query)
                prog = output[after_query_idx:]
                prog = re.split(f"{comment_tok} Query", prog)[0]
                prog = re.sub("Please generate.*","",prog)
                return prog
            else:
                return output

        prog = re.sub(f"\[\/?{language}\]", "", final_match)
        prog = re.sub("<\|im_end\|>", "", prog)
        prog = re.sub("<\|im_start\|> user:", "", prog)

        # get query 
        test_query = re.findall(".*Query:.*", prompt)[-1].strip()

        try:
            # often the model generates multiple programs, so we try to get from anywhere
            all_progs = re.split(f"({comment_tok} Query:.*)", output)
            progs_by_query = {}
            for i, row in enumerate(all_progs):
                if row.startswith(f"{comment_tok} Query:"): 
                    try:
                        body = all_progs[i+1]
                    except IndexError:
                        body = "NONE"
                    progs_by_query[row.strip()] = body
            try:
                prog_for_query = progs_by_query[test_query]
            except KeyError:
                # take the last one 
                prog_for_query = "".join(all_progs[-2:])
                # prog_for_query = "".join(all_progs[0:3])

        except IndexError:
            prog_for_query = prog

        prog_for_query = re.sub("<\|im_end\|>", "", prog_for_query)
        prog_for_query = re.sub("<\|im_start\|> user:", "", prog_for_query)
        prog_for_query = re.sub("Please generate.*","",prog_for_query)

        return prog_for_query

    def run_multiple(self, prompts, batch_size = 3, infilling = False, agent=False, language = "PYTHON", comment_tok = "#"):

        def dataset():
            print("Running prompts...")
            for prompt in tqdm(prompts, total=len(prompts)):
                yield prompt

        texts = self.gen_pipeline(dataset(),
                max_new_tokens=700,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                num_return_sequences=1,
                batch_size=batch_size) 

        texts = [x for x in texts]
        assert(len(prompts) == len(texts))
        results = []
        print("Cleaning results...")
        for prompt, text in tqdm(zip(prompts, texts), total=len(prompts)):
            res = self.clean_result(prompt, text, infilling, agent, language, comment_tok) 
            results.append(res)
        # pdb.set_trace()
        return results