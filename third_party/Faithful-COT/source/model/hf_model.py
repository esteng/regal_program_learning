import os
cwd = os.getcwd()
if cwd.endswith("source/model"):
    os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from configuration.configuration import Config
from dataset.utils import CODE_STOP_TOKEN, CODE_MAX_TOKEN, NO_CODE_STOP_TOKEN, NO_CODE_MAX_TOKEN
import sys
from io import StringIO
import openai
import itertools
from model.solver.MWP import math_solver
from model.solver.CLUTRR import CLUTRR_solver
from model.solver.StrategyQA import datalog_solver
from model.solver.saycan import pddl_planner
import errno
import os
import signal
import functools
import re

import torch

from program_refactoring.model.hf_model import CodeLlamaModel

from source.model.codex import Model

# The following are packages/funtions for exponential backoff
# (ref. https://platform.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff)
# in order to deal with OpenAI API "rate limit reached" errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)



class HFModel(Model):
    '''The model class for HF models.'''
    def __init__(self, config):
        '''Initialize the model with the given configuration.
        @:param config: the configuration object, see source/configuration/configuration.py for details
        '''
        super(HFModel, self).__init__(config)

        # dataset parameters
        self.dataset_name = config.dataset_name # name of evaluation dataset

        # core parameters
     

        self.LM = CodeLlamaModel(config.LM)
        self.prompt_name = config.prompt_name
        self.max_tokens = config.max_tokens

        # decoding parameters
        self.n_votes = config.n_votes  # number of programs to generate
        self.temperature = config.temperature  # temperature for the solver LM
        self.batch_size = config.batch_size  # batch size for querying the LM

        # analysis-related parameters
        self.no_solver = config.no_solver # whether to use the LM to solve the answer instead of calling the solver

        # load the prompt and template
        prompt_path = f"source/prompt/{config.dataset_name}/{self.prompt_name}_prompt.txt" # the prompt containing few-shot examples
        template_path = f"source/prompt/{config.dataset_name}/{self.prompt_name}_template.txt" # the template to convert a new example
        with open(prompt_path, 'r', encoding='utf-8') as fr:
            self.prompt = fr.read()
        with open(template_path, 'r', encoding='utf-8') as fr:
            self.template = fr.read()

        # load the API keys
        self.api_keys = itertools.cycle(config.api_keys)
        self.org_ids = itertools.cycle(config.org_ids)


    
    def _query(self, prompt, stop, LM, n=1, logprobs=None, temperature=0.0, max_tokens=1024):
        """query an HF model
        @:param prompt (str): the prompt to be fed to the model
        @:param stop (list): the stop tokens
        @:param LM (str): the LM to be queried
        @:param n (int): the number of completions to be returned
        @:param logprobs (int): the number of most likely tokens whose logprobs are to be returned
        @:param temperature (float): the temperature of the model
        @:param max_tokens (int): the maximum number of tokens to be returned

        @:return (dict): the response from the model
        """
        if temperature == 0.0:
            completions = LM.gen_pipeline(prompt,
                max_new_tokens=max_tokens,
                pad_token_id=self.LM.tokenizer.eos_token_id,
                eos_token_id=self.LM.tokenizer.eos_token_id,
                do_sample=False,
                num_return_sequences=1)
        else:
            completions = LM.gen_pipeline(prompt,
                max_new_tokens=max_tokens,
                pad_token_id=self.LM.tokenizer.eos_token_id,
                eos_token_id=self.LM.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=1)
        

        completions = [x['generated_text'] for x in completions]
        return completions


if __name__ == "__main__":
    '''Run a simple test.'''

    dataset_name = "date"

    config_frn = f"source/configuration/config_files/{dataset_name}/codellama-7b-NL+SL.json"
    config = Config.from_json_file(config_frn)

    config.dataset_name = dataset_name

    model = HFModel(config)

    example = {"question": "Jane thinks today is 6/18/2019, but John thinks today is 6/19/2019. John is correct. What is the date 10 days ago in MM/DD/YYYY?",
               "answer": "#### 06/09/2019"}

    output = model.predict(example)
    answer = output["answer"]
    completion = output["completion"]
    print("Answer:", [answer])
    print("Completion:", [completion])
