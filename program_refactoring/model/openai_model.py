import openai
import time 
import re 
import pdb 
import os 
import tiktoken

import timeout_decorator

from program_refactoring.model.model import Model





class OpenAIModel(Model):

    def __init__(self, model_name):
        super().__init__(model_name)

    def clean(self, text):
        if "Program" in text:
            # find the last program 
            results = re.findall("Program:(.*)", text, flags=re.DOTALL)
            if len(results) > 0:
                return results[-1]
        if "```" in text:
            text = re.sub("```python", "", text)
            text = re.sub("```", "", text)
        return text 

    def build_model(self):
        api_key = os.environ.get("AZURE_API_KEY")
        openai.api_key = api_key
        openai.api_base = "https://instance-east-us2.openai.azure.com/"
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        # openai.api_version = 
        if self.model_name == "gpt-3.5-turbo":
            deployment_name = "gpt-35-turbo-16k" 
        elif self.model_name == "gpt-4-turbo": 
            deployment_name = "gpt-4-turbo"
        else:
            raise ValueError(f"Invalid model: {self.model_name}")

        completion_lambda = lambda x: openai.ChatCompletion.create(
            engine=deployment_name, 
            # prompt = x,
            messages=[{"role": "user", "content": x}],
            temperature=0.2,
            n = 1,
            max_tokens=2000,
            )
        
        
        return completion_lambda

    # def build_model(self):

    #     completion_lambda = lambda x: openai.ChatCompletion.create(
    #         model=self.model_name, 
    #         messages=[{"role": "user", "content": x}],
    #         temperature=0.2,
    #         n = 1,
    #         max_tokens=900,
    #         )
        
        
    #     return completion_lambda

    def build_wide_model(self, beam_size=10):

        completion_lambda = lambda x: openai.ChatCompletion.create(
            model=self.model_name, 
            messages=[{"role": "user", "content": x}],
            temperature=0.7,
            n = beam_size,
            max_tokens=900,
            )
        
        
        return completion_lambda

    @timeout_decorator.timeout(240, timeout_exception=ValueError) # set timeout for 4 minutes in case API gets stuck 
    def call_helper(self, x, agent):
        output = self.build_model()(x)
        text = output["choices"][0]["message"]["content"]
        if agent:
            return self.clean(text)
        return text

    def __call__(self, x, attempts = 0, infilling=False, agent=False, language=None, comment_tok = None): 
        # for now, infilling is just here for Llama compatability 
        try:
            output = self.call_helper(x, agent = agent)
            return output 

        except openai.error.APIError:
            if attempts > 3: 
                raise Exception("OpenAI API error, please try again later")
            time.sleep(5)
            return self(x, attempts + 1, agent = agent)
        except ValueError: 
            if attempts > 3: 
                raise Exception("API timed out")
            time.sleep(5)
            return self(x, attempts + 1, agent = agent)

    @timeout_decorator.timeout(240, timeout_exception=ValueError) # set timeout for 4 minutes in case API gets stuck 
    def wide_call_helper(self, x, agent):
        output = self.build_wide_model()(x)
        texts = []
        for choice in output["choices"]:
            text = choice['message']['content']
            texts.append(text)
        # no duplicates 
        return list(set(texts))

    def wide_call(self, x, attempts=0, agent=False):
        try:
            return self.wide_call_helper(x, agent = agent)

        except openai.error.APIError:
            if attempts > 3: 
                raise Exception("OpenAI API error, please try again later")
            time.sleep(5)
            return self.wide_call(x, attempts + 1, agent = agent)
        except ValueError: 
            if attempts > 3: 
                raise Exception("API timed out")
            time.sleep(5)
            return self.wide_call(x, attempts + 1, agent = agent)


class TokenCounter(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.tokenizer = tiktoken.get_encoding("c1100k_base")

    def __call__(self, x):
        return len(self.tokenizer.encode(x))