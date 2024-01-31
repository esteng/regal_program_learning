import numpy as np
from typing import Any, List
import pdb 
import ast 
import re 
import logging 

from program_refactoring.tree.node import Node, LogoNode, PythonNode, TextCraftNode
from program_refactoring.codebank.function import Function
from program_refactoring.model.model import Model
from program_refactoring.model.openai_model import OpenAIModel
from program_refactoring.model.prompts import test_case_refactor_prompt
from program_refactoring.domains.logos.visual_sim import vis_compare


logger = logging.getLogger(__name__)

try:
    import sys
    sys.path.insert(0, "third_party/Faithful-COT/source")
    from evaluate.evaluate_answer_acc import is_correct, extract_gold_answer, extract_pred_answer
except ModuleNotFoundError:
    logger.warn("Running without Faithful-CoT checks...")

class TestCase:
    def __init__(self, pred_node: Node, gold_node: Node, model: Model, expected: Any = None, is_correct=None):
        self.pred_node = pred_node
        self.gold_node = gold_node
        self.model = model
        self.gold_node = gold_node
        self.expected = expected
        self.is_correct = is_correct 

    def parse_response(self, output):
        output = re.sub("from temp\.codebank import \*", "", output)
        # remove added comments 
        output = re.sub("^#.*$", "", output)
        output = output.strip()
        return output 

    def refactor(self, old_to_new_mapping):
        # rewrite calls after helper functions have been rewritten
        logger.info(f"Signature changed, refactoring test case")
        change_str = []
        old_prog_str, new_prog_str = [], []
        for old_prog, new_prog in old_to_new_mapping:
            old_prog_str.append(old_prog)
            new_prog_str.append(new_prog)
        old_prog_str = "\n".join(old_prog_str)
        new_prog_str = "\n".join(new_prog_str)
            # change_str.append(f"Old program:\n{old_prog}\nNew program:\n{new_prog}\n\n")
        # change_str = "".join(change_str)
        change_str = ""
        if len(old_prog_str) > 0:
            change_str = f"Old program(s):\n{old_prog_str}\nNew program(s):\n{new_prog_str}\n"

        program_str = self.pred_node.program
        program_str = re.sub("from .*\.codebank import \*", "", program_str)
        program_str = program_str.strip()

        prompt = test_case_refactor_prompt.format(func_change_str = change_str, program = program_str, query = self.pred_node.query)

        response = self.model(prompt, agent=False)
        try:
            output = self.parse_response(response)
        except: 
            pdb.set_trace

        logger.info(f"New test case call:\n{output}")
        self.pred_node.program = output 
        try:
            self.pred_node.exec_program = self.pred_node.wrap_program(self.pred_node.program)
        except (SyntaxError, ValueError, AttributeError) as e:
            # back off to original program 
            self.pred_node.exec_program = self.pred_node.program
            self.is_correct = False

    def get_acc(self, overwrite: List[Function]):
        # we need to overwrite the codebank with the new functions
        pred = self.pred_node.execute()
        if pred == self.expected:
            return True
        return False
    
    def to_json(self):
        return {
            "node": self.pred_node.to_json(),
            "model": self.model.to_json(),
            "expected": self.expected
        }

    @classmethod    
    def from_json(cls, json):
        pred_node = Node.from_json(json["pred_node"])
        gold_node = Node.from_json(json["gold_node"])
        model = Model.from_json(json['model'])
        expected = json.get('expected', None)
        return cls(pred_node, gold_node, model, expected)

class TextCraftTestCase(TestCase):

    def get_acc(self, task = "textcraft", overwrites: List[Function] = []):
        try:
            parsed = ast.parse(self.pred_node.exec_program)
        except SyntaxError:
            return False 

        # pull out import statements and group together 
        if len(overwrites) > 0:
            imports, skip = [], []
            body = []
            for i, line in enumerate(parsed.body):
                if isinstance(line, ast.ImportFrom):
                    imports.append(line)
                    skip.append(i)
                else:
                    body.append(line)
            overwrite_code = "\n".join([x._original_code for x in overwrites])
            parsed_overwrite = ast.parse(overwrite_code) 
            parsed.body = imports + parsed_overwrite.body + body
            new_exec_program = ast.unparse(parsed)
            self.pred_node.exec_program = new_exec_program


        pred = self.pred_node.execute()
        gold = self.gold_node.execute()
        return pred == gold

    def to_json(self):
        return {
            "pred_node": self.pred_node.to_json(),
            "gold_node": self.gold_node.to_json(),
            "model": self.model.to_json(),
        }

    @classmethod    
    def from_json(cls, data):
        if data is None:
            return None
        data['pred_node']['type'] = "pred"
        data['gold_node']['type'] = "gold"
        pred_node = TextCraftNode.from_json(data["pred_node"])
        model = OpenAIModel.from_json(data['model'])
        gold_node = TextCraftNode.from_json(data['gold_node'])
        return cls(pred_node, gold_node, model)   

class LogoTestCase(TestCase):

    def get_acc(self, task = "logo", overwrites: List[Function] = []):
        try:
            parsed = ast.parse(self.pred_node.exec_program)
        except SyntaxError:
            return False 

        # pull out import statements and group together 
        if len(overwrites) > 0:
            imports, skip = [], []
            body = []
            for i, line in enumerate(parsed.body):
                if isinstance(line, ast.ImportFrom):
                    imports.append(line)
                    skip.append(i)
                else:
                    body.append(line)
            overwrite_code = "\n".join([x._original_code for x in overwrites])
            parsed_overwrite = ast.parse(overwrite_code) 
            parsed.body = imports + parsed_overwrite.body + body
            new_exec_program = ast.unparse(parsed)
            self.pred_node.exec_program = new_exec_program


        pred = self.pred_node.execute()
        gold = self.gold_node.execute()
        vis_sim = vis_compare(pred, gold)
        if vis_sim > 0.98:
            return True
        return False

    def to_json(self):
        return {
            "pred_node": self.pred_node.to_json(),
            "gold_node": self.gold_node.to_json(),
            "model": self.model.to_json(),
        }

    @classmethod    
    def from_json(cls, data):
        if data is None:
            # adding this for now so we can use an agent on checkpoints without test cases 
            return None
        data['pred_node']['type'] = "pred"
        data['gold_node']['type'] = "gold"
        pred_node = LogoNode.from_json(data["pred_node"])
        model = OpenAIModel.from_json(data['model'])
        gold_node = LogoNode.from_json(data['gold_node'])
        # expected = np.array(data['expected'])
        return cls(pred_node, gold_node, model)
    


class PythonTestCase(TestCase):

    def get_acc(self, task:str, overwrites: List[Function]):
        parsed = ast.parse(self.pred_node.exec_program)
        # pull out import statements and group together 
        imports, skip = [], []
        body = []
        for i, line in enumerate(parsed.body):
            if isinstance(line, ast.ImportFrom):
                imports.append(line)
                skip.append(i)
            else:
                body.append(line)
        overwrite_code = "\n".join([x._original_code for x in overwrites])
        parsed_overwrite = ast.parse(overwrite_code) 
        parsed.body = imports + parsed_overwrite.body + body
        new_exec_program = ast.unparse(parsed)
        self.pred_node.exec_program = new_exec_program

        try:
            pred = self.pred_node.execute()
        except:
            pred = None
        gold = self.gold_node.execute()
        pred_result = extract_pred_answer(task, pred)
        gold_result = extract_pred_answer(task, gold)
        result = is_correct(task, gold_result, pred_result) 
        if type(result) == bool:
            return result
        return result.all()

    def to_json(self):
        return {
            "pred_node": self.pred_node.to_json(),
            "gold_node": self.gold_node.to_json(),
            "model": self.model.to_json(),
        }

    @classmethod    
    def from_json(cls, data):
        if data is None:
            # adding this for now so we can use an agent on checkpoints without test cases 
            return None
        data['pred_node']['type'] = "pred"
        data['gold_node']['type'] = "gold"
        pred_node = PythonNode.from_json(data["pred_node"])
        model = OpenAIModel.from_json(data['model'])
        gold_node = PythonNode.from_json(data['gold_node'])
        # expected = np.array(data['expected'])
        return cls(pred_node, gold_node, model)
