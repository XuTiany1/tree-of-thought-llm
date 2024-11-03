import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.hotpotQA import *
from tot.models import gpt



class hotpotQAtask(Task):


    def __init__(self, file='hotpotqa_dev_distractor_v1.json'):

        super().__init__()
        
        # Load dataset
        path = os.path.join(DATA_PATH, 'hotpotqa', file)
        with open(path, 'r') as f:
            data = json.load(f)
        self.data = data

        # max depth of tree
        self.steps = 3

        # STOPPING criteria for generation (Not sure what)
        # TODO: Tackle this

        # Value cache
        self.value_cache = {}


    # TODO: whats the point of this even..
    def __len__(self) -> int:
        return len(self.data)


#-------------------------
# Basic input processing and final output testing
#-------------------------
    def get_input(self, idx: int) -> dict:
        item = self.data[idx]
        question = item['question']
        context = ''
        for doc in item['context']:
            title = doc[0]
            paragraphs = ' '.join(doc[1])
            context += f"{title}: {paragraphs}\n"
        return {'question': question, 'context': context}

    def test_output(self, idx: int, output: str):
        item = self.data[idx]
        true_answer = item['answer'].lower()
        pred_answer = output.strip().split('\n')[-1].lower()
        correct = int(true_answer == pred_answer)
        return {'r': correct}

#-------------------------
# Prompting functions
#-------------------------

    #TODO: implement this
    @staticmethod
    def standard_prompt_wrap(x, y=''):
        return standard_prompt.format(question=x['question']) + y

    #TODO: implement this
    @staticmethod
    def cot_prompt_wrap(x, y=''):
        return cot_prompt.format(question=x['question']) + y

    @staticmethod
    def propose_prompt_wrap(x: dict, y: str = '') -> str:
        
        # 'x' is dictionary containing:
        #       question
        #       context
        # 'y' = previous reasoning steps

        return propose_prompt.format(question=x['question'], 
                                     context=x['context'],
                                     number_of_thoughts = 3
                                     ) + y
        # Like this function should allow my model to get 3 thought/action pair
        # then, the model should take them as a list and then perform an observation on them
        # then, it should evaulate the plausibility of the partial solution using the value prompt or osmething
        # i don't know if I should implement them here or at the bfs.py file or something?
        # pLEASE HELP  






#-----------------------------------------------
# Evaluation Functions
#-----------------------------------------------
    @staticmethod
    def value_prompt_wrap(question: str, thought: str, action: str) -> str:
        # Prompt for evaluating each individual thought-action pair
        return value_prompt.format(question=question, thought=thought, action=action)


    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs):
        value_map = {'high': 1.0, 'medium': 0.5, 'low': 0.1}
        values = []
        for output in value_outputs:
            lines = output.strip().split('\n')
            if lines:
                first_line = lines[0].lower()
                for key in value_map:
                    if key in first_line:
                        values.append(value_map[key])
                        break
        return sum(values) / len(values) if values else 0.0






































































