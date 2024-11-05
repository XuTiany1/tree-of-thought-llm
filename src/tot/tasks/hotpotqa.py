import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.hotpotqa import * 
from tot.models import gpt
import tot.react.wikienv as wikienv, tot.react.wrappers as wrappers




class hotpotqa(Task):
    def __init__(self, file='hotpot_dev_v1_simplified.json'):

        super().__init__()

        # LOAD EXAMPLE
        folder = '../..//reAct/prompts/'
        prompt_file = 'prompts_naive.json'
        with open(folder + prompt_file, 'r') as f:
            prompt_dict = json.load(f)
        self.webthink_prompt = prompt_dict['webthink_simple6']

        # LOAD DATA
        self.file = os.path.join(DATA_PATH, 'hotpotQA', file)
        self.file = json.load(open(self.file))

        # INIT ENV variables
        env = wikienv.WikiEnv()
        env = wrappers.HotPotQAWrapper(env, split="dev")        # Wraps env with HotPotQAWrappe
        env = wrappers.LoggingWrapper(env)                      # Wraps env again with LoggingWrapper, adding logging functionality to monitor and track the actions

        self.steps = 5


    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
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


#-----------------------
# Step 1: Generation 
#-----------------------

    # x is the question
    # y is the context, which is basically the current context
    @staticmethod
    def propose_prompt_wrap(x: str, y:str='') -> str:

        return propose_prompt.format(examples=self.webthink_prompt,
                                     question=x, 
                                     context=y)
        

