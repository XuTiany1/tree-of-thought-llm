# hotpotqa_task.py

import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.hotpotQA import *
from tot.models import gpt

class HotpotQATask(Task):
    """
    Input (x): A dictionary containing 'question' and 'context'.
    Output (y): The answer to the question, possibly including reasoning steps.
    Reward (r): A score indicating correctness, such as 1 for correct and 0 for incorrect.
    """

    def __init__(self, file='hotpotqa_dev_distractor_v1.json'):
        """
        file: A JSON file containing HotpotQA data.
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'hotpotqa', file)
        with open(path, 'r') as f:
            data = json.load(f)
        self.data = data
        self.value_cache = {}
        self.steps = 8  # Adjust based on desired number of reasoning steps
        self.stops = ['\nAnswer:\n', None]

    def __len__(self) -> int:
        return len(self.data)

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

    # Prompt Wrapping Functions

    @staticmethod
    def standard_prompt_wrap(x: dict, y: str = '') -> str:
        return standard_prompt.format(question=x['question'], context=x['context']) + y

    @staticmethod
    def cot_prompt_wrap(x: dict, y: str = '') -> str:
        return cot_prompt.format(question=x['question'], context=x['context']) + y

    @staticmethod
    def propose_prompt_wrap(x: dict, y: str = '') -> str:
        # Assuming 'y' contains previous reasoning steps
        return propose_prompt.format(question=x['question'], context=x['context']) + y

    @staticmethod
    def value_prompt_wrap(x: dict, y: str) -> str:
        return value_prompt.format(question=x['question'], context=x['context'], reasoning=y)

    @staticmethod
    def value_outputs_unwrap(x: dict, y: str, value_outputs: list) -> float:
        evaluations = [output.strip().split('\n')[-1].lower() for output in value_outputs]
        value_map = {'low': 0.1, 'medium': 0.5, 'high': 1.0}
        values = [value_map.get(eval.split()[-1], 0) for eval in evaluations]
        return sum(values) / len(values)

    @staticmethod
    def vote_prompt_wrap(x: dict, ys: list) -> str:
        choices_str = ''
        for idx, y in enumerate(ys, 1):
            choices_str += f"Choice {idx}:\n{y}\n\n"
        return vote_prompt.format(question=x['question'], context=x['context']) + choices_str

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            lines = vote_output.strip().split('\n')
            for line in lines:
                if 'The best answer is Choice' in line:
                    choice_num = int(line.strip().split('Choice')[-1])
                    if 1 <= choice_num <= n_candidates:
                        vote_results[choice_num - 1] += 1
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: dict, ys: list) -> str:
        assert len(ys) == 2, 'compare_prompt supports only two candidates'
        return compare_prompt.format(question=x['question'], context=x['context'], answer1=ys[0], answer2=ys[1])

    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'The better answer is Choice 1' in compare_output:
            return 0
        elif 'The better answer is Choice 2' in compare_output:
            return 1
        else:
            return -1  # Unable to determine

    @staticmethod
    def score_prompt_wrap(x: dict, y: str) -> str:
        return score_prompt.format(question=x['question'], context=x['context'], answer=y)

    @staticmethod
    def score_outputs_unwrap(score_outputs: list) -> float:
        scores = []
        for output in score_outputs:
            lines = output.strip().split('\n')
            for line in lines:
                if 'Assign a score from 1 to 10' in line:
                    continue
                if 'Score:' in line:
                    try:
                        score = int(line.strip().split('Score:')[-1])
                        scores.append(score)
                    except ValueError:
                        pass
        return sum(scores) / len(scores) if scores else 0