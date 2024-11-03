import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()

# self.data: 
#   List of puzzles loaded from the CSV file.
# self.value_cache: 
#   Dictionary for caching computed values to avoid redundant calculations.
# self.steps: 
#   Set to 4, indicating a maximum of 4 steps to solve each puzzle.
# self.stops: 
#   A list of stop conditions, here each being a newline character ('\n') for separating steps.
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

# Solution Testing
    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}



#-----------------------------------------------
# Prompt Wrapping Functions
#-----------------------------------------------


#-----------------------------------------------
# Prompt Wrapping Functions: Section1 -> Generation step
#-----------------------------------------------
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        # Formats the standard_prompt by inserting the input puzzle 'x' 
        # and appending any partial solution 'y'
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        # Similar to standard_prompt_wrap but uses cot_prompt for generating CoT-style prompts.
        # Encourages the model to first make a plan before providing a solution.
        return cot_prompt.format(input=x) + y
    

#-----------------------------------------------
# Prompt Wrapping Functions: Section2 -> Proposal step
#-----------------------------------------------
    @staticmethod
    # Used in the get_proposals function.
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)

        # If the current numbers are not yet reduced to 24, it formats the propose_prompt with the remaining numbers.
        if current_numbers == '24':

            # The prompt uses cot_prompt to produce a final output, marking the solution as complete.
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:

            # The prompt uses propose_prompt with current_numbers to generate potential moves that can bring the result closer to 24.
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
#-----------------------------------------------
# Prompt Wrapping Functions: Section3 -> Evaluation step
#-----------------------------------------------
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    

# Output Scoring
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value