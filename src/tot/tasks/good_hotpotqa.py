import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.hotpotqa import * 
from tot.models import gpt
import tot.react.wikienv as wikienv
import tot.react.wrappers as wrappers



class hotpotQATask(Task):


    def __init__(self):

        super().__init__()

        # LOAD reACT prompting examples
        #folder = '../prompts'
        #prompt_file = '/prompts_naive.json'
        #with open(folder + prompt_file, 'r') as f:
        #    prompt_dict = json.load(f)

        #hardcode 
        hard_code_name = 'src/tot/prompts/prompts_naive.json'
        with open(hard_code_name, 'r') as f:
            prompt_dict = json.load(f)
        self.webthink_prompt = prompt_dict['webthink_simple6']


        # INIT ENV variables for both both wiki as well as input/output 
        env = wikienv.WikiEnv()
        env = wrappers.HotPotQAWrapper(env, split="dev")        # Wraps env with HotPotQAWrappe
        env = wrappers.LoggingWrapper(env)     
        self.env = env  # Store the environment as an attribute for further use

        
        # Max depth of the tree
        self.steps = 5

        # Cache to make sure no duplicates
        self.value_cache = {}

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        question = self.env.reset(idx=idx)

        print(idx, question)
        #prompt += question + "\n"
        return question
    
    def get_env(self):
        return self.env

    def test_output(self, idx: int, output: str):
        pass


    

# -------------------
# Propose prompt generation
# -------------------

    # x is question
    # y is partial solution
    @staticmethod
    def propose_prompt_wrap(x: str, y: str = '') -> str:
        prompt = propose_prompt.format(
            examples=react_example_prompt,
            question=x,
            context=y
        )
        #print(prompt)

        return prompt



# -------------------
# Evaluate prompt generation
# -------------------

    # x is question
    # y is the current reasoning path
    @staticmethod
    def value_prompt_wrap(x:str, y: str) -> str:
        prompt = value_prompt.format(
            question = x,
            reasoning_path=y
        )
        print(prompt)

        return prompt
    

    # value_outputs is a list of the judgements given from gpt from previous evaluation on the values
    @staticmethod
    def value_outputs_unwrap(value_outputs: list) -> float:
        value_map = {'low': 0.1, 'medium': 0.5, 'high': 1.0}
        values = []
        
        for output in value_outputs:
            # Take the first word (converted to lowercase) as the rating
            first_word = output.strip().split('\n')[0].lower()
            values.append(value_map.get(first_word, 0))
        
        return sum(values) / len(values) if values else 0





    def main(self):

        question = self.get_input(1)

        # Testing propose_prompt
        question_prompt = self.propose_prompt_wrap(self.webthink_prompt, question, '')
        # print(question_prompt)
        # question_samples = gpt(question_prompt, n=1, stop=None)
        # print(question_samples)


        # Testing value_prompt
        reason_question = "Question: Geoff LaTulippe is an American writer whose best-known work was directed by whom?"
        reasoning_path = """
            Thought 1: I need to search Geoff LaTulippe, find his best-known work, then find who directed it.
            Action 1: Search[Geoff LaTulippe]
            Observation 1: Geoff LaTulippe is an American screenwriter and film director best known as the writer of the 2010 film Going the Distance.. LaTulippe was born in Cleveland, Ohio. His father's job as a casket salesman caused his family to move across the United States between numerous cities before they settled in Harrisburg, Pennsylvania.[1] LaTulippe attended James Madison University in Virginia, where he studied film and writing.[1]. Shortly after moving to Los Angeles, California, LaTulippe was hired as a script reader at New Line Cinema, a job he was offered by his friend who worked at the studio. After working there for two years, he tired of the job, saying that it "start[ed] to suck away [my] creativity".[1] In July 2008, he sold a spec script titled Going the Distance to New Line Cinema, a story based on his friend David Neustadter's previous long-term relationship.[1] The film was ultimately directed by Nanette Burstein and released in September 2010.

            Thought 2: Geoff LaTulippe's best-known work is Going the Distance. I need to find who directed it.
            Action 2: Lookup[directed by]
            Observation 2: (Result 1 / 1) After working there for two years, he tired of the job, saying that it "start[ed] to suck away [my] creativity".[1] In July 2008, he sold a spec script titled Going the Distance to New Line Cinema, a story based on his friend David Neustadter's previous long-term relationship.[1] The film was ultimately directed by Nanette Burstein and released in September 2010.
            """     
        value_prompt = self.value_prompt_wrap(reason_question, reasoning_path)
        value_samples = gpt(value_prompt, n=3, stop=None)
        print(value_samples)

        # Testing value_outputs_unwrap

        output_sample_score = self.value_outputs_unwrap(value_samples)
        print("================================")
        print(f"output score is {output_sample_score}")
        print("================================")




# To test/debug, instantiate the class and call the main method
if __name__ == "__main__":
    task = hotpotQATask()
    task.main()










































































































