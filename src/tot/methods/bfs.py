import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import tot.react.wikienv as wikienv
import tot.react.wrappers as wrappers


#----------------------------------------------------------------------------------
# REACT search Functions
#----------------------------------------------------------------------------------

# Attempts to execute an action within the environment
def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1




#----------------------------------------------------------------------------------
# Utility Functions
#----------------------------------------------------------------------------------


# retrieves a score (or “value”) for a specific solution candidate
def get_value(task, x, y, n_evaluate_sample, cache_value=True):

    # task.value_prompt_wrap is a method that takes x (the input) and y (the partial solution) and wraps them into a format suitable for querying gpt.
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    
    # calls gpt with value_prompt to generate n_evaluate_sample responses.
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(value_outputs)  # extract a single score or “value.”
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


# builds on get_value to handle multiple candidates in one go.
def get_values(task, x, under_evaluation_y, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}

    # Loop Over Candidates
    for y in under_evaluation_y:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


# another method for evaluating solution candidates, but this time using a voting mechanism.
def get_votes(task, x, ys, n_evaluate_sample):

    # Create a Voting Prompt
    # The method task.vote_prompt_wrap combines x (the input) and ys (the list of candidates) 
    # to create a prompt that asks GPT to score or “vote” on each candidate’s quality.
    vote_prompt = task.vote_prompt_wrap(x, ys)

    # Calls gpt with the vote_prompt to generate n_evaluate_sample responses.
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)

    # Processes vote_outputs to extract individual scores for each candidate in ys.
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


# get_proposals: Generates structured proposals for extending a solution, potentially providing more directed suggestions for the next steps.
# Unlike get_samples, which creates more exploratory variations, 
# get_proposals suggests more specific paths or strategies for continuation.
def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')


    # Filter out empty entries and create pairs of (Thought, Action)
    thought_action_pairs = []
    temp_pair = []
    
    for line in proposals:
        if line:  # Ignore empty lines
            temp_pair.append(line)
            # Once a pair (Thought, Action) is complete, add it to the list
            if len(temp_pair) == 2:
                thought_action_pairs.append(tuple(temp_pair))
                temp_pair = []  # Reset for the next pair

    # Limit to 4 pairs and format each as "Thought X: ... Action X: ..."
    formatted_pairs = [f"{pair[0]}\n{pair[1]}" for pair in thought_action_pairs[:4]]
    
    # Print for debugging and return
    print(formatted_pairs)
    return formatted_pairs


    #return [y + _ + '\n' for _ in proposals]



# get_samples: Expands a solution by sampling several different continuations, exploring multiple potential paths.
def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    
    # Calls GPT to generate n_generate_sample completions based on the prompt.
    # Each generated sample is a continuation of the partial solution (x, y).
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


#----------------------------------------------------------------------------------
# SOLVE Functions
#----------------------------------------------------------------------------------

def solve(args, task, idx, to_print=True):

    # SETUP
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)

    # Setup input as well as the current solution candidates
    x = task.get_input(idx)  # input
    current_prompt = x
    ys = ['']  # current output candidates
    infos = []

    # Loops through a series of steps to iteratively refine and select candidates.
    for curr_step in range(task.steps):

        under_evaluation_ys = []

        # GENERATION
        if args.method_generate == 'sample':

            # calls get_samples to expand each candidate y in ys.
            new_ys = [get_samples(task, 
                                  current_prompt, 
                                  y, 
                                  args.n_generate_sample, 
                                  prompt_sample=args.prompt_sample,         # Chooses either cot or standard
                                  stop=task.stops[curr_step]) for y in ys]
            
        elif args.method_generate == 'propose':

            

            for y in ys:

                new_ys = [get_proposals(task, 
                                        current_prompt, 
                                        y)]
                
                new_ys = list(itertools.chain(*new_ys))
                

                if args.react_search == True:
                    
                    # Loop through all the thought-action pairs
                    for i, pair in enumerate(new_ys, start=1):
                        # Separate thought and action using newline
                        parts = pair.split('\n')
                        thought = parts[0].replace(f"Thought {i}: ", "").lower()
                        action = parts[1].replace(f"Action {i}: ", "").lower()
                        
                        # Get environment and perform the action
                        env = task.get_env()
                        obs, r, done, react_infos = step(env, action[0].lower() + action[1:])  # Perform the step in the environment with the action
                        obs = obs.replace('\\n', '')  # Clean up observation output

                        # Construct step string for each thought-action-observation sequence
                        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
                        print(step_str)  # Optionally print to see the constructed prompt string

                        # basically append this step_str to the 

                        under_evaluation_y = y + "\n" + step_str

                        under_evaluation_ys.append(under_evaluation_y)
                    

            # Id over the range of under evaluation y things
            ids = list(range(len(under_evaluation_ys)))




        # EVALUATE
        if args.method_evaluate == 'vote':
            # uses get_votes to score the candidates.
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)

        elif args.method_evaluate == 'value':
            # uses get_values to retrieve scores.
            values = get_values(task, x, under_evaluation_ys, args.n_evaluate_sample)

        # SELECTION
        if args.method_select == 'sample':
            # samples candidate indices (select_ids) based on the probabilities derived from values
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()


        elif args.method_select == 'greedy':
            # picks the top candidates by sorting values in descending order.
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]


        # Creates the list of selected candidates by indexing new_ys with select_ids.
        select_new_ys = [under_evaluation_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'curr_step': curr_step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        
        # Updates ys to the selected candidates, continuing with these in the next step.
        ys = select_new_ys

    return ys, {'steps': infos}
    




def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}