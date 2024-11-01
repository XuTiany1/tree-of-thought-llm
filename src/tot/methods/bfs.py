import itertools
import numpy as np
from functools import partial
from tot.models import gpt


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
    value = task.value_outputs_unwrap(x, y, value_outputs)  # extract a single score or “value.”
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


# builds on get_value to handle multiple candidates in one go.
def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}

    # Loop Over Candidates
    for y in ys:  # each partial output
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
    return [y + _ + '\n' for _ in proposals]



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
    ys = ['']  # current output candidates
    infos = []

    # Loops through a series of steps to iteratively refine and select candidates.
    for step in range(task.steps):

        # GENERATION
        if args.method_generate == 'sample':
            # calls get_samples to expand each candidate y in ys.
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            # calls get_proposals for each candidate.
            new_ys = [get_proposals(task, x, y) for y in ys]
        # Flattens the list of generated candidates (new_ys) into a single list for evaluation.
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))


        # EVALUATE
        if args.method_evaluate == 'vote':
            # uses get_votes to score the candidates.
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            # uses get_values to retrieve scores.
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # SELECTION
        if args.method_select == 'sample':
            # samples candidate indices (select_ids) based on the probabilities derived from values
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            # picks the top candidates by sorting values in descending order.
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        # Creates the list of selected candidates by indexing new_ys with select_ids.
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        
        # Updates ys to the selected candidates, continuing with these in the next step.
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}