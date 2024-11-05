import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
from tot.tasks.text import TextTask
from tot.tasks.good_hotpotqa import hotpotQATask


import openai
print(openai.__version__)

args = argparse.Namespace(
    backend='gpt-4', 
    temperature=0.7, 
    task='hotpotqa', 
    naive_run=False, 
    prompt_sample=None, 
    method_generate='propose', 
    method_evaluate='value', 
    method_select='greedy', 
    react_search=True,
    n_generate_sample=1, 
    n_evaluate_sample=3, 
    n_select_sample=2)

#args = argparse.Namespace(
#    backend='gpt-4', 
#    temperature=0.7, 
#    task='text', 
#    naive_run=False, 
#    prompt_sample='cot', 
#    method_generate='sample', 
#    method_evaluate='vote', 
#    method_select='greedy', 
#    n_generate_sample=4, 
#    n_evaluate_sample=3, 
#    n_select_sample=5)

task = hotpotQATask()
ys, infos = solve(args, task, 1)
print(ys[0])