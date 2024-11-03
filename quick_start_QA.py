# run_hotpotqa_test.py

import argparse
from tot.methods.bfs import solve
from tot.tasks.not_good import HotpotQATask

args = argparse.Namespace(
    backend='gpt-4',
    temperature=0.7,
    task='hotpotqa',
    naive_run=False,
    prompt_sample='cot',  # Use chain-of-thought prompts
    method_generate='sample',
    method_evaluate='value',
    method_select='greedy',
    n_generate_sample=3,
    n_evaluate_sample=1,
    n_select_sample=1
)

task = HotpotQATask()

# Test with the first sample
ys, infos = solve(args, task, idx=0)
print("Test Case 1 Output:")
print(ys[0])

# Test with the second sample
ys, infos = solve(args, task, idx=1)
print("Test Case 2 Output:")
print(ys[0])