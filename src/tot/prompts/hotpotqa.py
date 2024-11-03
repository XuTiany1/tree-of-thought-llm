









#------------------------
# Generative prompts
#------------------------
propose_prompt = '''
Solve the following question by generating a possible reasoning steps
For each path, provide a Thought and an associated Action.

Thought: A reasoning step that considers how to approach the question.
Action: One of the following:
  - Search[entity]: Searches for the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return similar entities to search.
  - Lookup[keyword]: Returns the next sentence containing the keyword in the current passage.
  - Finish[answer]: Returns the answer and finishes the task.

Example Structure:



Question: Which planet has a longer day length, Venus, Mars, or Jupiter?

Provide 3 possible Thought and Action pairs as next steps:
Thought 1: I need to find out the length of a day on Venus.
Action 1: Search[Day length of Venus]

Thought 2: I need to find out the length of a day on Mars.
Action 2: Search[Day length of Mars]

Thought 3: I need to find out the length of a day on Jupiter.
Action 3: Search[Day length of Jupiter]


Question: {question}
Provide {number_of_thoughts} Thought and Action pairs as next steps:
'''


















#------------------------
# Evaluation prompts
#------------------------

