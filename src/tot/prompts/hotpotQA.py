

#--------------------------------------------------
# STAGE 1 prompt: Generation
#--------------------------------------------------

# Standard Prompt
standard_prompt = '''Use Wikipedia search to answer the following question:

Question: {question}

Action Options:
- **Search[entity]**: Looks up the specified entity in Wikipedia.
- **Finish[answer]**: Provides the final answer if found.

Example:
Question: Which planet has the longest day length?
Action: Search[Day length of each planet]

Answer: Venus has the longest day length among the planets.
Question: {question}
Action:
'''



# Chain-of-Thought Prompt
cot_prompt = '''
Answer the following question by reasoning through each step. 
Use Wikipedia search for each specific term as needed.

Question: {question}

Action Options:
- **Search[entity]**: Looks up the specified entity in Wikipedia.
- **Lookup[keyword]**: Finds information on the specified keyword.
- **Finish[answer]**: Concludes with the answer when you have sufficient information.

Example:
Question: Who wrote the song about "The Simpsons" character Milhouse, and who was the character named after?
Step 1:
- Thought: I need to find who wrote the song about Milhouse.
- Action: Search[Milhouse song writer]
Step 2:
- Thought: I found Allie Goertz wrote a song about Milhouse. Now I need to find who Milhouse was named after.
- Action: Search[Milhouse named after]
Answer: Milhouse was named after Richard Nixon.

Question: {question}
Step 1:
'''


# Propose Prompt (This is what i will use for Tree of thought)
propose_prompt = '''
Solve the following question by generating {number_of_thoughts} possible reasoning paths. 
For each path, provide a Thought and an associated Action.

Thought: A reasoning step that considers how to approach the question.
Action: One of the following:
  - Search[entity]: Searches for the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return similar entities to search.
  - Lookup[keyword]: Returns the next sentence containing the keyword in the current passage.
  - Finish[answer]: Returns the answer and finishes the task.

Example Structure:

Question: Which planet has a longer day length, Venus, Mars, or Jupiter?

Thought 1: I need to find out the length of a day on Venus.
Action 1: Search[Day length of Venus]

Thought 2: I need to find out the length of a day on Mars.
Action 2: Search[Day length of Mars]

Thought 3: I need to find out the length of a day on Jupiter.
Action 3: Search[Day length of Jupiter]

Now, answer the following question:

Question: {question}

Provide {number_of_thoughts} Thought and Action pairs:
'''


#--------------------------------------------------
# STAGE 2 prompt: Evaluation
#--------------------------------------------------

# Value Prompt
value_prompt = '''
Evaluate the likelihood that the following reasoning path will lead to the correct answer (high/medium/low). 
Provide a brief justification for this evaluation.

Question: {question}

Context:
{context}

Reasoning Path:
Thought: {thought}
Action: {action}

Evaluation:
'''


value_last_step_prompt = '''
Verify the final answer to the following question based on the provided evidence. 
Determine if the answer is correct and justified.

Question: {question}

Evidence Gathered:
{evidence}

Proposed Answer: {answer}

Judgment (justified/correct or unjustified/incorrect):
'''


