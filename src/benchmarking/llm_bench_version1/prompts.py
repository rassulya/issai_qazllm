def hellaswag_prompt(row):
    prompt = f"""Please complete the sentence given below: '{row['ctx']}'...
Choose the correct ending from the options listed below:
1: '{row['option1']}'
2: '{row['option2']}'
3: '{row['option3']}'
4: '{row['option4']}'
Return your answer as a single word, in the following format: X, where X is the number indicating the correct option. Do not explain your answer.
Answer: """
    return prompt

def arc_prompt(row):
    prompt = f"""The following question is presented for you to answer:
Question: {row['question']}
Choose the correct answer from the options listed below:
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
Return your answer as a single word, in the following format: X, where X is the letter indicating the correct option. Do not explain your answer.
Answer: """
    return prompt

def gsm8k_prompt(row):
    prompt = f"""Solve the following mathematical problem: {row['question']}
Return your answer as a single word, in the following format: X, where X is the exact numerical answer. Do not include any explanation, just the final answer.
Answer: """
    return prompt

def truthfulqa_prompt(row):
    prompt = f"""You are presented with a question and a list of possible answers. Your task is to identify the correct answer.
Question: {row['question']}
Choices:
{row['choices']}
Return your answer as a single word, in the following format: X, where X is the number corresponding to the correct choice (e.g., 1 for the first choice, 2 for the second, etc.). Do not include any explanations—simply provide the number.
Answer: """
    return prompt

def winogrande_prompt(row):
    prompt = f"""You are given a sentence with one missing word, indicated by an underscore. Read the sentence and choose the appropriate option from the two provided to fill in the blank.
Sentence: {row['sentence']}
Option 1: {row['option1']}
Option 2: {row['option2']}
Return your answer as a single word, in the following format: X, where X is the number (1 or 2) corresponding to the correct option. Do not include any explanation.
Answer: """
    return prompt

def mmlu_prompt(row):
    prompt = f"""The following question is presented for you to answer:
Question: {row['question']}
Choose the correct answer from the options listed below:
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
Return your answer as a single word, in the following format: X, where X is the letter indicating the correct option. Do not explain your answer.
Answer: """
    return prompt