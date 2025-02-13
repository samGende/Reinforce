import faulthandler
import sys
from utils.strings import extract_answer
import json
faulthandler.enable(file=sys.stderr, all_threads=True)

from datasets import load_dataset

print(f'faulhandler status {faulthandler.is_enabled()}')
ds = load_dataset("openai/gsm8k", "main")
test = ds['test']

class GSM8K_evaluation:
    def __init__(self):
        ds = load_dataset("openai/gsm8k", "main")
        self.test = ds['test']

    def eval(self, model, n_evals=3, output_file="evaluation_results.json"):
        results = []
        correct_answers = []
        
        for i, row in enumerate(self.test):
            if i >= n_evals:
                break
            
            proposed_answer = model(row['question'])
            correct_answer = extract_answer(row['answer'])
            extracted_answer = extract_answer(proposed_answer)
            
            is_correct = correct_answer == extracted_answer
            correct_answers.append(is_correct)
            
            results.append({
                "question": row['question'],
                "model_answer": proposed_answer,
                "correct_answer": correct_answer,
                "extracted_model_answer": extracted_answer,
                "is_correct": is_correct
            })
            
            print(f'finished row {i}')
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        return sum(correct_answers) / n_evals, correct_answers
