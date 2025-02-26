import faulthandler
import sys
from utils.strings import extract_answer
import json
import random
faulthandler.enable(file=sys.stderr, all_threads=True)
from datasets import load_dataset
import torch




class GSM8K_evaluation:
    def __init__(self):
        ds = load_dataset("openai/gsm8k", "main")
        self.test = ds['test']
        random.seed(10)

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


class GSM8K_Env:
    def __init__(self, tokenizer, max_tokens):
        ds = load_dataset("openai/gsm8k", "main", streaming=True)
        self.train = ds['train'].shuffle(seed=42)
        self.train_iter = iter(self.train)
        self.tokenizer = tokenizer
        #current index
        self.cur = None
        self.state = -1
        self.n_tokens = 0
        self.max_tokens = max_tokens
    '''
    reset the envs state and move to new prompt
    ''' 
    def reset(self):
        self.cur = next(self.train_iter)
        return self.clear()
    '''
    clear the envs state which removes any model generations
    '''
    def clear(self):
        self.n_tokens = 0
        inputs = self.tokenizer(self.cur['question'], return_tensors='pt')
        if 'input_ids' in inputs:
            self.state = inputs['input_ids']
            return self.state
        else:
            return -1

    
    def step(self, next_token):
        self.n_tokens += 1
        self.state = torch.cat((self.state, next_token), dim=-1)
        reward = self.get_reward()
        #return next prompt in training data
        return self.state, reward
    
    def render(self):
        print(self.tokenizer.decode(self.state[0]))

    def get_state(self):
        return self.tokenizer.decode(self.state[0])

    def get_reward(self):
        if len(self.state) == self.max_tokens and self.state[0,-1] != self.tokenizer.eos_token_id:
            return -1    
        completion = self.tokenizer.decode(self.state[0])
        #calc reward for completion 
        proposed_answer = extract_answer(completion)
        #extract correct anwer
        correct_answer = extract_answer(self.cur['answer'])
        if(correct_answer == proposed_answer):
            return 1
        else:
            return 0.1
