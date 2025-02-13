from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from gsm8k import GSM8K_evaluation
from model_wrapper import TokenWrapper

model_name = 'meta-llama/Llama-3.2-1B'  # Replace with your desired model variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("model loaded")

evaluator = GSM8K_evaluation()

wrapped_model = TokenWrapper(model, tokenizer)

print(wrapped_model("Hello how are you?"))

percentage, results = evaluator.eval(wrapped_model,n_evals=2000, output_file="base_model_results.json")
print("list of results")
print(results)
print(f'score on gsm8k {percentage}')
