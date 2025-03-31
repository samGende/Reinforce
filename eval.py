from gsm8k import GSM8K_evaluation
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from model_wrapper import TokenWrapper
import argparse


# gather args
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--formatted', type=bool, default=True)
args = parser.parse_args()  

model_name = args.model_name
formatted = args.formatted

evaluator = GSM8K_evaluation(formatted=formatted)

# load model 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model_wrapped = TokenWrapper(model, tokenizer)

output_file = model_name.split("/")[-1] + "_results.json"

deepseek_prompt = 'Please reason step by step, and put your final answer within \boxed{}.'

# evaluate model
acc, corrects = evaluator.eval(model_wrapped, n_evals=1, output_file=output_file, format_instructions=deepseek_prompt)
