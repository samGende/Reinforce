from gsm8k import GSM8K_evaluation
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from model_wrapper import TokenWrapper

evaluator = GSM8K_evaluation()

# load model 

model_name = 'meta-llama/Llama-3.2-1B-Instruct'  # Replace with your desired model variant
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model_wrapped = TokenWrapper(model, tokenizer)
# evaluate model
acc, corrects = evaluator.eval(model_wrapped, n_evals=100, output_file="instruct_model_results.json")
