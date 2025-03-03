import gymnasium as gym
import torch
from gsm8k import GSM8K_Env
from transformers import AutoModelForCausalLM, AutoTokenizer
from t_reinforce import TReinforce
from peft import LoraConfig, TaskType, get_peft_model
import argparse

# importing the library
import faulthandler
faulthandler.enable()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model with lora or the whole model')
    parser.add_argument('--lora', type=bool, help='The model name to train')
    args = parser.parse_args()
    lora = args.lora


    model_name = 'meta-llama/Llama-3.2-1B'  # Replace with your desired model variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if lora:
        print('starting training with lora')
        #fix no grad issue
        model.enable_input_require_grads()
        lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.0
    )
        adapter= "r++"
        model = get_peft_model(model, lora_config, adapter)


    max_tokens =200 
    env = GSM8K_Env(tokenizer=tokenizer, max_tokens=max_tokens)
    env.reset()
    optimizer = torch.optim.Adam(model.parameters(), lr =0.1)



    reinforce = TReinforce(env, model, tokenizer, 800, 32, optimizer, True, logging=True, max_tokens=max_tokens, use_lora=lora, adapter=adapter, run_num=0)

    reinforce.train(10000)

    model.save_predtrained('first_llama')
    
