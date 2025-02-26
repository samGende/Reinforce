import gymnasium as gym
import torch
from gsm8k import GSM8K_Env
from transformers import AutoModelForCausalLM, AutoTokenizer
from t_reinforce import TReinforce
# importing the library
import faulthandler
faulthandler.enable()

if __name__ == '__main__':


    model_name = 'meta-llama/Llama-3.2-1B'  # Replace with your desired model variant
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_tokens = 5
    env = GSM8K_Env(tokenizer=tokenizer, max_tokens=max_tokens)
    env.reset()
    model = AutoModelForCausalLM.from_pretrained(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr =0.0001)


    reinforce = TReinforce(env, model, tokenizer, 4, optimizer, True, max_tokens=5)

    reinforce.train(1000)

    model.save_predtrained('first_llama')