import gymnasium as gym
import torch
from BaseRL.model import Policy
from t_reinforce import TReinforce
import wandb

#hyperparams 
lr = 0.001
env_name = "CartPole-v1"
batch_size = 256

logging = True

env = gym.make(env_name)
observation, info = env.reset()

policy = Policy(4, 2)
optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

if(logging):
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-RL",

        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "env": env_name,
        "batch_size": batch_size,
        }
    )

t_reinforce = TReinforce(env, policy, batch_size=batch_size, optimizer=optimizer, logging=logging)
t_reinforce.train(500000)

# evaluate

env = gym.make(env_name, render_mode = "human")
for i in range(5):
    observation, info = env.reset()
    done = False
    rewards = []
    while not done:
        probs = policy(torch.tensor(observation, dtype=torch.float))
        action = torch.distributions.Categorical(probs=probs).sample().item()
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        done = done or truncated
    print("Reward: ", sum(rewards))

