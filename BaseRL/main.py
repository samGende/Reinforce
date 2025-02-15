import gymnasium as gym
import torch
from reinforce import  Reinforce
from model import Policy
env = gym.make("CartPole-v0")
observation, info = env.reset()

policy = Policy(4, 2)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

reinforce = Reinforce(env, policy, 256,  optimizer)

reinforce.train(5000)


# evaluate

env = gym.make("CartPole-v0", render_mode = "human")
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

