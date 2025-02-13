import gymnasium as gym
import torch
from reinforce import collect_samples, calculate_advantages, train, generate_batches
from model import Policy

env = gym.make("CartPole-v0")
observation, info = env.reset()

policy = Policy(4, 2)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

train(env, policy, 256, 500000, optimizer)


# evaluate

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

