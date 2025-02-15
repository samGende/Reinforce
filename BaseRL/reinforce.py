import torch
import numpy as np
import wandb

class Reinforce:
    def __init__(self, env, policy, batch_size, optimizer, normed_advantages=True, logging=False):
        self.env = env
        self.policy = policy
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.normed_advantages = normed_advantages
        self.prev_reward = 0
        self.previous_observation = None
        self.logging = logging

    def collect_samples(self, n_samples=64):
        obs, actions, rewards, terms, probs_list = [], [], [], [], []
        observation = self.previous_observation if self.previous_observation is not None else self.env.reset()[0]
        observation = torch.tensor(observation, dtype=torch.float)
        samples = 0

        while samples < n_samples:
            episode_over = False
            while not episode_over:
                probs = self.policy(observation)
                action = torch.distributions.Categorical(probs=probs).sample().item()
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                next_observation = torch.tensor(next_observation, dtype=torch.float)
                episode_over = terminated or truncated

                obs.append(observation)
                probs_list.append(probs)
                actions.append(action)
                rewards.append(reward)
                terms.append(episode_over)

                observation = next_observation
                samples += 1
                
                if episode_over:
                    observation = torch.tensor(self.env.reset()[0], dtype=torch.float)
                if samples >= n_samples:
                    break

        self.previous_observation = observation
        return torch.stack(obs), torch.stack(probs_list), torch.tensor(actions), torch.tensor(rewards), np.array(terms)

    def calc_advantages(self, rewards, terminated, gamma=0.9999):
        discounted_rewards, prev_reward = [], 0
        for i, reward in enumerate(reversed(rewards)):
            if terminated[len(terminated) - i - 1]:
                prev_reward = 0
            prev_reward = reward + (gamma * prev_reward)
            discounted_rewards.append(prev_reward)
        
        advantages = torch.tensor(discounted_rewards[::-1], dtype=torch.float)
        if self.normed_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def calc_mean_episode_reward(self, rewards, terms, prev_mean_reward):
        episode_rewards, episode_reward = [], self.prev_reward
        for i, reward in enumerate(rewards):
            episode_reward += reward
            if terms[i]:
                episode_rewards.append(episode_reward)
                episode_reward = 0
        self.prev_reward = episode_reward
        if episode_rewards:
            return torch.tensor(episode_rewards, dtype=torch.float).mean()
        else:
            return prev_mean_reward

    def generate_batches(self, batch_size, mini_batch_size):
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        return [indices[i:i + mini_batch_size] for i in range(0, batch_size, mini_batch_size)]

    def train(self, n_steps):
        steps = 0
        mean_reward = 0
        while steps < n_steps:
            obs, old_probs, actions, rewards, terms = self.collect_samples(self.batch_size)
            mean_reward = self.calc_mean_episode_reward(rewards, terms, mean_reward)
            print(f'mean batch reward: {mean_reward.item():.2f}')
            
            advantages = self.calc_advantages(rewards, terms)
            batches = self.generate_batches(self.batch_size, 64)
            losses = []

            for batch in batches:
                probs = self.policy(obs[batch])
                dists = torch.distributions.Categorical(probs=probs)
                log_probs = dists.log_prob(actions[batch])
                old_dists = torch.distributions.Categorical(probs=old_probs[batch].detach())
                old_log_probs = old_dists.log_prob(actions[batch])
                r_t = log_probs.exp() / old_log_probs.exp()
                policy_clip = 0.2
                
                loss = -torch.min(r_t * advantages[batch], torch.clamp(r_t, 1 - policy_clip, 1 + policy_clip) * advantages[batch])
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                losses.append(loss.detach().mean())
            if(self.logging):
                wandb.log({"loss": np.mean(losses), "rewards": mean_reward.item()})
            
            steps += self.batch_size
