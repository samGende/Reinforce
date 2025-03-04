import torch
import numpy as np
import wandb
from utils.math import softmax_temp, pad_sequence
from memory_profiler import profile
from utils.loggin import log_memory_usage


class TReinforce:
    def __init__(self, env, policy, tokenizer, batch_size, mini_batch_size, optimizer, normed_advantages=True, logging=False, max_tokens=200, temp=1.5, n_shot = 4, kl_scale= 0.1, use_lora=False, adapter=None, run_num =0):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
        self.normed_advantages = normed_advantages
        self.prev_reward = 0
        self.previous_observation = None
        self.logging = logging
        self.max_tokens = max_tokens
        self.temp = temp
        self.n_shot = n_shot
        self.use_lora = use_lora
        self.adapter = adapter
        self.kl_scale = kl_scale
        if logging:
            config = {"model":policy.config._name_or_path, "batch_size": batch_size, "mini_batch_size": mini_batch_size, "normed_advantages": normed_advantages, "max_tokens": max_tokens, "temp": temp, "n_shot": n_shot, "use_lora": use_lora, 'kl_scale': kl_scale, "device": self.device.type}
            wandb.init(project='my-awesome-rl', entity='sam-gende-tu-dortmund', name='t_reinforce0', config = config)
            wandb.watch(self.policy)
            
    def collect_samples(self, n_shots=4):
        observations, actions, rewards, terms, probs_list = [], [], [], [], []
        
        self.env.reset()

        #loop through each shot
        for i in range(0, n_shots):
            print(f'shot {i}')
            obs = self.env.clear()
            # Move observation to device
            obs = obs.to(self.device)
            term = False
            while not term:
                with torch.no_grad():
                    outputs = self.policy(obs)
                    probs = softmax_temp(outputs.logits[0, -1], self.temp)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    action_tensor = action.unsqueeze(0).unsqueeze(0)
                    next_obs, reward = self.env.step(action_tensor)
                    # Move next observation to device
                    next_obs = next_obs.to(self.device)
                    term = (action == self.tokenizer.eos_token_id or self.env.n_tokens >= self.max_tokens)
                    
                    #if use lora enable kl penalty
                    if(self.use_lora):
                        #forward pass with base model
                        self.policy.disable_adapter_layers()
                        outputs_base = self.policy(obs)
                        probs_base = softmax_temp(outputs_base.logits[0, -1], self.temp)
                        dist_base = torch.distributions.Categorical(probs_base)
                        #log probs
                        base_log = dist_base.log_prob(action)
                        log = dist.log_prob(action)
                        #calc kl diff 
                        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                        kl_diff = log.exp()* (log - base_log)

                        #adjus reward by kl diff 
                        reward = reward - self.kl_scale *  kl_diff.item()

                        #enable adapter layers
                        self.policy.enable_adapter_layers()
                        if (self.logging):
                            wandb.log({"kl_diff": kl_diff.item()})

                    observations.append(obs)
                    actions.append(action)
                    rewards.append(reward)
                    terms.append(term)
                    probs_list.append(probs)

                    obs = next_obs
                    
        padded_observations = pad_sequence(observations)
        return padded_observations, torch.stack(probs_list), torch.tensor(actions, device=self.device), torch.tensor(rewards, device=self.device), torch.tensor(terms, device=self.device)

    def calc_advantages(self, rewards, terminated, gamma=0.9999):
        discounted_rewards, prev_reward = [], 0
        for i, reward in enumerate(reversed(rewards)):
            if terminated[len(terminated) - i - 1]:
                prev_reward = 0
            prev_reward = reward + (gamma * prev_reward)
            discounted_rewards.append(prev_reward)
        
        advantages = torch.tensor(discounted_rewards[::-1], dtype=torch.float, device=self.device)
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
            return torch.tensor(episode_rewards, dtype=torch.float, device=self.device).mean()
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
            obs, old_probs, actions, rewards, terms = self.collect_samples(self.n_shot)
            mean_reward = self.calc_mean_episode_reward(rewards, terms, mean_reward)
            print(f'mean batch reward: {mean_reward.item():.2f}')
            
            advantages = self.calc_advantages(rewards, terms)
            #TODO batch size should probably not be dependent on len of responses 
            batch_size = min(obs.shape[0], self.batch_size)
            batches = self.generate_batches(batch_size, self.mini_batch_size)
            losses = []

            print(obs.shape)
            print(old_probs.shape)
            for batch in batches:
                batch_indices = torch.tensor(batch, device=self.device)
                outputs = self.policy(obs[batch_indices])
                probs = softmax_temp(outputs.logits[:,-1, :], self.temp)
                dists = torch.distributions.Categorical(probs)
                log_probs = dists.log_prob(actions[batch_indices])
                old_dists = torch.distributions.Categorical(probs=old_probs[batch_indices].detach())
                old_log_probs = old_dists.log_prob(actions[batch_indices])
                r_t = log_probs.exp() / old_log_probs.exp()
                policy_clip = 0.2
                
                loss = -torch.min(r_t * advantages[batch_indices], torch.clamp(r_t, 1 - policy_clip, 1 + policy_clip) * advantages[batch_indices])
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                losses.append(loss.detach().mean().item())
            if(self.logging):
                wandb.log({"loss": np.mean(losses), "rewards": mean_reward.item()})
            
            steps += self.batch_size