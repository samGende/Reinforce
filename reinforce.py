import torch

#collect samples 
def collect_samples(env, policy, n_samples = 64):
    obs = []
    actions = []
    rewards = []
    terms = []
    observation, info = env.reset()
    observation = torch.tensor(observation, dtype=torch.float)

    episode_over = False
    while not episode_over:
        # reset the environment
        # get action from policy
        probs = policy(observation)
        action = torch.distributions.Categorical(probs=probs).sample().item() 
        # take action
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = torch.tensor(next_observation, dtype=torch.float)
        # check if episode is over
        episode_over = terminated or truncated
        # store the sample
        obs.append(observation)
        actions.append(action)
        rewards.append(reward)
        terms.append(episode_over)
        observation = next_observation
        
    return obs, actions, rewards, terms


#callculte rewards 
def calculate_advantages(rewards, terminated,  gamma = 0.9999):
    discounted_rewards = []
    prev_reward = 0
    for i, reward in enumerate(reversed(rewards)):
        if terminated[len(terminated)- i - 1]:
            prev_reward = 0
        prev_reward = reward + (gamma * prev_reward)
        discounted_rewards.append(prev_reward)

    advantages = discounted_rewards[::-1]
    #normed_advantages = (advantages) / (advantages.std() + 1e-8)
    return advantages

def calc_discount_rewards(rewards):
    gamma = .9999
    DiscountedReturns = []
    for t in range(len(rewards)):
        G = 0.0
        for k, r in enumerate(rewards[t:]):
            G += (gamma**k)*r
        DiscountedReturns.append(G)
    return DiscountedReturns

def calc_mean_episode_reward(rewards, terms):
    episode_rewards = []
    episode_reward = 0
    for i, reward in enumerate(rewards):
        episode_reward += reward
        if terms[i]:
            episode_rewards.append(episode_reward)
            episode_reward = 0
    return torch.tensor(episode_rewards, dtype=torch.float).mean()

def train(env, policy, batch_size, n_steps, optimizer):
    steps = 0
    while steps < n_steps:
        obs, actions, rewards, terms = collect_samples(env, policy, batch_size)
        print("mean episode reward: ", calc_mean_episode_reward(rewards, terms).item())
        #just as broken using calc_discount_rewards
        advantages = calculate_advantages(rewards, terms)

        # train the policy
        for state, action, discounted_reward in zip(obs, actions, advantages):
            # get probs from model
            probs = policy(state)
            dist = torch.distributions.Categorical(probs=probs)
            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.int))
            # calc loss from probs
            loss = -log_prob*discounted_reward
            optimizer.zero_grad()
            loss.backward()
            # update model
            optimizer.step()
        steps += batch_size
