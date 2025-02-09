import torch

#collect samples 
def collect_samples(env, policy, n_samples = 64, previous_observation = None):
    obs = []
    actions = []
    rewards = []
    terms = []
    if (previous_observation != None):
        observation = previous_observation
    else:
        observation, info = env.reset()
    observation = torch.tensor(observation, dtype=torch.float)
    samples = 0

    while samples< n_samples:
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
            samples += 1
            # end of episode reset env
            if episode_over:
                observation, info = env.reset()
                observation = torch.tensor(observation, dtype=torch.float)
            if samples >= n_samples:
                break
    return obs, actions, rewards, terms


#callculte rewards 
def calculate_advantages(rewards, terminated,  gamma = 0.9999, normed = True):
    discounted_rewards = []
    prev_reward = 0
    for i, reward in enumerate(reversed(rewards)):
        if terminated[len(terminated)- i - 1]:
            prev_reward = 0
        prev_reward = reward + (gamma * prev_reward)
        discounted_rewards.append(prev_reward)

    advantages = discounted_rewards[::-1]
    advantages = torch.tensor(advantages, dtype=torch.float)
    if normed:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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

def calc_mean_episode_reward(rewards, terms, prev_reward = 0):
    episode_rewards = []
    episode_reward = prev_reward
    for i, reward in enumerate(rewards):
        episode_reward += reward
        if terms[i]:
            episode_rewards.append(episode_reward)
            episode_reward = 0
    #used for next batch
    prev_reward = episode_reward

    return torch.tensor(episode_rewards, dtype=torch.float).mean(), prev_reward

def train(env, policy, batch_size, n_steps, optimizer, normed_advatages = True):
    steps = 0
    obs = None
    prev_reward = 0
    while steps < n_steps:
        if obs:
            previous_observation = obs[-1]
        else:
            previous_observation = None
        obs, actions, rewards, terms = collect_samples(env, policy, batch_size, previous_observation)
        print(len(obs))
        mean_reward, prev_reward = calc_mean_episode_reward(rewards, terms, prev_reward)
        print('mean reward: ', mean_reward)

        #just as broken using calc_discount_rewards
        advantages = calculate_advantages(rewards, terms, normed_advatages)

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
