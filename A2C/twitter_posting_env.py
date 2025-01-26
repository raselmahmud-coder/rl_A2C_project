# twitter_posting_env.py
import gym
from gym import spaces
import numpy as np
from data_loader import load_and_preprocess_data

class TwitterPostingEnv(gym.Env):
    def __init__(self, data_path):
        super(TwitterPostingEnv, self).__init__()

        self.features, self.rewards = load_and_preprocess_data(data_path)
        
        # Normalize rewards
        self.reward_mean = np.mean(self.rewards)
        self.reward_std = np.std(self.rewards)
        
        self.rewards = (self.rewards - self.reward_mean) / self.reward_std  # Normalize the rewards
        
        self.current_step = 0
        self.max_steps = len(self.features)
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 3 actions: post tweet, edit tweet, delete tweet
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32)


    def reset(self):
        self.current_step = 0
        return self.features[self.current_step]

    def step(self, action):
        # The action is simple for now, affecting the engagement (likes, retweets)
        tweet_features = self.features[self.current_step]
        true_engagement = self.rewards[self.current_step]

        # The reward is engagement based on action
        if action == 0:  # post tweet
            reward = true_engagement * 1.05  # increase engagement a bit (simulating success)
        elif action == 1:  # edit tweet
            reward = true_engagement * 1.02  # smaller increase
        else:  # delete tweet
            reward = true_engagement * 0.5  # engagement drops

        # Increment step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Next state (or reset if done)
        next_state = self.features[self.current_step] if not done else np.zeros_like(tweet_features)

        return next_state, reward, done, {}

    def render(self):
        print(f"Step {self.current_step}, Tweet: Engagement {self.rewards[self.current_step]}")
