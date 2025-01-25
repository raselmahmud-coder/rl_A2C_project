import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from sklearn.preprocessing import StandardScaler

# Define the PPO environment for tweet engagement optimization
class EngagementEnv(gym.Env):
    def __init__(self, data):
        super(EngagementEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.num_actions = 3  # Actions: ['tweet', 'like', 'share']
        
        # Define observation space: [retweets, likes, quotes, text length]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Define action space: 3 discrete actions
        self.action_space = gym.spaces.Discrete(self.num_actions)

    def reset(self):
        """Reset the environment."""
        self.current_step = 0
        tweet = self.data.iloc[self.current_step]
        state = np.array([tweet['retweets'], tweet['likes'], tweet['quotes'], len(tweet['cleaned_text'])])
        return state

    def step(self, action):
        """Take an action and return the new state, reward, done, and info."""
        tweet = self.data.iloc[self.current_step]
        reward = 0
        
        # Reward based on action taken (for now simple reward structure based on engagement)
        if action == 0:  # 'tweet'
            reward = tweet['likes'] + tweet['retweets']
        elif action == 1:  # 'like'
            reward = tweet['likes']
        elif action == 2:  # 'share'
            reward = tweet['retweets']

        self.current_step += 1
        done = self.current_step >= len(self.data)

        # If done, next state is all zeros
        if not done:
            next_state = np.array([self.data.iloc[self.current_step]['retweets'], self.data.iloc[self.current_step]['likes'],
                                   self.data.iloc[self.current_step]['quotes'], len(self.data.iloc[self.current_step]['cleaned_text'])])
        else:
            next_state = np.zeros(self.observation_space.shape)  # Reset state when done

        return next_state, reward, done, {}

def train_ppo_model(train_data):
    """Train PPO model on user engagement data."""
    env = DummyVecEnv([lambda: EngagementEnv(train_data)])

    # Create the PPO model using stable-baselines3
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    # Train the model (total timesteps can be adjusted as needed)
    model.learn(total_timesteps=50000)

    # Save the trained model
    model.save("ppo_engagement_model")
    return model
