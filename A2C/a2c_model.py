import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

class TweetEngagementEnv(gym.Env):
    def __init__(self, data):
        super(TweetEngagementEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.num_features = 4  # Number of features: likes, retweets, quotes, timestamp_numeric

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_features,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # Actions: Low, Medium, High engagement

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)  # Gym's built-in seed handling
        self.current_step = 0
        state = self._get_state()
        return state, {}  # Return state and an empty info dictionary

    def _get_state(self):
        """Retrieve the current state."""
        tweet = self.data.iloc[self.current_step]
        return np.array([
            tweet['likes'],
            tweet['retweets'],
            tweet['quotes'],
            tweet['timestamp_numeric']
        ], dtype=np.float32)

    def step(self, action):
        """Execute an action."""
        tweet = self.data.iloc[self.current_step]
        true_label = tweet['engagement_category']

        # Reward logic
        reward = 1 if action == true_label else -1

        self.current_step += 1
        done = self.current_step >= len(self.data)
        truncated = False  # For this use case, truncated is always False

        if not done:
            next_state = self._get_state()
        else:
            next_state = np.zeros(self.num_features, dtype=np.float32)

        return next_state, reward, done, truncated, {}

# Training function
def train_a2c_model(train_data):
    """Train the A2C model."""
    env = DummyVecEnv([lambda: TweetEngagementEnv(train_data)])
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_tensorboard/")
    print("[INFO] Starting A2C training...")
    model.learn(total_timesteps=200000)  # Adjust timesteps as necessary
    model.save("a2c_engagement_model")
    print("[INFO] Model saved as 'a2c_engagement_model'.")
    return model
