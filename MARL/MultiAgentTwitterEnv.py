import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class MultiAgentTwitterEnv(gym.Env):
    def __init__(self, accounts_data):
        """
        Single-agent Twitter Environment for one account at a time.
        :param accounts_data: A Pandas DataFrame containing tweet data (username, engagement metrics, etc.).
        """
        super(MultiAgentTwitterEnv, self).__init__()

        # Extract unique accounts and shuffle them for random selection
        self.accounts_data = accounts_data
        self.accounts = accounts_data['username'].unique()
        self.current_account = None  # Track the current agent

        # Action space: Decide engagement strategy (Post, Retweet, Reply, Wait)
        self.action_space = spaces.Discrete(4)  # Actions: 0 = Post, 1 = Retweet, 2 = Reply, 3 = Wait

        # Observation space: Engagement metrics and trending score
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )  # [retweets, likes, quotes, trending_score]

    def _initialize_state(self, account):
        """
        Initialize the state for the given account.
        :param account: The username for the current account.
        """
        account_data = self.accounts_data[self.accounts_data['username'] == account]

        if account_data.empty:
            # Handle case where no data exists for the account
            return np.array([0, 0, 0, 0.0], dtype=np.float32)  # Default state
        else:
            # Extract the first row of data for the account
            initial_data = account_data.iloc[0]
            return np.array([
                initial_data['retweets'],
                initial_data['likes'],
                initial_data['quotes'],
                random.uniform(0, 1)  # Simulated trending score
            ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        """
        # Select a random account
        self.current_account = random.choice(self.accounts)
        self.state = self._initialize_state(self.current_account)

        # Return the observation and empty info
        return self.state, {}

    def step(self, action):
        """
        Execute an action and update the state.
        :param action: The action taken by the agent.
        :return: Tuple of (observation, reward, terminated, truncated, info)
        """
        # Simulate engagement changes based on action
        engagement_change = random.uniform(0.1, 0.5)
        if action == 0:  # Post
            engagement_change *= 2
        elif action == 1:  # Retweet
            engagement_change *= 1.5
        elif action == 2:  # Reply
            engagement_change *= 1.2

        # Update state
        self.state[0] += engagement_change * random.randint(1, 10)  # Retweets
        self.state[1] += engagement_change * random.randint(1, 10)  # Likes
        self.state[2] += engagement_change * random.randint(1, 5)   # Quotes
        self.state[3] = random.uniform(0, 1)  # Trending score

        # Reward is proportional to engagement change
        reward = (
            self.state[0] * 0.5
            + self.state[1] * 0.3
            + self.state[2] * 0.2
        )  # Weighted sum of engagement metrics

        # Terminate the episode randomly for simplicity
        done = random.random() < 0.1  # End based on random chance
        truncated = False  # No specific truncation logic for now

        return self.state, reward, done, truncated, {}
