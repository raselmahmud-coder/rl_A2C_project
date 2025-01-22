import numpy as np
import gym
from gym import spaces

class ClusteringEnv(gym.Env):
    """
    A custom Gym environment for clustering-based proxy rewards.
    """
    def __init__(self, embeddings_path, clusters_path):
        """
        Initialize the environment.
        :param embeddings_path: Path to the .npz file containing tweet embeddings.
        :param clusters_path: Path to the CSV file with cluster assignments.
        """
        super(ClusteringEnv, self).__init__()

        # Load embeddings and cluster labels
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"]
        clusters = np.loadtxt(clusters_path, delimiter=",", skiprows=1, dtype=int)
        self.cluster_labels = clusters[:, 1]

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(set(self.cluster_labels)))  # Number of clusters
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.embeddings.shape[1],), dtype=np.float32
        )

        # Initialize state
        self.current_index = 0
        self.seed_value = None

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        :param seed: Integer seed value.
        :return: [seed] for compatibility with Gym API.
        """
        self.seed_value = seed
        np.random.seed(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: Initial observation (embedding of the first tweet).
        """
        self.current_index = 0
        return self.embeddings[self.current_index]

    def step(self, action):
        """
        Take a step in the environment.
        :param action: Predicted cluster label.
        :return: A tuple (next_state, reward, done, info).
        """
        # Get the ground-truth cluster label
        true_label = self.cluster_labels[self.current_index]

        # Calculate reward
        reward = 1.0 if action == true_label else -1.0  # Reward correct prediction, penalize incorrect

        # Move to the next tweet
        self.current_index += 1
        done = self.current_index >= len(self.embeddings)

        # Get the next state
        next_state = self.embeddings[self.current_index] if not done else None

        return next_state, reward, done, {}

    def render(self, mode="human"):
        """
        Optional: Render the environment (not used here).
        """
        pass
