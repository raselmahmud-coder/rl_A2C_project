import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import ClusteringEnv

def train_rl_agent(embeddings_path, clusters_path, model_path, timesteps=10000):
    """
    Train a PPO agent on the custom ClusteringEnv.
    :param embeddings_path: Path to the .npz file containing tweet embeddings.
    :param clusters_path: Path to the CSV file with cluster assignments.
    :param model_path: Path to save the trained PPO model.
    :param timesteps: Number of training timesteps.
    """
    # Initialize the custom environment
    env = ClusteringEnv(embeddings_path, clusters_path)

    # Wrap the environment for vectorized training
    env = make_vec_env(lambda: env, n_envs=1)

    # Initialize the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=timesteps)

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

def evaluate_model_with_accuracy(embeddings_path, clusters_path, model_path, episodes=10):
    """
    Evaluate the trained PPO model on the custom ClusteringEnv and calculate accuracy.
    :param embeddings_path: Path to the .npz file containing tweet embeddings.
    :param clusters_path: Path to the CSV file with cluster assignments.
    :param model_path: Path to the trained PPO model.
    :param episodes: Number of episodes to evaluate.
    """
    # Load the environment and model
    env = ClusteringEnv(embeddings_path, clusters_path)
    model = PPO.load(model_path)

    total_rewards = []
    total_correct_predictions = 0
    total_predictions = 0

    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        correct_predictions = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            total_predictions += 1

            # Count correct predictions (reward +1 means correct)
            if reward == 1.0:
                correct_predictions += 1
                total_correct_predictions += 1

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Accuracy = {(correct_predictions / env.embeddings.shape[0]) * 100:.2f}%")

    avg_reward = sum(total_rewards) / episodes
    avg_accuracy = (total_correct_predictions / (episodes * env.embeddings.shape[0])) * 100
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Average Accuracy over {episodes} episodes: {avg_accuracy:.2f}%")



if __name__ == "__main__":
    # File paths
    embeddings_file = "./data/tweet_embeddings.npz"
    clusters_file = "./data/tweet_clusters.csv"
    model_file = "./models/ppo_clustering"

    # Train the RL agent
    # train_rl_agent(embeddings_file, clusters_file, model_file, timesteps=100000)
# Evaluate the trained model with accuracy metrics
evaluate_model_with_accuracy(embeddings_file, clusters_file, model_file, episodes=10)
