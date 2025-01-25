import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from MultiAgentTwitterEnv import MultiAgentTwitterEnv

# Define paths
BEST_MODEL_PATH = './logs/best_model/best_model.zip'
EVALUATIONS_PATH = './logs/evaluations.npz'
DATA_PATH = './data/split_processed_tweets.csv'

def load_best_model(model_path):
    """Load the best model saved during training."""
    model = PPO.load(model_path)
    print("Best model loaded successfully.")
    return model

def load_evaluation_data(evaluations_path):
    """Load evaluation data from the evaluations.npz file."""
    data = np.load(evaluations_path)
    timesteps = data['timesteps']
    mean_rewards = data['results'].mean(axis=1)
    std_rewards = data['results'].std(axis=1)
    print("Evaluation data loaded successfully.")
    return timesteps, mean_rewards, std_rewards

def plot_evaluation_curve(timesteps, mean_rewards, std_rewards):
    """Plot the learning curve based on evaluation results."""
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label='Mean Reward')
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='±1 Std Dev')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Evaluation Reward Curve')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_model_in_environment(model, env, episodes=10):
    """Evaluate the trained model in the environment."""
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Evaluation over {episodes} episodes:")
    print(f"  - Average Reward: {avg_reward:.2f}")
    print(f"  - Standard Deviation of Reward: {std_reward:.2f}")
    return avg_reward, std_reward

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv(DATA_PATH)

    # Create the environment
    env = MultiAgentTwitterEnv(data)

    # Load the best model
    model = load_best_model(BEST_MODEL_PATH)

    # Load evaluation data
    timesteps, mean_rewards, std_rewards = load_evaluation_data(EVALUATIONS_PATH)

    # Plot evaluation curve
    plot_evaluation_curve(timesteps, mean_rewards, std_rewards)

    # Evaluate the model in the environment
    evaluate_model_in_environment(model, env)
