import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
from MultiAgentTwitterEnv import MultiAgentTwitterEnv


def train_marl(env, total_timesteps=500000):
    """
    Train the MARL agent with improved hyperparameters and evaluation callback.
    """
    # Wrap the environment for parallelized training
    env = SubprocVecEnv([lambda: env] * 4)  # Use 4 parallel environments

    # Improved PPO hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",  # Use CPU
        learning_rate=2.5e-4,  # Lower learning rate for more stable updates
        n_steps=2048,  # More steps per update
        batch_size=64,  # Smaller batch size
        ent_coef=0.01,  # Encourage exploration
        clip_range=0.2,  # Default PPO clipping range
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter for advantage calculation
    )

    # Callback to evaluate the model during training
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    return model


def evaluate_agents(model, env, episodes=10):
    """
    Evaluate the trained agent in the environment.
    """
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()  # Reset returns (observation, info)
        total_reward = 0
        terminated, truncated = False, False
        while not (terminated or truncated):  # Continue until the episode ends
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)  # Handle five return values
            total_reward += reward
        rewards.append(total_reward)
    avg_reward = sum(rewards) / episodes
    print(f"Average Reward over {episodes} episodes: {avg_reward}")


def random_policy(env, episodes=10):
    """
    Evaluate a random policy in the environment.
    """
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    avg_reward = sum(rewards) / episodes
    print(f"Random Policy Average Reward: {avg_reward}")


def plot_learning_curve(rewards):
    """
    Plot the learning curve over training episodes.
    """
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')
    plt.show()


if __name__ == "__main__":
    # Load the dataset
    data_path = './data/split_processed_tweets.csv'  # Replace with the correct path to your dataset
    data = pd.read_csv(data_path)

    # Create the environment
    env = MultiAgentTwitterEnv(data)

    # Train the agent
    model = train_marl(env)

    # Evaluate the trained agent
    evaluate_agents(model, env)
    random_policy(env)
