# evaluate.py: Evaluate the trained A2C agent on the test data and plot the results
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from twitter_posting_env import TwitterPostingEnv
import os

def evaluate_agent(data_path, save_path='./evaluation_results/'):
    # Create the directory to save plots if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

     # Create the base environment
    base_env = lambda: TwitterPostingEnv(data_path)

    # Wrap the environment in DummyVecEnv
    env = DummyVecEnv([base_env])
    # Load VecNormalize stats
    env = VecNormalize.load("vec_normalize_3rd_phase_stats.pkl", env)
    env.training = False  # Turn off training mode
    env.norm_reward = False  # Turn off reward normalization during evaluation


    # Load the trained model
    model = A2C.load("a2c_tweet_trend_model")

    # Initialize variables for tracking metrics
    total_rewards = []
    average_rewards = []
    random_rewards = []
    episodes = 100
    cumulative_reward = 0
    discount_factor = 0.99  # Discounted reward

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(state)
            state, reward, done, _info = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        cumulative_reward += total_reward
        average_reward = np.mean(total_rewards[-50:])  # Last 50 episodes for smoother learning curve
        average_rewards.append(average_reward)

        # Calculate discounted reward
        discounted_reward = sum([r * (discount_factor ** i) for i, r in enumerate(total_rewards)])

        # Track random agent performance for comparison
        random_reward = np.random.random()  # Simulate a random reward for comparison
        random_rewards.append(random_reward)

    # Plotting the results
    plt.figure(figsize=(12, 8))

    # Total Reward and Average Reward per Episode
    plt.subplot(2, 2, 1)
    plt.plot(total_rewards, label="Total Reward")
    plt.plot(average_rewards, label="Average Reward", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total and Average Reward per Episode")
    plt.legend()
    plt.savefig(f"{save_path}/total_and_average_reward.png")

    # Learning Curve: Reward trend over episodes
    plt.subplot(2, 2, 2)
    plt.plot(average_rewards, label="Average Reward", color='b')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve (Reward Trend)")
    plt.legend()
    plt.savefig(f"{save_path}/learning_curve.png")

    # Comparison with Random Agent
    plt.subplot(2, 2, 3)
    plt.plot(random_rewards, label="Random Agent", linestyle="--")
    plt.plot(average_rewards, label="RL Agent")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Comparison with Random Agent")
    plt.legend()
    plt.savefig(f"{save_path}/comparison_with_random_agent.png")

    # Convergence (Variance Check) - Track variance over episodes
    variance_rewards = [np.var(total_rewards[max(0, i-100):i]) for i in range(1, episodes + 1)]
    plt.subplot(2, 2, 4)
    plt.plot(variance_rewards, label="Variance (Last 100 Episodes)")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.title("Convergence Check")
    plt.legend()
    plt.savefig(f"{save_path}/convergence_check.png")

    plt.tight_layout()
    plt.show()

    window_size = 50  # Adjusted for better smoothing

    def moving_average(arr, window_size):
        return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

    # Use this moving average for analysis
    smoothed_rewards = moving_average(average_rewards, window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_rewards, label="Smoothed Average Reward", color='b')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Smoothed Learning Curve (Reward Trend)")
    plt.legend()
    plt.savefig(f"{save_path}/smoothed_learning_curve.png")
    plt.show()

    # Return Cumulative, Average, Discounted Reward
    return cumulative_reward, average_rewards[-1], discounted_reward

if __name__ == "__main__":
    cumulative_reward, average_reward, discounted_reward = evaluate_agent('./data/test_processed_tweets.csv', save_path='./test_evaluation_results')
    print(f"Cumulative Reward: {cumulative_reward}")
    print(f"Average Reward: {average_reward}")
    print(f"Discounted Reward: {discounted_reward}")
