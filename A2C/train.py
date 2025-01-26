# train.py
from stable_baselines3 import A2C
from twitter_posting_env import TwitterPostingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

def train_agent(data_path):
    # Create the environment
    env = TwitterPostingEnv(data_path)
    env = DummyVecEnv([lambda: TwitterPostingEnv('./data/split_processed_tweets.csv')])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # Save the normalized stats for evaluation consistency
    env.save("vec_normalize_3rd_phase_stats.pkl")


    # Initialize the A2C model
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0001, ent_coef=0.05)
    # Normalize the environment for better stability

    # Train the model
    model.learn(total_timesteps=200000)  # You can adjust the number of timesteps

    # Save the model
    model.save("a2c_tweet_trend_model")

if __name__ == "__main__":
    train_agent('./data/split_processed_tweets.csv')
