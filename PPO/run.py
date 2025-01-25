from preprocess import load_and_split_data
from ppo_model import train_ppo_model

def main():
    # Load and split data
    train_data, test_data = load_and_split_data('./data/processed_tweets.csv')

    # Train the PPO model
    model = train_ppo_model(train_data)

    # Evaluate the PPO model
    from evaluate import evaluate_model
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
