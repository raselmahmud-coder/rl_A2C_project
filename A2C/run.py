from a2c_model import train_a2c_model  # Import the A2C training function
from preprocess import load_and_split_data

def main():
    # Load and split data
    train_data, test_data = load_and_split_data('./data/split_processed_tweets.csv')

    # Train the A2C model
    model = train_a2c_model(train_data)
    print("Model trained successfully!")

if __name__ == "__main__":
    main()
