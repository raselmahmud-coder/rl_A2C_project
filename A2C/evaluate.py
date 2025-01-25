from stable_baselines3 import A2C
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import load_and_split_data
from a2c_model import TweetEngagementEnv
from stable_baselines3.common.vec_env import DummyVecEnv


# Define engagement bins (same as before)
def categorize_engagement(score):
    if score < 0.33:  
        return 0  # Low engagement
    elif score < 0.66:
        return 1  # Medium engagement
    else:
        return 2  # High engagement

def evaluate_model(model, test_data):
    env = DummyVecEnv([lambda: TweetEngagementEnv(test_data)])

    true_categories = []
    predicted_categories = []

    # Dynamically calculate engagement category
    test_data['engagement_score'] = test_data['likes'] + test_data['retweets'] * 0.5 + test_data['quotes'] * 2
    test_data['engagement_category'] = test_data['engagement_score'].apply(categorize_engagement)

    for i in range(len(test_data)):
        state = env.reset()
        done = False
        while not done:
            action, _states = model.predict(state)
            state, reward, done, info = env.step(action)

            true_categories.append(test_data.iloc[i]['engagement_category'])
            predicted_categories.append(action)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_categories, predicted_categories)
    precision = precision_score(true_categories, predicted_categories, average='weighted', zero_division=0)
    recall = recall_score(true_categories, predicted_categories, average='weighted', zero_division=0)
    f1 = f1_score(true_categories, predicted_categories, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")


if __name__ == "__main__":
    # Load the trained model
    model = A2C.load("a2c_engagement_model")

    # Load and split data
    train_data, test_data = load_and_split_data('./data/split_processed_tweets.csv')

    # Evaluate the model
    evaluate_model(model, test_data)
