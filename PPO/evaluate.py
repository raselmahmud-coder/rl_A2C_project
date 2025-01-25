from stable_baselines3 import PPO
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import load_and_split_data
from ppo_model import EngagementEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluate_model(model, test_data):
    env = DummyVecEnv([lambda: EngagementEnv(test_data)])
    
    true_labels = []
    predictions = []

    # Evaluate the model over the test data
    for _ in range(len(test_data)):
        state = env.reset()
        done = False
        while not done:
            # Get action from the trained model
            action, _states = model.predict(state)
            
            # Get the reward and the next state
            state, reward, done, info = env.step(action)
            
            # Assuming 'action' is the prediction (you can map your actions to the appropriate class)
            predicted_label = action  # This should be adjusted to match your task (e.g., 'FAVOR', 'AGAINST', 'NONE')
            true_label = test_data.iloc[_]['text']  # Replace with the correct true label (such as stance, etc.)
            
            predictions.append(predicted_label)
            true_labels.append(true_label)

    # Compute the accuracy and other metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

if __name__ == "__main__":
    # Load the trained model
    model = PPO.load("ppo_engagement_model")

    # Load and split data
    train_data, test_data = load_and_split_data('./data/processed_tweets.csv')

    # Evaluate the model
    evaluate_model(model, test_data)
