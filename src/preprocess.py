import pandas as pd
import re
import os
import json

def load_data(file_path):
    """
    Load the dataset from a JSON file.
    :param file_path: Path to the JSON dataset.
    :return: DataFrame containing the dataset.
    """
    # Load JSON data
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    # Flatten and extract the necessary fields
    tweets = []
    for tweet in data:
        tweets.append({
            "id": tweet.get("id"),
            "timestamp": tweet.get("timestamp"),
            "username": tweet.get("username"),
            "text": tweet.get("text", ""),
            "retweets": tweet.get("retweets", 0),
            "likes": tweet.get("likes", 0),
            "quotes": tweet.get("quotes", 0),
        })
    return pd.DataFrame(tweets)

def clean_text(text):
    """
    Clean the tweet text by removing URLs, mentions, hashtags, and special characters.
    :param text: Raw tweet text.
    :return: Cleaned tweet text.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_text(text):
    """
    Check if the text is valid and should be kept.
    :param text: Cleaned tweet text.
    :return: True if the text is valid, False otherwise.
    """
    # Remove tweets with unsupported or garbled text (e.g., non-English scripts or gibberish)
    if re.search(r'^[“ð]|[ð]|[à¦]|[\u0980-\u09FF]|[\u0900-\u097F]|[\u0600-\u06FF]|[“ð]$', text):  # Garbled text pattern and Bengali, Hindi, Urdu text
        return False
    # Remove tweets with less than 30 characters
    if len(text) < 30:
        return False
    return True

def preprocess_data(input_path, output_path):
    """
    Preprocess the dataset by cleaning tweet text, filtering invalid tweets, and saving the result.
    :param input_path: Path to the raw dataset.
    :param output_path: Path to save the processed dataset.
    """
    # Load data
    data = load_data(input_path)
    print(f"Loaded {len(data)} tweets.")

    # Drop rows if any row is empty
    data = data[data['text'].notnull()]
    data = data[data["id"].notnull()]
    data = data[data["timestamp"].notnull()]
    data = data[data["username"].notnull()]
    data = data[data["retweets"].notnull()]
    data = data[data["likes"].notnull()]
    data = data[data["quotes"].notnull()]
    print(f"After removing empty rows: {len(data)} tweets.")

    # Clean tweet text
    data['cleaned_text'] = data['text'].apply(clean_text)

    # Filter out invalid tweets
    data = data[data['cleaned_text'].apply(is_valid_text)]
    print(f"After filtering invalid tweets: {len(data)} tweets.")

    # Convert the timestamp column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Define the timestamp range
    start_date = '2024-01-15T05:46:00.000Z'
    end_date = '2024-09-15T05:46:00.000Z'

    # Filter the tweets based on the timestamp range
    filtered_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
    print(f"After filtering by timestamp: {len(filtered_data)} tweets.")

    # Save the processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Define file paths
    input_file = "./data/tweets.json"  # Adjust path as needed
    output_file = "./data/split_processed_tweets.csv"

    # Preprocess the data
    preprocess_data(input_file, output_file)