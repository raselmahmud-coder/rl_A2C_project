# data_loader.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Preprocessing cleaned_text: Use TF-IDF to vectorize tweet content
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_features = vectorizer.fit_transform(df['cleaned_text']).toarray()

    # Add engagement features: retweets, likes, quotes
    engagement_features = df[['retweets', 'likes', 'quotes']].values

    # Normalize engagement features
    scaler = StandardScaler()
    engagement_features = scaler.fit_transform(engagement_features)

    # Combine text and engagement features
    features = np.hstack([tfidf_features, engagement_features])
    return features, df['likes'].values  # Reward: Number of likes

