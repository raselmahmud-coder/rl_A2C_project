import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Preprocess data (e.g., normalize engagement metrics)
    df[['retweets', 'likes', 'quotes']] = StandardScaler().fit_transform(df[['retweets', 'likes', 'quotes']])
    
    # Split into training and testing sets (80% train, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return train_df, test_df
