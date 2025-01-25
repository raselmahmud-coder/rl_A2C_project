import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Categorize engagement into bins
def categorize_engagement(score):
    """Categorize engagement scores into Low, Medium, and High bins."""
    if score < 0.33:
        return 0  # Low engagement
    elif score < 0.66:
        return 1  # Medium engagement
    else:
        return 2  # High engagement

# Balance the dataset using SMOTE
def balance_data(df):
    """Balance the dataset using SMOTE."""
    print("\n[DEBUG] Original engagement category distribution:")
    print(df['engagement_category'].value_counts())

    # Determine the appropriate k_neighbors for SMOTE
    smote_k_neighbors = min(df['engagement_category'].value_counts().min() - 1, 5)
    if smote_k_neighbors < 1:
        print("[ERROR] Not enough samples in the minority class to apply SMOTE.")
        return df

    print(f"[DEBUG] Setting SMOTE k_neighbors to: {smote_k_neighbors}")
    smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)

    # Features and labels for SMOTE
    features = df[['likes', 'retweets', 'quotes', 'timestamp_numeric']]
    labels = df['engagement_category']

    # Perform SMOTE resampling
    features_resampled, labels_resampled = smote.fit_resample(features, labels)

    # Combine resampled features and labels into a DataFrame
    resampled_df = pd.DataFrame(features_resampled, columns=['likes', 'retweets', 'quotes', 'timestamp_numeric'])
    resampled_df['engagement_category'] = labels_resampled

    # Map 'cleaned_text' correctly for synthetic samples
    original_texts = df['cleaned_text'].tolist()
    resampled_df['cleaned_text'] = [
        original_texts[i % len(original_texts)] for i in range(len(resampled_df))
    ]

    print("\n[DEBUG] Resampled engagement category distribution:")
    print(resampled_df['engagement_category'].value_counts())
    return resampled_df

# Load and preprocess the dataset
def load_and_split_data(file_path):
    """Load, preprocess, and split the dataset into train and test sets."""
    print("[DEBUG] Loading dataset...")
    df = pd.read_csv(file_path)

    if 'cleaned_text' not in df.columns:
        raise KeyError("[ERROR] 'cleaned_text' column is missing in the dataset.")

    # Fill missing values in 'cleaned_text'
    df['cleaned_text'] = df['cleaned_text'].fillna('')

    # Fill missing numeric values and normalize
    df[['likes', 'retweets', 'quotes']] = df[['likes', 'retweets', 'quotes']].fillna(0)
    scaler = MinMaxScaler()
    df[['likes', 'retweets', 'quotes']] = scaler.fit_transform(df[['likes', 'retweets', 'quotes']])

    # Convert timestamp to numeric and normalize
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp_numeric'] = df['timestamp'].apply(lambda x: x.timestamp() if pd.notnull(x) else 0)
    df['timestamp_numeric'] = scaler.fit_transform(df[['timestamp_numeric']])

    # Categorize engagement
    df['engagement_score'] = df['likes'] + df['retweets'] * 0.5 + df['quotes'] * 2
    df['engagement_category'] = df['engagement_score'].apply(categorize_engagement)

    print("\n[DEBUG] Columns in dataset after preprocessing:", df.columns)
    print("[DEBUG] Sample rows:", df[['cleaned_text']].head())

    # Balance the dataset
    df = balance_data(df)

    # Split into train and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    print("\n[DEBUG] Train-test split complete.")
    print("[DEBUG] Training data size:", len(train_data))
    print("[DEBUG] Test data size:", len(test_data))
    return train_data, test_data
