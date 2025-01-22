import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import os
import numpy as np

def load_processed_data(file_path):
    """
    Load the processed dataset from a CSV file.
    :param file_path: Path to the processed CSV file.
    :return: DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)

def generate_embeddings(texts, model, tokenizer, batch_size=32, device="cuda"):
    """
    Generate embeddings for a list of texts using a pre-trained language model.
    :param texts: List of cleaned tweet texts.
    :param model: Pre-trained transformer model.
    :param tokenizer: Tokenizer for the pre-trained model.
    :param batch_size: Number of texts to process in a single batch.
    :param device: Device to run the model (e.g., "cuda" or "cpu").
    :return: Numpy array of embeddings.
    """
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            # Use the [CLS] token's embedding as the sentence embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())

    # Concatenate all embeddings into a single numpy array
    return np.vstack(embeddings)

def save_embeddings(output_path, ids, embeddings):
    """
    Save embeddings to a numpy file.
    :param output_path: Path to save the embeddings.
    :param ids: List of tweet IDs.
    :param embeddings: Numpy array of embeddings.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, ids=ids, embeddings=embeddings)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    # File paths
    processed_file = "./data/processed_tweets.csv"
    output_file = "./data/tweet_embeddings.npz"

    # Load processed data
    data = load_processed_data(processed_file)
    texts = data["cleaned_text"].tolist()
    ids = data["id"].tolist()

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    # Generate embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = generate_embeddings(texts, model, tokenizer, device=device)

    # Save embeddings
    save_embeddings(output_file, ids, embeddings)
