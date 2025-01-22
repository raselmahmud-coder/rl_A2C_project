import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt

def load_embeddings(file_path):
    """
    Load tweet embeddings and IDs from a .npz file.
    :param file_path: Path to the .npz file containing embeddings.
    :return: Tuple of tweet IDs and embeddings.
    """
    data = np.load(file_path)
    return data["ids"], data["embeddings"]

def perform_clustering(embeddings, num_clusters):
    """
    Perform K-Means clustering on the embeddings.
    :param embeddings: Numpy array of tweet embeddings.
    :param num_clusters: Number of clusters for K-Means.
    :return: KMeans model and cluster labels.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels

def find_optimal_clusters(embeddings, max_clusters=10):
    """
    Determine the optimal number of clusters using the silhouette score.
    :param embeddings: Numpy array of tweet embeddings.
    :param max_clusters: Maximum number of clusters to evaluate.
    :return: Optimal number of clusters.
    """
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker="o")
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.show()

    # Return the number of clusters with the highest silhouette score
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")
    return optimal_clusters

def save_clusters(output_path, ids, labels):
    """
    Save cluster assignments for each tweet to a CSV file.
    :param output_path: Path to save the cluster assignments.
    :param ids: List of tweet IDs.
    :param labels: Cluster labels for each tweet.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cluster_data = {"id": ids, "cluster": labels}
    np.savetxt(output_path, np.column_stack([ids, labels]), delimiter=",", fmt="%s", header="id,cluster", comments="")
    print(f"Cluster assignments saved to {output_path}")

if __name__ == "__main__":
    # File paths
    embeddings_file = "./data/tweet_embeddings.npz"
    output_file = "./data/tweet_clusters.csv"

    # Load embeddings
    ids, embeddings = load_embeddings(embeddings_file)

    # Determine optimal number of clusters
    optimal_clusters = find_optimal_clusters(embeddings, max_clusters=10)

    # Perform clustering
    kmeans, labels = perform_clustering(embeddings, num_clusters=optimal_clusters)

    # Save cluster assignments
    save_clusters(output_file, ids, labels)
