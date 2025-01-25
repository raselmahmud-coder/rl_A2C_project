import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def visualize_clusters(embeddings_path, clusters_path, output_path="./visualizations/clusters.png"):
    """
    Visualize embeddings in 2D using t-SNE and color-code them by cluster labels.
    :param embeddings_path: Path to the .npz file containing tweet embeddings.
    :param clusters_path: Path to the CSV file with cluster assignments.
    :param output_path: Path to save the cluster visualization plot.
    """
    # Load embeddings and clusters
    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    clusters = pd.read_csv(clusters_path)

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=3000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=clusters["cluster"],
        cmap="viridis",
        s=10
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("t-SNE Visualization of Tweet Embeddings with Clusters")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(output_path)
    plt.show()

# Run the visualization
# visualize_clusters("./data/tweet_embeddings.npz", "./data/tweet_clusters.csv")





def interpret_clusters(embeddings_path, clusters_path, processed_tweets_path, output_path="./analysis/cluster_analysis.txt"):
    """
    Analyze and interpret clusters by displaying example tweets from each cluster.
    :param embeddings_path: Path to the .npz file containing tweet embeddings.
    :param clusters_path: Path to the CSV file with cluster assignments.
    :param processed_tweets_path: Path to the processed tweets CSV file.
    :param output_path: Path to save cluster analysis results.
    """
    # Load clusters and tweets
    clusters = pd.read_csv(clusters_path)
    tweets = pd.read_csv(processed_tweets_path)

    # Merge data
    data = pd.merge(clusters, tweets, on="id")

    # Analyze clusters
    analysis_results = []
    for cluster_id in sorted(data["cluster"].unique()):
        cluster_tweets = data[data["cluster"] == cluster_id]["cleaned_text"]
        analysis_results.append(f"Cluster {cluster_id} ({len(cluster_tweets)} tweets):\n")
        analysis_results.extend([f"  - {tweet}" for tweet in cluster_tweets.head(5)])
        analysis_results.append("\n")

    # Save and print results
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines("\n".join(analysis_results))
    print(f"Cluster analysis saved to {output_path}")

# Run the analysis
# if __name__ == "__main__":
    # interpret_clusters("./data/tweet_embeddings.npz", "./data/tweet_clusters.csv", "./data/processed_tweets.csv")






def compare_clustering(embeddings_path, rl_clusters_path, processed_tweets_path):
    """
    Compare RL-based clustering with K-Means clustering.
    :param embeddings_path: Path to the .npz file containing tweet embeddings.
    :param rl_clusters_path: Path to RL clustering CSV.
    :param processed_tweets_path: Path to the processed tweets CSV file.
    """
    # Load embeddings
    data = np.load(embeddings_path)
    embeddings = data["embeddings"]

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)

    # Compute Silhouette Score for K-Means
    kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
    print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")

    # Compute Silhouette Score for RL-based clustering
    rl_clusters = pd.read_csv(rl_clusters_path)["cluster"]
    rl_silhouette = silhouette_score(embeddings, rl_clusters)
    print(f"RL Clustering Silhouette Score: {rl_silhouette:.4f}")

# Run comparison
if __name__ == "__main__":
    compare_clustering("./data/tweet_embeddings.npz", "./data/tweet_clusters.csv", "./data/processed_tweets.csv")
