import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

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
visualize_clusters("./data/tweet_embeddings.npz", "./data/tweet_clusters.csv")


def plot_training_progress(log_folder, output_path="./visualizations/training_progress.png"):
    """
    Plot training progress (rewards over timesteps) from PPO logs.
    :param log_folder: Folder where training logs are stored.
    :param output_path: Path to save the training progress plot.
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Training Progress: Rewards Over Timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.grid()
    plt.savefig(output_path)
    plt.show()

# Run the visualization
# plot_training_progress("./logs")
