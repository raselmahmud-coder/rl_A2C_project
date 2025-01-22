# Clustering-Based Proxy Rewards for Topic Modeling
Goal: Automatically cluster tweets into topics without manual labels and use RL to optimize the clustering quality.
Approach:
Use unsupervised clustering (e.g., K-Means or DBSCAN) on embeddings of tweets to form initial "pseudo-labels."
Train an RL agent to predict cluster assignments for tweets.
Reward: Use a metric like silhouette score, Davies-Bouldin index, or inter-cluster vs. intra-cluster distance as a proxy reward.
Outcome:
Optimized topic assignments and an RL policy that clusters tweets dynamically.
Research Question: Can RL improve clustering quality by learning a better tweet-to-cluster assignment policy?