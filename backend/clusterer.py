import numpy as np
from sklearn.cluster import KMeans

def cluster_articles(embeddings, n_clusters=3):
    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels.tolist()
