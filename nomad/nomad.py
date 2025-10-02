import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor

class AutoEncoder(nn.Module):
    pass

class Nomad:
    def __init__(
            self,
    ) -> None:
        # self.auto_encoder = AutoEncoder()
        self.auto_encoder = PCA(n_components=0.9)
        # self.clustering_model = HDBSCAN()
        self.clustering_model = KMeans(n_clusters=30)
        self.distance_model = LocalOutlierFactor(
            n_neighbors=20,
            contamination='auto',
            metric='euclidean'
        )

    def get_entropy(
            self,
            node
    ):
        freqs = np.unique(node, return_counts=True)[1] # [1] to get counts
        norm_freqs = freqs / np.sum(freqs)
        entropy = norm_freqs * np.log2(norm_freqs)
        entropy = - np.sum(entropy)
        return entropy

    def evaluate_cluster_purity(
            self,
            y,
            clusters,
    ):
        # loop through different clusters
        entropies = []
        for c in range(len(np.unique(clusters))):
            c_idx = clusters == c
            entropy = self.get_entropy(y[c_idx])
            entropies.append(entropy)

        return np.array(entropies)
    
    def evaluate_distance(
            self,
            x,
    ):
        y_pred = self.distance_model.fit_predict(x)
        raw_lof_distances = self.distance_model.negative_outlier_factor_

        return - raw_lof_distances # lower value means higher anomaly, so negative

    def score_anomalies(
            self,
            purity,
            distance,
            alpha,
    ):
        anomaly_scores = alpha * distance + (1 - alpha) * purity
        return anomaly_scores

    def fit(self, x):
        self.auto_encoder.fit(x)

    def detect(self, x, y):
        # auto encode data
        # x_hat = self.auto_encoder(x)
        x_hat = self.auto_encoder.transform(x)
        clusters = self.clustering_model.fit_predict(x_hat)

        cluster_purity = self.evaluate_cluster_purity(y, clusters)
        point_purity = cluster_purity[clusters]
        point_distance = self.evaluate_distance(x)

        anomaly_score = self.score_anomalies(point_purity, point_distance, alpha=0.5)

        return anomaly_score