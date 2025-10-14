import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor

# TODO
# see impact of point on entropy increase or NMI increase
# get -1 outliers from HDBSCAN

class AutoEncoder(nn.Module):
    pass

class Nomad:
    def __init__(
            self,
    ) -> None:
        self.weights = {
            'purity': 0.5,
            'distance': 0.5,
            'incongruity': 1,
        }
        # self.auto_encoder = AutoEncoder()
        self.auto_encoder = PCA(n_components=0.9)
        # self.clustering_model = HDBSCAN()
        self.clustering_model = KMeans(n_clusters=100)
        self.distance_model = LocalOutlierFactor(
            n_neighbors=20,
            contamination='auto',
            metric='euclidean'
        )

    def _normalize_scores(self, scores):
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.zeros_like(scores) # Avoid division by zero
        return (scores - min_score) / (max_score - min_score)

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
        for c in np.unique(clusters):
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
    
    def evaluate_local_incongruity(
            self,
            y,
            clusters
    ):
        # incongruities = np.ones_like(y)
        # for c in np.unique(clusters):
        #     cluster_idx = clusters == c
        #     cluster_targets = y[cluster_idx]
        #     cluster_incongruities = incongruities[cluster_idx]

        #     targets, counts = np.unique_counts(cluster_targets)
        #     lognorm_freqs = -np.log2(counts / len(cluster_targets))

        #     for i, target in enumerate(targets):
        #         idx = cluster_targets == target
        #         cluster_incongruities[idx] = lognorm_freqs[i]
                
        # return incongruities

        # sort data by cluster, then target label
        # this groups the members of clusters together, and then the members of targets
        # this will help avoid our nested loops, and probably also helps with IO since elements are placed next to each other
        sorted_idx = np.lexsort((y, clusters))
        sorted_clusters = clusters[sorted_idx]
        sorted_y = y[sorted_idx]

        # use np unique indices to find the boundaries of each new cluster
        _, c_start_idx, c_counts = np.unique(
            sorted_clusters, return_index=True, return_counts=True
        )

        incongruities = np.zeros_like(y, dtype=float)

        # iterate through cluster boundaries
        for i in range(len(c_start_idx)):
            start = c_start_idx[i]
            count = c_counts[i]
            end = start + count

            # get targets of cluster
            c_targets = sorted_y[start:end]

            # find target frequencies
            targets, inv_idx, counts = np.unique(
                c_targets, return_inverse=True, return_counts=True
                )

            # get log-norm frequencies for targets
            lognorm_freqs = -np.log2(counts / count)

            # need to map the scores back to items, not sure how to do it better than this
            # target_incongruity_map = dict(zip(targets, lognorm_freqs))
            # incongruities_slice = np.array([target_incongruity_map[t] for t in c_targets])
            # incongruities[start:end] = incoangruities_slice

            # found a better way to do this using return inverse!
            incongruities[start:end] = lognorm_freqs[inv_idx]

        # unsort
        inv_sort_idx = np.empty_like(sorted_idx)
        inv_sort_idx[sorted_idx] = np.arange(len(y))

        return incongruities[inv_sort_idx]

    def score_anomalies(
            self,
            purity,
            distance,
            intrusion,
    ):
        # Normalize scores
        normalized_purity = self._normalize_scores(purity)
        normalized_distance = self._normalize_scores(distance)
        normalized_intrusion = self._normalize_scores(intrusion)

        # Apply weights
        anomaly_scores = (
            self.weights.get('purity', 0) * normalized_purity +
            self.weights.get('distance', 0) * normalized_distance +
            self.weights.get('intrusion', 0) * normalized_intrusion
        )
        return anomaly_scores

    def fit(self, x):
        self.auto_encoder.fit(x)

    def detect(self, x, y, weights=None):
        if weights:
            self.weights = weights

        # auto encode data
        # x_hat = self.auto_encoder(x)
        x_hat = self.auto_encoder.transform(x)
        clusters = self.clustering_model.fit_predict(x_hat)

        cluster_purity = self.evaluate_cluster_purity(y, clusters)
        point_purity = cluster_purity[clusters]
        point_distance = self.evaluate_distance(x)
        point_intrusion = self.evaluate_local_incongruity(y, clusters)

        anomaly_score = self.score_anomalies(
            point_purity,
            point_distance,
            point_intrusion,
        )

        return anomaly_score