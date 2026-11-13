# -*- coding: utf-8 -*-
"""
FINCH - First Integer Neighbor Clustering Hierarchy Algorithm
"""

# Author: Eren Cakmak <eren.cakmak@uni-konstanz.de>
#
# License: MIT

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from sklearn.utils import check_array
from sklearn.metrics import silhouette_score


def select_partition(partitions, target_count=None, mode='last'):
    """Select a FINCH level; closest maximizes rectangular match coverage."""
    if not partitions:
        raise ValueError('FINCH produced no partitions')
    if mode == 'last':
        return list(partitions.values())[-1]
    if mode != 'closest' or target_count is None or target_count < 1:
        raise ValueError(f'Invalid partition selection: {mode}, {target_count}')
    return min(
        partitions.values(),
        key=lambda part: (
            max(part['n_clusters'], target_count) / min(part['n_clusters'], target_count),
            abs(part['n_clusters'] - target_count),
        ),
    )


class FINCH():
    """
    A class to perform the FINCH clustering
    
    Read more in paper see reference below.
    
    Parameters
    ----------        
    metric : string default='euclidean'
        The used distance metric - more options are
        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘rogerstanimoto’, ‘sqeuclidean’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘yule’.
        
    n_jobs : int or None, default=1
        The number of processes to start -1 means use all processors 
        
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data
        
    partitions : dict, contains all partitioning and their resulting labels, cluster centroids
        Cluster labels for the data
        
    References
    ----------
    Sarfraz, Saquib, Vivek Sharma, and Rainer Stiefelhagen. 
    "Efficient parameter-free clustering using first neighbor relations." 
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.

    """
    def __init__(self, metric='cosine', n_jobs=1, verbose=True):
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _finch(self, X, prev_clusters, prev_cluster_core_indices):
        """
        Compute the adjacency link matrix as described in the paper Eq.1
        Afterwards get connected components and their cluster centroids
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data samples that should be clustered.
        
        prev_clusters : list of ndarray of shape (2,)
            The cluster centroids of the previous partitioning.
            
        prev_cluster_core_indices : list 
            The previous samples belonging to the cluster of prev_clusters
        Returns
        -------
        n_connected_components_ : int
            The number of clusters in the partitioning.
        
        labels_ : list
            Cluster labels for all data samples between 0 and n_connected_components_
        
        cluster_centers_ : list of ndarray of shape (2,)
            The cluster centroids of the partitioning.
            
        cluster_core_indices_  : list 
            The samples belonging to the cluster of cluster_centers_        
        """

        # Adjacency link matrix by Eq.1
        connectivity = None

        # compute the adjacency link matrix
        if not prev_clusters:
            # first partitioning
            data = X
        else:
            data = prev_clusters

        # Compute the adjacency link matrix as described in the paper Eq.1
        # NN in sklearn
        nbrs = NearestNeighbors(n_neighbors=2,#2
                                metric=self.metric,
                                n_jobs=self.n_jobs).fit(data)

        # condition j = k_i - link nearest neighbors
        connectivity = nbrs.kneighbors_graph(data)

        # condition k_i = k_j - link same first neighbors
        # dot product forces symmtery therefore k_j plus k_j = i
        connectivity @= connectivity.T

        # remove diagonal
        connectivity.setdiag(0)
        connectivity.eliminate_zeros()

        # TODO this could also be solved by computing a linkage matrix
        # and then just calling the method scipy.cluster.hierarchy.fcluster
        # This will be probably increase the performance of the method further
        #
        # set values to one required for the linkage matrix
        # connectivity.data[:] = 1

        # get connected components
        n_connected_components_, labels_ = connected_components(
            csgraph=connectivity)

        # labels remap to previous cluster core indices
        # only called for second paritioning
        if len(labels_) < self.n_samples:
            new_labels = np.full(self.n_samples, 0)
            for i in range(n_connected_components_):
                idx = np.where(labels_ == i)[0]
                idx = sum([prev_cluster_core_indices[j] for j in idx], [])
                new_labels[idx] = i
            labels_ = new_labels

        # list of centroids and sample indices for each cluster
        cluster_centers_ = []
        cluster_core_indices_ = []

        # compute cluster centers with labels indicies
        for i in range(n_connected_components_):
            # update the cluster core indicies
            idx = np.where(labels_ == i)[0]
            cluster_core_indices_.append(idx.tolist())

            # compute the cluster means
            xc_mean = X[idx].mean(axis=0)
            cluster_centers_.append(xc_mean)

        return n_connected_components_, labels_, cluster_centers_, cluster_core_indices_

    def fit(self, X):
        """
        Apply the FINCH algorithm 
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data samples that are clustered

        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        self.n_samples = X.shape[0]

        # the results of the partitioning
        results = {}

        # intermediate results
        cluster_centers_ = None
        cluster_core_indices_ = None

        n_connected_components_ = len(X)
        if self.verbose:
            print('FINCH Partitionings')
            print('-------------------')

        i = 0
        while n_connected_components_ > 1:
            n_connected_components_, labels_, cluster_centers_, cluster_core_indices_ = self._finch(
                X, cluster_centers_, cluster_core_indices_)

            if n_connected_components_ == 1:
                break
            else:
                if self.verbose:
                    print('Clusters in %s partition: %d' %
                          (i, n_connected_components_))

            results['parition_' + str(i)] = {
                'n_clusters': n_connected_components_,
                'labels': labels_,
                'cluster_centers': cluster_centers_,
                'cluster_core_indices': cluster_core_indices_
            }
            i += 1

        self.partitions = results

        return self

    def fit_predict(self, X, verbose):
        """
        Apply the FINCH algorithm and returns a reasonable partitioning labels based on the silhouette coeffcient
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data samples that are clustered

        Returns
        -------
        self
        """
        # check if input is correct
        X = check_array(X)

        self.n_samples = X.shape[0]

        # the results of the partitioning
        results = {}

        # intermediate results
        cluster_centers_ = None
        cluster_core_indices_ = None

        # min silhouette coefficent score
        max_sil_score = -1

        n_connected_components_ = len(X)

        print('FINCH Partitionings')
        print('-------------------')

        i = 0
        while n_connected_components_ > 1:
            n_connected_components_, labels_, cluster_centers_, cluster_core_indices_ = self._finch(
                X, cluster_centers_, cluster_core_indices_)

            if n_connected_components_ == 1:
                break
            else:
                # in this version the silhouette coefficent is computed
                sil_score = silhouette_score(X, labels_, metric=self.metric)
                # store the max silhouette coefficent
                # do not pick the first partitioning
                if max_sil_score <= sil_score and i != 0:
                    best_labels = labels_
                    max_sil_score = sil_score

                print(
                    'Clusters in %s partition: %d with average silhouette score %0.2f'
                    % (i, n_connected_components_, sil_score))

            results['parition_' + str(i)] = {
                'n_clusters': n_connected_components_,
                'labels': labels_,
                'cluster_centers': cluster_centers_,
                'cluster_core_indices': cluster_core_indices_,
                'silhouette_coefficient': sil_score
            }
            i += 1

        self.labels = best_labels
        self.partitions = results

        return self.labels
