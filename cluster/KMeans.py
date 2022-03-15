import numpy as np

class KMeans(object):

    def __init__(self,n_clusters=3,max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        pass

    def fit_predict(self,X):

        self.n_samples,self.n_features = X.shape

        #initialize clusters
        random_sample_idxs = np.random.choice(self.n_samples,self.n_clusters,replace=False)
        self.centroids = [X[idx] for idx in random_sample_idxs]

        for _ in range(self.max_iter):
            #update clusters
            self.clusters = self._create_clusters(X,self.centroids)

            centroids_old = self.centroids

            #Update centroids
            self.centroids = self._get_centroids_for_this_cluster(X,self.clusters)

            #Check if converged
            if self.is_converged(centroids_old,self.centroids):
                break
        
        
        return self._get_cluster_labels(self.clusters)

    def _calculate_euclidean_distance(self,x,y):
        return np.sqrt(np.sum((x-y)**2))

    def _create_clusters(self,X,centroids):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx,sample in enumerate(X):
            centroid_idx = self._find_closest_sample(sample,centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    def _find_closest_sample(self,sample,centroids):
        distances = [self._calculate_euclidean_distance(sample,point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids_for_this_cluster(self,X,clusters):
        centroids = np.zeros((self.n_clusters,self.n_features))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean = np.mean(X[cluster],axis=0)
            centroids[cluster_idx] = cluster_mean
        
        return centroids

    def is_converged(self,old_centroids,new_centroids):
        distances = [self._calculate_euclidean_distance(old_centroids[i],new_centroids[i]) for i in range(self.n_clusters)]
        return np.sum(distances)==0

    def _get_cluster_labels(self,clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx,cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

if __name__ == '__main__':
    from sklearn import datasets
    iris=datasets.load_iris()
    X,_=iris.data,iris.target
    kmeans = KMeans()
    clusterlabels = kmeans.fit_predict(X)
    print(clusterlabels)
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)