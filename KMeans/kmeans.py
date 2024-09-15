import random
import numpy as np

def euc_dist(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class KMeans():
    def __init__(self,k,max_iters) -> None:
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
    
    def assign_cluster(self,x):
        distances = [euc_dist(x,c) for c in self.centroids]
        return np.argmax(distances)
    
    def fit(self,X_train):
        random_indices = [random.randrange(0,len(X_train)-1) for i in range(self.k)]
        self.centroids = np.array([X_train[i] for i in random_indices])
        for _ in range(self.max_iters):
            clusters = np.array([self.assign_cluster(x).item() for x in X_train])
            previous_centroids = self.centroids.copy()            
            centroids = np.array([X_train[clusters==i].mean(axis=0) for i in range(self.k)])
            if np.all(centroids==previous_centroids):
                break
    def get_centroids(self):
        return self.centroids

kmeans = KMeans(3,100)
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])
kmeans.fit(X)
print(kmeans.get_centroids())



