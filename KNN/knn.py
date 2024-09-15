import numpy as np

def euc_dist(x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))

class KNN():
    def __init__(self,k=3):
        self.k=k
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def labels(self,x):
        distance = [euc_dist(x_tr,x) for x_tr in self.X_train]
        top_k_indices = np.argsort(distance)[:self.k]
        top_k_label = [self.y_train[i] for i in top_k_indices]
        label_count = {}
        max_label = -1
        max_label_count = -1
        for label in top_k_label:
            if label not in label_count.keys():
                label_count[label]=1
            else:
                label_count[label]+=1
        for label in label_count.keys():
            if label_count[label]>max_label_count:
                max_label_count = label_count[label]
                max_label = label
        return max_label
    
    def predict(self,X_test):
        return [self.labels(x) for x in X_test]
        
knn = KNN(3)
X_train = np.array([[0,0],[0,1],[1,0],[1,1],[0.5,0]])
y_train=np.array([0,1,1,1,0])
knn.fit(X_train,y_train)
X_test = X_train.tolist().copy()
X_test.append([0.7,0])
X_test.append([0.8,0])
X_test.append([0,0.6])
X_test = np.array(X_test)
y_pred = knn.predict(X_test)
print(y_pred)
