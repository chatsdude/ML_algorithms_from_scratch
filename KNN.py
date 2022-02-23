import numpy as np
class KNN():
    def __init__(self,k):
        self.k=k
    def fit(self,X,y):
        self.features,self.target=X,y
    def predict(self,X):
        predicted_labels=[self.predict_helper(point) for point in X]
        return np.array(predicted_labels)
    def predict_helper(self,x):
        import heapq
        from collections import Counter
        distance,heap= lambda x,y:np.sqrt(np.sum((x-y)**2)),[]
        for coord in range(len(self.features)):
            heapq.heappush(heap,(distance(x,self.features[coord]),self.target[coord]))
        heap=heapq.nsmallest(self.k,heap)
        results=[heap[i][1] for i in range(len(heap))]
        return Counter(results).most_common(1)[0][0]

        
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
knn=KNN(5)
#labels=np.array(['2','1','0','2'])
knn.fit(X_train,y_train)
res=knn.predict(X_test)
acc=np.sum(res==y_test)/len(y_test)
print(acc)



