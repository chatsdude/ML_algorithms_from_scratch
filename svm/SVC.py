import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LinearSVC():

    def __init__(self,learning_rate=0.001,lambda_param=0.1,max_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iters = max_iters
        self.weights = None
        self.bias = None

    def fit(self,X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y<=0,-1,1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iters):
            for idx,data in enumerate(X):
                condition = y_[idx] * (np.dot(data,self.weights) - self.bias) >=1
                if condition:
                    derivative_with_respect_to_weights = 2 * self.lambda_param*self.weights
                    self.weights -= self.learning_rate*derivative_with_respect_to_weights
                else:
                    derivative_with_respect_to_weights = 2 * self.lambda_param*self.weights - y_[idx] *data
                    derivative_with_respect_to_bias = y_[idx]
                    self.weights -= self.learning_rate*derivative_with_respect_to_weights
                    self.bias-= self.learning_rate*derivative_with_respect_to_bias


        pass

    def predict(self,X):
        linear_output = np.dot(X,self.weights) - self.bias
        predictions = []
        for output in linear_output:
            if output < 1:
                predictions.append(0)
            else:
                predictions.append(1)
        return np.array(predictions)



if __name__ == '__main__':
    dataset=datasets.load_breast_cancer()
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
    lr = LinearSVC()
    lr.fit(X_train,y_train)
    res = lr.predict(X_test)
    print(res)
    print(y_test)
    acc=np.sum(res==y_test)/len(y_test)
    print(acc)
