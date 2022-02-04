import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LinearRegression:

    def __init__(self,epochs=1000,learningRate=0.0001):
        self.epochs = epochs
        self.learningRate = float(learningRate)


    def fit(self, X, y):
        self.numSamples,self.numFeatures = X.shape
        self.bias = 0
        self.thetas = np.zeros(self.numFeatures)
    
        for _ in range(self.epochs):
            y_pred = np.dot(X,self.thetas) + self.bias
            
    
            diff = y - y_pred
            partialDerivativeOfSlope = (-2//self.numSamples)*np.dot(X.T,diff)
            partialDerivativeOfBias = (-2//self.numSamples)*np.sum(diff)
            self.thetas -= self.learningRate*partialDerivativeOfSlope
            self.bias -= self.learningRate*partialDerivativeOfBias

    def predict(self,X):
        prediction = np.dot(X,self.thetas) + self.bias
        return prediction

    
    def calculateLoss(self,y_true,y_pred):
        #MSE as loss function
        return np.mean((y_true - y_pred)**2)



if __name__=="__main__":
    X,y = datasets.make_regression(n_samples=5000,n_features=7,random_state=42)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1234)
    lr = LinearRegression(epochs=5000)
    lr.fit(X_train,y_train)
    prediction = lr.predict(X_test)
    print(prediction)
    print(y_test)
    print(lr.calculateLoss(y_test, prediction))
