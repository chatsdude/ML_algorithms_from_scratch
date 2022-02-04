import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LogisticRegression(object):
    def __init__(self,learningRate=0.001,epochs=1000,regularization="l2",C=0.1):
        self.learningRate = learningRate
        self.epochs = epochs
        self.regularization = regularization
        self.C = C

    def fit(self, X, y):
        self.numSamples,self.numFeatures = X.shape
        print(self.numSamples,self.numFeatures)
        self.bias = 0
        self.thetas = np.zeros(self.numFeatures)
        
        self.thetas,self.bias = self._perform_gradient_descent(X,y,self.thetas,self.bias,self.epochs)

    def predict(self, X):
        #Predict classes
        y_pred = np.dot(X,self.thetas) + self.bias
        probabilities = self._sigmoid(y_pred)
        prediction = []
        for probability in probabilities:
            if probability >= 0.5:
                prediction.append(1)
            else:
                prediction.append(0)

        return prediction

    def predict_proba(self, X):
        #Function to predict probabilities
        y_pred = np.dot(X,self.thetas) + self.bias
        return self._sigmoid(y_pred)

    def _sigmoid(self,y_pred):
        #Helper function to calculate sigmoid
        return 1.0 / ( 1 + np.exp(-y_pred))

    def _perform_gradient_descent(self,X,y,weights,bias,epochs):
        #Perform updates to weights and bias for given number of epochs
        for _ in range(epochs):
            
            #Linear model.Similar to linear regression
            y_pred = np.dot(X,weights) + bias

            #Apply sigmoid function to linear model
            probability = self._sigmoid(y_pred)
            
            diff = probability - y

            #Calculate partial derivative of weights and bias
            partialDerivativeOfSlope = (1/self.numSamples)*np.dot(X.T,diff)
            partialDerivativeOfBias = (1/self.numSamples)*np.sum(diff)

            #Update the new values
            weights -= self.learningRate*partialDerivativeOfSlope
            bias-= self.learningRate*partialDerivativeOfBias

        return weights,bias


if __name__ == '__main__':
    dataset=datasets.load_breast_cancer()
    X,y=dataset.data,dataset.target
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
    lr = LogisticRegression(epochs=10000)
    lr.fit(X_train,y_train)
    res = lr.predict(X_test)
    print(res)
    print(y_test)
    acc=np.sum(res==y_test)/len(y_test)
    print(acc)
