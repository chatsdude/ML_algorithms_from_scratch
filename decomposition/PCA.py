import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

class PCA():

    def __init__(self,n_dimensions):
        self.n_dimensions = n_dimensions
        self.mean = None
        self.components = None

    def fit(self, X):
        #Mean
        self.mean = np.mean(X,axis=0)
        X = X - self.mean

        #Covariance
        cov = np.cov(X.T)

        #Eigen values
        eigenvalues,eigenvectors = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        idxs = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[0:self.n_dimensions]

    def transform(self, X):
        X = X - np.mean(X,axis=0)
        return np.dot(X,self.components.T)
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
 
    #Get the IRIS dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
    
    #prepare the data
    x = data.iloc[:,0:4]
    
    #prepare the target
    target = data.iloc[:,4]
    
    #Applying it to PCA function
    mat_reduced = PCA(2).fit_transform(x)
    
    #Creating a Pandas DataFrame of reduced Dataset
    principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
    
    #Concat it with target variable to create a complete Dataset
    principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)

    plt.figure(figsize = (6,6))
    sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2' , hue = 'target' , s = 60 , palette= 'icefire')
