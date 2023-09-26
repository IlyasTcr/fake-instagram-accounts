import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Define a ratio transformer class
class RatioTransformer(BaseEstimator, TransformerMixin):
    # Initialize the class with the numerator and denominator
    def __init__(self, numerator, denominator, epsilon=1e-7):
        self.numerator = numerator
        self.denominator = denominator
        self.epsilon = epsilon

    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self

    # Transform method to return the ratio  
    def transform(self, X):
        ratio = X[self.numerator] / (X[self.denominator] + self.epsilon)
        return ratio.to_numpy().reshape(-1, 1)


# Define a PCA transformer class
class PCATransformer(BaseEstimator, TransformerMixin):
    # Initialize the class with a standardscaler and pca
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.pca = PCA()

    # Fit method for pca (with scaled inputs)
    def fit(self, X, y=None):
        self.pca.fit(self.standard_scaler.fit_transform(X))
        return self

    # Transform method to return the principal components 
    def transform(self, X):
        return self.pca.transform(self.standard_scaler.transform(X))


# Define a boolean feature aggregator transformer class
class BooleanFeatureAggregator(BaseEstimator, TransformerMixin):
    # Initialize the class with a list of boolean features
    def __init__(self, boolean_features):
        self.boolean_features = boolean_features

    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self

    # Transform method to return a cumulative count of the boolean attributes for each individual row. 
    def transform(self, X):
        sums = X[self.boolean_features].sum(axis=1)
        return sums.to_numpy().reshape(-1, 1)