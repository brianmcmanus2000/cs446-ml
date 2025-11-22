import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

class ModelPipeline:
    def __init__(self, degree, model, scaler):
        self.degree = degree
        self.model = model
        self.scaler = scaler
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

def create_polynomial_features(X, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    return X_poly