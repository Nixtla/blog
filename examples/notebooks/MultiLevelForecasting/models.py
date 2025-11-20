import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def split_timeseries(data, val_size=0.1, test_size=0.1):
    """
    Splits a 1D timeseries into train/val/test sequentially.
    """
    N = len(data)
    test_len = int(N * test_size)
    val_len = int(N * val_size)
    train_len = N - val_len - test_len
    
    train = data[:train_len]
    val = data[train_len:train_len + val_len]
    test = data[train_len + val_len:]
    
    return train, val, test

def polynomial_fit_and_select(train, val, max_degree=5):
    """
    Trains polynomial regressors of increasing degree and selects best on val set.
    """
    best_degree = None
    best_model = None
    lowest_mse = float('inf')
    
    x_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train
    x_val = np.arange(len(train), len(train) + len(val)).reshape(-1, 1)
    y_val = val

    for degree in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(x_train)
        X_val_poly = poly.transform(x_val)
        
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred_val = model.predict(X_val_poly)
        mse = mean_squared_error(y_val, y_pred_val)
        
        print(f"Degree {degree} - Val MSE: {mse:.4f}")
        if mse < lowest_mse:
            lowest_mse = mse
            best_degree = degree
            best_model = (model, poly)

    return best_model, best_degree

def evaluate_on_test(model_tuple, train_len, val_len, test):
    """
    Evaluates the best model on the test set.
    """
    model, poly = model_tuple
    x_test = np.arange(train_len + val_len, train_len + val_len + len(test)).reshape(-1, 1)
    X_test_poly = poly.transform(x_test)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(test, y_pred)
    print(f"Test MSE (degree {poly.degree}): {mse:.4f}")
    return y_pred
