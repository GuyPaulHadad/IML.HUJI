from __future__ import annotations
from typing import NoReturn
from IMLearn import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics import loss_functions as lf
import sklearn.metrics as met
import math


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """

        number_of_samples, number_of_features = X.shape
        if self.include_intercept_:
            X = np.hstack((np.ones((number_of_samples, 1)), X))

        self.coefs_ = np.linalg.pinv(X, 1e-15).dot(y)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return np.dot(X, self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return lf.mean_square_error(y, np.dot(X, self.coefs_))


if __name__ == '__main__':
    arr2 = np.array([1, 2, 2, 2, 2, 3])
    arr1 = np.array(
        [[1, 2, 3, 4, 4], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 7]])
    check = LinearRegression(True)
    check.fit(arr1, arr2)
    n, m = arr1.shape

    arr1 = np.hstack((np.ones((n, 1)), arr1))
    print(check._predict(arr1))
    print(arr2)
    one = np.linalg.lstsq(arr1, arr2, rcond=None)[0]
    print(check.loss(arr1, arr2))

"""

    number_of_features = arr1.shape[1]
    x_rank = np.linalg.matrix_rank(arr1)

    u, sigma, v_transpose = np.linalg.svd(arr1, full_matrices=False)

    v = v_transpose.T

    sigma_dagger = np.diag(np.hstack([1 / sigma[:x_rank], np.zeros(number_of_features - x_rank)]))
    x_dagger = np.dot(np.dot(v, sigma_dagger), u.T)
    print("coef")
    print(check.coefs_)
    # print(np.linalg.pinv(arr1,1e-15).dot(arr2))
    
    print(np.linalg.lstsq(arr1, arr2, rcond=None))
"""
