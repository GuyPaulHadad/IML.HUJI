from __future__ import annotations

import math

import numpy as np
import scipy
from numpy.linalg import inv, det, slogdet
from scipy.stats import *
import scipy.stats
import plotly
import plotly.express as px
from plotly import graph_objects as go


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X)
        if self.biased_ is True:
            self.var_ = X.var(ddof=1)
        else:
            self.var_ = np.var(X)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        sd = math.sqrt(self.var_)
        return (1.0 / (sd * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((X - self.mu_) / sd) ** 2)

    @staticmethod
    def normal_pdf(mu: float, var: float, value: float) -> float:
        sd = math.sqrt(var)
        return (1.0 / (sd * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((value - mu) / sd) ** 2)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        # n*log(1/(sqrt(2*pi)*sigma)) + sum(-0.5*((sample - mu)/sigma)^2)
        # log in base of e. mu - expectation. sigma - standard deviation

        sample_amount = X.size
        sum_of_normal_exp_power = np.sum(-0.5*((X-mu)/sigma)**2)

        return sample_amount * math.log((1.0 / (sigma * math.sqrt(2 * math.pi))), math.e) + sum_of_normal_exp_power


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean(0)
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        size = self.mu_.size
        deter = det(self.cov_)
        if deter != 0:
            norm_const = 1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(deter, 1.0 / 2))
            x_mu = (X - self.mu_)
            inverse = inv(self.cov_)
            product_arr = np.einsum('ij,jk,ki->i', x_mu, inverse, x_mu.T)
            result = np.exp(-0.5 * product_arr)
            return norm_const * result
        else:
            raise ValueError("Cov matrix determinant cannot equal zero")

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        size = mu.size
        sample_amount = X.size
        deter = det(cov)
        if deter != 0:
            norm_const = sample_amount * math.log(
                1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(deter, 1.0 / 2)), math.e)
            x_mu = (X - mu)
            inverse = inv(cov)
            if x_mu.ndim == 1:
                sum_exp_power = -0.5 * np.einsum('j,jk,k', x_mu, inverse, x_mu.T)
            else:
                sum_exp_power = -0.5 * np.einsum('ij,jk,ki', x_mu, inverse, x_mu.T)
            return norm_const + sum_exp_power
        else:
            raise ValueError("Cov matrix determinant cannot equal zero")



