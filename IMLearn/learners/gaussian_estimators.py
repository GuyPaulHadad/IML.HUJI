from __future__ import annotations

import math

import numpy as np
import scipy
from numpy.linalg import inv, det, slogdet
from scipy.stats import *
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
        sum_of_normal_exp_power = 0
        for i in range(sample_amount):
            sum_of_normal_exp_power = sum_of_normal_exp_power - 0.5 * ((X[i] - mu) / sigma) ** 2
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
        """
        sample_amount = X.size
        dim = X[0].size
        inverted_cov = inv(cov)

        sum_of_normal_exp_power = 0

        for i in range(sample_amount):
            sum_of_normal_exp_power = sum_of_normal_exp_power - 0.5 * np.multiply((X - mu).transpose(), inverted_cov,
                                                                                  (X - mu))
            return sample_amount * math.log(
                1.0 / math.sqrt(np.linalg.det(cov) * (2 * math.pi) ** dim)) + sum_of_normal_exp_power
        raise NotImplementedError()
"""
        size = mu.size
        sample_amount = X.size
        deter = det(cov)
        if deter != 0:
            norm_const = sample_amount * math.log(
                1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(deter, 1.0 / 2)), math.e)
            x_mu = (X - mu)
            inverse = inv(cov)
            sum_exp_power = -0.5 * np.einsum('ij,jk,ki', x_mu, inverse, x_mu.T)
            return norm_const + sum_exp_power
        else:
            raise ValueError("Cov matrix determinant cannot equal zero")


if __name__ == '__main__':
    """
    normal1 = UnivariateGaussian();
    print(str(normal1.mu_) + " " + str(normal1.var_))
    numpyArr = np.random.normal(10, 1, 1000)
    normal1.fit(numpyArr)
    print("Estimated mu: " + str(normal1.mu_) + " | Estimated cov: " + str(normal1.var_))
    # normal1.pdf(normal1)
    print(numpyArr.size)
    ms = np.linspace(5, numpyArr.size, round(numpyArr.size / 5)).astype(np.int)
    absolute_mean_error = []

    for m in ms:
        absolute_mean_error.append(abs(normal1.mu_ - np.mean(numpyArr[1:m + 1])))

    go.Figure([go.Scatter(x=ms, y=absolute_mean_error, mode='markers+lines', name=r"text{$\mu-error$}"),
               go.Scatter(x=ms, y=[0] * len(ms), mode='lines', name=r"$no-error$")],
              layout=go.Layout(title=r"$\text{Absolute Mean Error as a Function of Number of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\\text{Absolute Mean Error}$",
                               height=400)).show()
    pdf_array = normal1.pdf(numpyArr)
    empirical_pdf = np.full(1000, 1 / 1000)

    go.Figure(
        [go.Scatter(x=numpyArr, y=pdf_array, mode='markers')],
        layout=go.Layout(
            title=r"$" + "\\text{Density Function of a normal distribution: }" + "\mu=" + "{:.2f}".format(
                normal1.mu_) + ", \sigma^2=" + "{:.2f}".format(normal1.var_) + "$",
            xaxis_title="$Value of Samples$",
            yaxis_title="r$\\text{Probability Density}$",
            height=400)).show()
    print(normal1.log_likelihood(10, 1, numpyArr))

    """
    mu_multivariate = np.array([0, 0, 4, 0])
    cov_multivariate = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    # print(cov_multivariate)
    multivariate_samples = np.random.multivariate_normal(mu_multivariate, cov_multivariate, 1000)
    # print(multivariate_samples)
    # multivariate_gaussian = MultivariateGaussian()
    # multivariate_gaussian.fit(multivariate_samples)
    # print(multivariate_gaussian.mu_)
    # print("")
    # print(multivariate_gaussian.cov_)
    # print(multivariate_samples[0])
    # print(multivariate_gaussian.cov_)
    # print(multivariate_gaussian.pdf(multivariate_samples[0]))
    # multi = scipy.stats.multivariate_normal(multivariate_gaussian.mu_,
    # multivariate_gaussian.cov_)
    # print(multi.pdf(multivariate_samples[0]))
    multi = MultivariateGaussian()
    # multi.mu_ = np.array([0, 0, 4, 0])
    # multi.cov_ = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multi.fit(multivariate_samples)
    # check = np.tile([0,0,4,0],(1000,1))

    # multi.fitted_ = True

    # print(multivariate_normal.pdf(multivariate_samples,multi.mu_,multi.cov_)-multi.pdf(multivariate_samples))
    """
    ls = np.linspace(-10, 10, 200)
    arr = np.zeros((200,200))
    for i in range(200):
        counter = 0
        for j in range(50):

            arr[i][counter] = ls[i]
            arr[i][counter+2] = ls[j]
            counter = counter+4
    print(arr)
    """
    # test1 = np.array([[1,2,3,4],[1,2,3,4],[4,5,6,7]])
    # test2 = np.array([[4,5,6,7],[1,2,3,4],[1,2,3,4]])
    # print(np.sum(test1*test2))
    multi_scipy = multivariate_normal(multi.mu_, multi.cov_)
    f1 = f3 = np.linspace(-10, 10, 200)

    likelihood_arr = np.zeros((200, 200))
    max_mu = np.array([0, 0, 0, 0])
    max_likelihood_value = multi.log_likelihood(max_mu, cov_multivariate, multivariate_samples)

    for i in range(200):
        for j in range(200):
            cur_mu = np.array([f1[i], 0, f3[j], 0])
            likelihood_value = multi.log_likelihood(cur_mu, multi.cov_, multivariate_samples)
            likelihood_arr[i][j] = likelihood_value
            if max_likelihood_value < likelihood_value:
                max_mu = cur_mu
                max_likelihood_value = likelihood_value

    print(max_likelihood_value)
    print(max_mu)
    fig3 = px.imshow(likelihood_arr, x=f1, y=f3, title="Log - likelihood pertaining different expectation values",
                     height=450,
                     labels=dict(x="Value of F1", y="Value of F3", color="Log - Likelihood Value"),
                     color_continuous_scale="Hot")
    plotly.offline.plot(fig3)
