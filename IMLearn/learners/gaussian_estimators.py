from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet
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
        pdf_array = np.empty(1000)
        for index in range(X.size):
            pdf_array[index] = self.normal_pdf(self.mu_, self.var_, X[index])

        return pdf_array

    @staticmethod
    def general_pdf(X: np.ndarray) -> np.ndarray:
        mu = np.mean(X)
        var = np.var(X)
        pdf_array = np.empty(1000)
        for index in range(X.size):
            pdf_array[index] = UnivariateGaussian.normal_pdf(mu, var, X[index])

        return pdf_array

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
        raise NotImplementedError()


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
        raise NotImplementedError()
        self.mu_ = np.mean(X)
        self.cov_ = np.var(X)
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
        raise NotImplementedError()

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
        raise NotImplementedError()


if __name__ == '__main__':
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
            xaxis_title="$X$",
            yaxis_title="r$\\text{Probability Density}$",
            height=400)).show()
"""
    X = np.linspace(6, 14, numpyArr.size).astype(np.int)
    theoretical_dist_m = UnivariateGaussian.general_pdf(X)

    go.Figure([go.Histogram(x=numpyArr, opacity=0.75, bingroup=1, histnorm='probability density',
                            marker_color="rgb(0,124,134)", name=r'$\hat\mu$'),

               go.Scatter(x=X, y=theoretical_dist_m, mode='lines', line=dict(width=4, color="rgb(204,68,83)"),
                          name=r'$N(\mu, \frac{\sigma^2}{m1})$')],
              layout=go.Layout(barmode='overlay',
                               title=r"$\text{(8) Mean estimator distribution}$",
                               xaxis_title="r$\hat\mu$",
                               yaxis_title="density",
                               height=300)).show()

"""
