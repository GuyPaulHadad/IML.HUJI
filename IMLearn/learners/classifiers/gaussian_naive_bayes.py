from typing import NoReturn
from IMLearn import BaseEstimator
import numpy as np
from IMLearn.learners import MultivariateGaussian
from IMLearn.metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ = np.unique(y)
        m, d, k = X.shape[0], X.shape[1], self.classes_.shape[0]
        self.vars_ = np.zeros((k, d))
        self.mu_ = np.zeros((k, d))
        self.pi_ = np.zeros(k)
        for i, cls in enumerate(self.classes_):
            self.pi_[i] = (y == cls).mean()
            self.mu_[i] = X[y == cls].mean(axis=0)
            self.vars_[i] = X[y == cls].var(axis=0)
        self.fitted_ = True

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
        mg = MultivariateGaussian()
        mg.fitted_ = True
        sample_amount = X.shape[0]
        classified_arr = np.zeros(sample_amount)
        for sample_index, sample in enumerate(X):
            max_val = float('-inf')
            correct_cls = 0
            for index, cls in enumerate(self.classes_):
                likelihood_val = np.log(self.pi_[index]) + mg.log_likelihood(self.mu_[index],
                                                                             np.diag(self.vars_[index]), sample)

                if max_val < likelihood_val:
                    correct_cls = cls
                    max_val = likelihood_val
            classified_arr[sample_index] = correct_cls
        return classified_arr

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        mg = MultivariateGaussian()
        mg.fitted_ = True
        sample_amount = X.shape[0]
        likelihood_matrix = np.zeros((sample_amount, self.classes_.size))
        for sample_index, sample in enumerate(X):
            for index, cls in enumerate(self.classes_):
                likelihood_matrix[sample_index][index] = np.log(self.pi_[index]) + mg.log_likelihood(self.mu_[index],
                                                                                                     np.diag(self.vars_[
                                                                                                                 index]),
                                                                                                     sample)
        return likelihood_matrix

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `loss` function")
        return misclassification_error(y, self.predict(X))


if __name__ == '__main__':
    X = np.array([[1, -1], [2, 2], [5, 5], [6, 7], [19, 23], [27, 28]])
    y = np.array([1, 1, -1, -1, 2, 2])
    model = GaussianNaiveBayes().fit(X, y)
    # print(model.predict(X))
    print(model.likelihood(X))
    print(model._loss(X, y))
