from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly as pt
from math import atan2, pi

pio.templates.default = "simple_white"
from IMLearn.metrics import loss_functions as lf


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :-1], data[:, -1]

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")
def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        X, y = load_dataset(f"C:/Users/guyha/Desktop/uniCourse/Year 2/Semester B/IML/datasets/{f}")
        losses = []

        def callback(perc: Perceptron, x, num):
            loss = perc._loss(X, y)
            losses.append(loss)

        model = Perceptron(callback=callback).fit(X, y)
        ms = np.linspace(1, len(losses), len(losses))
        go.Figure([go.Scatter(x=ms, y=losses, mode='markers+lines', name="Loss Value"),
                   go.Scatter(x=ms, y=[0] * len(ms), mode='lines', name=r"$no-error$")],
                  layout=go.Layout(title=f"{n} - Loss as a function of Number of Iterations",
                                   xaxis_title="Number Of Iterations",
                                   yaxis_title="Loss",
                                   height=400)).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"C:/Users/guyha/Desktop/uniCourse/Year 2/Semester B/IML/datasets/{f}")

        # Fit models and predict over training set
        lda_model = LDA().fit(X, y)
        gnb_model = GaussianNaiveBayes().fit(X, y)
        lda_pred = lda_model.predict(X)
        gnb_pred = gnb_model.predict(X)
        fig = make_subplots(rows=1, cols=2, subplot_titles=
        (f + " GNB prediction - accuracy:" + str(lf.accuracy(y, gnb_pred)),
         f + " LDA prediction - accuracy:" + str(lf.accuracy(y, lda_pred))))

        fig.add_trace(go.Scatter(x= X.T[0], y=X.T[1], mode="markers", marker = go.scatter.Marker(color=lda_pred,symbol=y)),row=1,col=2)
        fig.add_trace(go.Scatter(x= X.T[0], y=X.T[1], mode="markers", marker = go.scatter.Marker(color=gnb_pred,symbol=y)),row=1,col=1)

        fig.add_trace(go.Scatter(x=lda_model.mu_.T[0], y=lda_model.mu_.T[1], mode='markers', marker=
        go.scatter.Marker(color='black', symbol='x', size=15)), row=1, col=2)
        fig.add_trace(go.Scatter(x=gnb_model.mu_.T[0], y=gnb_model.mu_.T[1], mode='markers', marker=
        go.scatter.Marker(color='black', symbol='x', size=15)), row=1, col=1)


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        for i in range(3):
            fig.add_trace(get_ellipse(lda_model.mu_[i],lda_model.cov_),row=1,col=2)
            fig.add_trace(get_ellipse(gnb_model.mu_[i],np.diag(gnb_model.vars_[i])),row=1,col=1)
        pt.offline.plot(fig)


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    #compare_gaussian_classifiers()
