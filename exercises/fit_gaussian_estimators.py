from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly
import plotly.express as px
pio.templates.default = "simple_white"



def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    np.random.seed(0)
    normal1 = UnivariateGaussian();
    numpy_normal_samples = np.random.normal(10, 1, 1000)
    normal1.fit(numpy_normal_samples)
    print("("+str(normal1.mu_)+","+str(normal1.var_)+")")
    #print("Estimated mu: " + str(normal1.mu_) + " | Estimated var: " + str(normal1.var_))



    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(5, numpy_normal_samples.size, round(numpy_normal_samples.size / 5)).astype(np.int)
    absolute_mean_error = []
    for m in ms:
        absolute_mean_error.append(abs(normal1.mu_ - np.mean(numpy_normal_samples[1:m + 1])))

    go.Figure([go.Scatter(x=ms, y=absolute_mean_error, mode='markers+lines', name=r"text{$\mu-error$}"),
               go.Scatter(x=ms, y=[0] * len(ms), mode='lines', name=r"$no-error$")],
              layout=go.Layout(title=r"$\text{Absolute Mean Error as a Function of Number of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\\text{Absolute Mean Error}$",
                               height=400)).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_array = normal1.pdf(numpy_normal_samples)
    go.Figure(
        [go.Scatter(x=numpy_normal_samples, y=pdf_array, mode='markers')],
        layout=go.Layout(
            title=r"$" + "\\text{Density Function of an Estimated Normal Distribution: }" + "\mu=" + "{:.2f}".format(
                normal1.mu_) + ", \sigma^2=" + "{:.2f}".format(normal1.var_) + "$",
            xaxis_title="$Value of Samples$",
            yaxis_title="r$\\text{Probability Density}$",
            height=400)).show()
    print(normal1.log_likelihood(10, 1, numpy_normal_samples))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu_multivariate = np.array([0, 0, 4, 0])
    cov_multivariate = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    multivariate_samples = np.random.multivariate_normal(mu_multivariate, cov_multivariate, 1000)
    multi = MultivariateGaussian()
    multi.fit(multivariate_samples)
    print(multi.mu_)
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(-10, 10, 200)

    likelihood_arr = np.zeros((200, 200))
    max_likelihood_value = -100000000

    for i in range(200):
        for j in range(200):
            cur_mu = np.array([f1[i], 0, f3[j], 0])
            likelihood_value = multi.log_likelihood(cur_mu, multi.cov_, multivariate_samples)
            likelihood_arr[i][j] = likelihood_value

    fig3 = px.imshow(likelihood_arr, x=f1, y=f3, title="Log - likelihood pertaining different expectation values",
                     height=450,
                     labels=dict(x="Value of F1", y="Value of F3", color="Log - Likelihood Value"),
                     color_continuous_scale="Hot")
    plotly.offline.plot(fig3)





    # Question 6 - Maximum likelihood
    ans = np.unravel_index(np.argmax(likelihood_arr, axis=None), likelihood_arr.shape)
    print("f1: " + str(f1[ans[0]]))
    print("f3: " + str(f3[ans[1]]))
    print("Max Likelihood: "+ str(likelihood_arr[ans[0]][ans[1]]))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
