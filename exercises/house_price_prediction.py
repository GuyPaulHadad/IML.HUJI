from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly as pt
import math

pio.templates.default = "simple_white"

NON_NEGATIVE = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
                'grade,sqft_above', 'sqft_basement', 'yr_built', "yr_renovated", 'sqft_living15', 'sqft_lot15']
NON_ZERO = ['bedrooms', 'floors', 'condition', 'sqft_living', 'sqft_living15', 'yr_built', 'price']


def renovate_year_value_normalize(year):
    if year == 0:
        return 0
    elif year < 2000:
        return (year % 100) / 10
    else:
        return (year % 100) + 10


def clean_data(house_information_df: pd.DataFrame):
    # clean negative and non zero
    house_information_df = house_information_df.drop(
        house_information_df.index[house_information_df['price'] <= 0])
    for non_negative_feature in house_information_df[
        ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
         'grade', 'sqft_above', 'sqft_basement', 'yr_built', "yr_renovated", 'sqft_living15', 'sqft_lot15']]:
        house_information_df = house_information_df.drop(
            house_information_df.index[house_information_df[non_negative_feature] < 0])
    for greater_than_zero_feature in house_information_df[
        ['bedrooms', 'floors', 'condition', 'sqft_living', 'sqft_living15', 'yr_built', 'price']]:
        house_information_df = house_information_df.drop(
            house_information_df.index[house_information_df[greater_than_zero_feature] == 0])

    # drop nan, id, and reset index
    house_information_df = house_information_df.drop(['id'], axis=1).dropna(
        axis=0).reset_index(drop=True)

    # Dropping date and long due to very low correlation (less than 0.004)
    house_information_df = house_information_df.drop(['date'], axis=1)

    max_bedrooms = 15
    house_information_df = house_information_df.drop(
        house_information_df.index[house_information_df['bedrooms'] > max_bedrooms])

    # remove sqft_living outlier
    max_sqft_living = 13000
    house_information_df = house_information_df.drop(
        house_information_df.index[house_information_df['sqft_living'] > max_sqft_living])

    # normalize yr_renovated

    house_information_df['yr_renovated'] = house_information_df['yr_renovated'].apply(
        lambda x: renovate_year_value_normalize(x))

    """
    # drop zip code -
    # EXPLANATION OF ZIP CODE - NOT NECESSARY TO READ
    # a 5 digit zip code represents the general location of the estate. While the first number
    # represents the state in which the estate is located, the second and third number represent a smaller section
    # in the country which the estate is in. Lastly the fourth and fifth specify the location and narrow it down
    # even more.
    # END OF ZIP CODE EXPLANATION
    # Their are two options I saw fit - The first - to use the third,fourth and fifth numbers as features
    # with 1 representing a house in that zone (Note that the first and second numbers are the same for every location)
    # this would add 30 features as we have numbers from 0-9 for three options.
    # Note: We don't need the first and second because they are the same for all the samples.
    # The second - latitude and longitude represent location much better than zip code. we are basically using two
    # different location features (zip and lat/long) where lat/long are much more specific.
    # Because of this I decided to remove the zip and use only the the lat/long
    """
    house_information_df = house_information_df.drop(['zipcode'], axis=1)
    """
    fig = go.Figure()

    for col in ['bathrooms', 'floors', 'waterfront', 'view', 'condition',
                'grade']:
        fig.add_trace(go.Box(y=house_information_df[col].values, name=house_information_df[col].name))
    pt.offline.plot(fig)
    """
    """
    fig = go.Figure()
    for col in ['yr_built', "yr_renovated",'sqft_above', 'sqft_basement']:
        fig.add_trace(go.Box(y=house_information_df[col].values, name=house_information_df[col].name))
    pt.offline.plot(fig)
    
    fig = go.Figure()
    fig.add_trace(go.Box(y=house_information_df["sqft_living"].values, name=house_information_df["sqft_living"].name))
    pt.offline.plot(fig)
    """

    return house_information_df


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    house_information_df = pd.read_csv(filename)
    house_information_df = clean_data(house_information_df)

    return house_information_df


def calc_pearson_corr(x: pd.Series, y: pd.Series) -> float:
    co_matrix = np.cov(x, y)
    co_x_y = (co_matrix[0][1])
    x_sd = math.sqrt(co_matrix[0][0])
    y_sd = math.sqrt(co_matrix[1][1])

    return co_x_y / (x_sd * y_sd)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    #loop that check correlation for most features
    for feature in X[
        ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
         'grade', 'sqft_above', 'sqft_basement', 'yr_built', "yr_renovated", 'sqft_living15', 'sqft_lot15', 'lat',
         'long']]:
        print(feature + ": " + str(calc_pearson_corr(X[feature], y)))

    fig = go.Figure(
        [go.Scatter(x=X["sqft_living"], y=y, mode='markers')],
        layout=go.Layout(
            title="Correlation equals " + str(calc_pearson_corr(X['sqft_living'], y))[
                                          :5] + " between house sqft & price",
            title_x=0.5, xaxis_title="House square ft living space",
            yaxis_title="House Price",
            height=400))

    pt.offline.plot(fig)
    fig2 = go.Figure(
        [go.Scatter(x=X["long"], y=y, mode='markers')],
        layout=go.Layout(
            title="Correlation equals " + str(calc_pearson_corr(X['long'], y))[:5] + " between longitude & price",
            title_x=0.5, xaxis_title="Longitude",
            yaxis_title="House Price",
            height=400))

    pt.offline.plot(fig2)
    """
    fig2 = go.Figure(
        [go.Scatter(x=X["date"], y=y, mode='markers')],
        layout=go.Layout(
            title="Correlation equals " + str(calc_pearson_corr(X['date'], y))[:5] + " between date & price",
            title_x=0.5, xaxis_title="Date",
            yaxis_title="House Price",
            height=400))

    pt.offline.plot(fig2)
    """


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    house_data = load_data(
        "C:/Users/guyha/Desktop/uniCourse/Year 2/Semester B/IML/datasets/house_prices.csv")
    # house_prices_series = house_information_df.pop("price").squeeze()
    price_series = house_data['price']
    # Question 2 - Feature evaluation with respect to response
    #feature_evaluation(house_data, price_series)

    # Question 3 - Split samples into training- and testing sets.
    train = house_data.sample(frac=0.75, random_state=0)
    test_data = house_data.drop(train.index)
    test_price = test_data.pop('price').squeeze()
    test_data = test_data.to_numpy()

    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))
    training_percent = np.linspace(10, 100, 91)

    loss_mean_arr = np.zeros(91)
    loss_std_arr = np.zeros(91)
    for training_per in range(10, 101):

        current_percent_loss_values = np.zeros(10)
        for index in range(10):
            train_group_percent = train.sample(frac=training_per / 100)
            train_group_percent_price = train_group_percent.pop('price').squeeze()
            lr = LinearRegression()
            lr._fit(train_group_percent.to_numpy(), train_group_percent_price)
            current_percent_loss_values[index] = lr._loss(test_data, test_price)

        loss_mean_arr[training_per - 10] = np.mean(current_percent_loss_values)
        loss_std_arr[training_per - 10] = np.std(current_percent_loss_values)

    fig = go.Figure(
        [go.Scatter(x=training_percent, y=loss_mean_arr, mode='markers+lines', showlegend=False), \
         go.Scatter(name='Upper Bound', x=training_percent, y=loss_mean_arr + 2 * loss_std_arr, mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0), showlegend=False), \
         go.Scatter(name='Lower Bound', x=training_percent, y=loss_mean_arr - 2 * loss_std_arr, mode='lines',
                    marker=dict(color="#444"), line=dict(width=0), showlegend=False, fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty')])

    fig.update_layout(title="Average Loss as function of the Training Size (in %)",
                      title_x=0.5, xaxis_title="Percentage of used test samples",
                      yaxis_title="Average Loss",
                      height=400)
    pt.offline.plot(fig)

    """
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()

    """
