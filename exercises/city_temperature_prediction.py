import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

import math
import plotly.graph_objects as go

import plotly as pt

pio.templates.default = "simple_white"
TEST_PERCENT = 0.25
FINAL_DEG = 5


def clean_data(temp_data: pd.DataFrame) -> pd.DataFrame:
    temp_data['DayOfYear'] = temp_data['DayOfYear'].apply(lambda x: x.dayofyear)
    for greater_than_zero_feature in temp_data[['Year', 'Month', 'Day']]:
        temp_data = temp_data.drop(
            temp_data.index[temp_data[greater_than_zero_feature] <= 0])
    temp_data = temp_data.drop(temp_data.index[temp_data["Temp"] < -50])

    temp_data = temp_data.dropna(
        axis=0).reset_index(drop=True)

    return temp_data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    daily_temp_df = pd.read_csv(filename, parse_dates={'DayOfYear': ['Date']})

    daily_temp_df = clean_data(daily_temp_df)

    return daily_temp_df


def q2fig(city_info_df: pd.DataFrame):
    israel_temp_df = city_info_df[city_info_df['Country'] == "Israel"].reset_index()
    fig1 = px.scatter(israel_temp_df, x='DayOfYear', y='Temp', color='Year',
                      color_discrete_sequence=px.colors.qualitative.Vivid,
                      title="Temperature as a function of The Day of the year",
                      labels={"DayOfYear": "Day of the Year", "Temp": "Temperature"})

    fig1.update_layout(title={'x': 0.5})
    fig1.show()
    months_temp_sd_df = pd.DataFrame()
    months_temp_sd_df['Month_STD'] = (israel_temp_df.groupby('Month').agg(func=np.std)['Temp'])

    """
    for month in range(1,13):
        index_col = israel_temp_df.index[israel_temp_df["Month"]==month]
        israel_temp_df.loc[index_col,'Month_STD'] = month_std[month]
    """
    months_temp_sd_df['Month'] = pd.Series(np.linspace(0, 12, 13))
    fig2 = px.bar(months_temp_sd_df, x='Month', y='Month_STD', title="Each month's standard deviation for daily "
                                                                     "temperatures ",
                  labels={'Month_STD': "Month's Standard Deviation"})
    fig2.update_layout(title={'x': 0.5}).show()


def q3fig(city_info_df: pd.DataFrame):
    mn = city_info_df.groupby(['Country', 'Month'])['Temp'].mean().reset_index().rename(columns={'Temp': 'Temp_Mean'})

    sd = city_info_df.groupby(['Country', 'Month'])['Temp'].agg(func=np.std).reset_index().rename(
        columns={'Temp': 'Temp_SD'})
    mn["Temp_SD"] = sd['Temp_SD']
    fig = px.line(mn, x='Month', y='Temp_Mean', error_y='Temp_SD', color='Country',
                  title='Temperature mean as a function of the month')
    fig.update_layout(title={'x': 0.5}).show()


def q4fig(city_temp_df: pd.DataFrame):
    isr_temp_df = city_temp_df[city_temp_df['Country'] == "Israel"].reset_index()
    israel_df_doy = isr_temp_df.pop('DayOfYear')
    temp_df = isr_temp_df.pop('Temp')

    train, test = split_training_and_test(israel_df_doy, temp_df, TEST_PERCENT)

    results = []
    for i in range(1, 11):
        results.append(round(fit_and_calc_loss(train, test, i), 3))

    print("Test Err:" + str(results))

    ls = np.linspace(1, 10, 10)
    poly_loss_df = pd.DataFrame({"Degree": ls, "Loss": results})
    fig = px.bar(poly_loss_df, x='Degree', y='Loss', title="Loss as a function of polynomial degree k ")
    fig.update_layout(title={'x': 0.5}).show()


def split_training_and_test(data_x, pred_y, test_percent):
    data_x = data_x.sample(frac=1)
    pred_y = pred_y.reindex_like(data_x)

    n = round(test_percent * len(pred_y))
    return (data_x[:-n], pred_y[:-n]), (data_x[-n:], pred_y[-n:])


def mult_coeff(coeffs, num):
    poly = 0
    for i in range(len(coeffs)):
        poly = num ** i * coeffs[i]
    return poly


def fit_and_calc_loss(train_data, test_data, k) -> float:
    train_features = train_data[0]
    train_true_y = train_data[1]

    pf = PolynomialFitting(k)
    pf.fit(train_features.to_numpy(), train_true_y.to_numpy())
    # test_feature = test_data[0].apply(lambda x: mult_coeff(pf.coefs_,x))

    return pf._loss(test_data[0].to_numpy(), test_data[1].to_numpy())


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    city_temp_df = load_data("C:/Users/guyha/Desktop/uniCourse/Year 2/Semester B/IML/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # q2fig(city_temp_df)
    # Question 3 - Exploring differences between countries
    # q3fig(city_temp_df)

    # Question 4 - Fitting model for different values of `k`
    # q4fig(city_temp_df)

    # Question 5 - Evaluating fitted model on different countries
    israel_df = city_temp_df[city_temp_df['Country'] == "Israel"].reset_index()
    israel_df_doy = israel_df.pop('DayOfYear')
    israel_temp_df = israel_df.pop('Temp')
    pf = PolynomialFitting(FINAL_DEG)
    pf._fit(israel_df_doy, israel_temp_df)

    jordan_df = city_temp_df[city_temp_df['Country'] == "Jordan"].reset_index()
    jordan_df_doy = jordan_df.pop('DayOfYear')
    jordan_temp_df = jordan_df.pop('Temp')
    south_africa_df = city_temp_df[city_temp_df['Country'] == "South Africa"].reset_index()
    south_africa_df_doy = south_africa_df.pop('DayOfYear')
    south_africa_temp_df = south_africa_df.pop('Temp')

    nether_df = city_temp_df[city_temp_df['Country'] == "The Netherlands"].reset_index()
    nether_df_doy = nether_df.pop('DayOfYear')
    nether_temp_df = nether_df.pop('Temp')
    country_temp = [jordan_temp_df, south_africa_temp_df, nether_temp_df]
    country_doy = [jordan_df_doy, south_africa_df_doy, nether_df_doy]

    results = []
    for i in range(3):
        results.append(pf._loss(country_doy[i], country_temp[i]))

    err_country_df = pd.DataFrame({'Country': ["Jordan", "South Africa", "The Netherlands"], "Model Error": results})
    fig = px.bar(err_country_df, x='Country', y='Model Error', title="Model Error for each country")
    fig.update_layout(title={'x': 0.5}).show()

