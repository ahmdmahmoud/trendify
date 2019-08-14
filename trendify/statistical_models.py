import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from pmdarima.arima import auto_arima





# dd = array / list of train time series data
def naive_method(X, y):
    dd = np.asarray(X)
    y_hat = dd[len(dd)-1]
    #print ("Naive gold/pred", y, y_hat)
    return [y_hat]



def simple_average_method(X,y):
    dd = np.asarray(X)
    y_hat = dd.mean()
    #print ("Simple average gold/pred", y, y_hat)
    return [y_hat]


def moving_average_method(X,y):
    dd = pd.Series(X)
    # df = pd.DataFrame({'A': X})
    # dd = df["A"]
    y_hat = dd.rolling(2).mean().iloc[-1]
    #print ("Moving average gold/pred", y, y_hat)
    return [y_hat]


def simple_exponential_smoothing(X,y):
    fit2 = SimpleExpSmoothing(np.asarray(X)).fit(smoothing_level=0.6,optimized=False)
    y_hat = fit2.forecast(len(y))
    #print ("Simple exponential smoothing gold/pred", y, y_hat)
    return y_hat


def holt_linear_trend_method(X,y):
    fit1 = Holt(np.asarray(X,dtype=np.float64)).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    y_hat = fit1.forecast(len(y))
    #print ("Holt linear trend gold/pred", y, y_hat)
    return y_hat



def holt_winters_method(X,y):
    fit1 = ExponentialSmoothing(np.asarray(X,dtype=np.float64) ,seasonal_periods=2 ,trend='add', seasonal='add',).fit()
    y_hat = fit1.forecast(len(y))
    #print ("Holt winters gold/pred", y, y_hat)
    return y_hat


def arima(X,y):
    # fit1 = sm.tsa.statespace.SARIMAX(X, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
    # y_hat = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
    fit1 = auto_arima(X, suppress_warnings=True)
    y_hat = fit1.predict(n_periods=len(y), return_conf_int=False)
    #print ("ARIMA gold/pred", y, y_hat)
    return y_hat

# samples = [
#     ([27],[36, 27, 45, 63, 27]),
#     ([0],[27, 45, 63, 27, 27]),
#     ([0],[45, 63, 27, 27, 0]),
#     ([45],[63, 27, 27, 0, 0]),
#     ([36],[27, 27, 0, 0, 45]),
#     ([100],[0, 0, 45, 36, 36]),
#     ([54],[36, 36, 100, 54, 45])]

# for sample in samples:
#     X = sample[1]
#     y = sample[0]
#     #naive_method(X,y)
#     moving_average_method(X,y)

# X = samples[0][1]
# y = samples[0][0]
# naive_method(X,y)
# simple_average_method(X,y)
# moving_average_method(X,y)
# simple_exponential_smoothing(X,y)
# holt_linear_trend_method(X,y)
# holt_winters_method(X,y)
# arima(X,y)