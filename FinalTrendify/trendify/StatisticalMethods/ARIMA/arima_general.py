import csv
import numpy as np
from numpy.core.fromnumeric import repeat
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def scaling_data(frequency_time_series_dictionary, scale):
    scaled_freq_time_series = {}
    scalers = {}
    for word in frequency_time_series_dictionary:
        data = np.array(frequency_time_series_dictionary[word])
        scaler = MinMaxScaler(feature_range=scale)
        transformed = scaler.fit_transform(data.reshape(-1,1))
        scalers[word] = scaler
        scaled_freq_time_series[word] = np.reshape(transformed, len(transformed))

    return scalers, scaled_freq_time_series


def trends(fc, actual):
    df = pd.DataFrame(list(zip(actual, fc)), columns=['actual', 'forecast'])
    df['error'] = df['actual'] - df['forecast']
    df['percentage_change'] = ((df['actual'] - df['forecast']) / df['actual']) * 100
    df['meanval'] = df['error'].rolling(window=24).mean()
    df['deviation'] = df['error'].rolling(window=24).std()
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    cut_list = df[['error', '-3s', '-2s', '-1s', 'meanval', '1s', '2s', '3s']]
    cut_values = cut_list.values
    cut_sort = np.sort(cut_values)
    df['impact'] = [(lambda x: np.where(cut_sort == df['error'][x])[1][0])(x) for x in
                            range(len(df['error']))]
    severity = {0: 3, 1: 2, 2: 1, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3}
    region = {0: "NEGATIVE", 1: "NEGATIVE", 2: "NEGATIVE", 3: "NEGATIVE", 4: "POSITIVE", 5: "POSITIVE", 6: "POSITIVE",
            7: "POSITIVE"}
    df['color'] =  df['impact'].map(severity)
    df['region'] = df['impact'].map(region)
    df['anomaly_points'] = np.where(df['color'] == 3, df['error'], np.nan)

    df.dropna(axis=0,inplace=True)
    
    flag = False
    # if df.shape[0] > 10:
        # flag = True

    # df.to_csv('temp-anomaly-class.csv')

    # print("this is trends classify df")
    
    # print(df)

    # return flag
    return df.shape[0]


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def get_arima_model_rmse(data, p_d_q, test_split, diff_interval):
    train_split = 1 - test_split
    train_size = int(train_split*len(data))
    train, test = data[0:train_size], data[train_size:]
    history = [x for x in train]
    predictions = []

    diff_history = difference(history, diff_interval)

    # for t in range(len(test)):
    #     model = ARIMA(diff_history, order=p_d_q)
    #     model_fit = model.fit()
    #     yhat = model_fit.forecast()[0]
    #     inverse_yhat = inverse_difference(history, yhat, 12)
    #     predictions.append(inverse_yhat)
    #     history.append(test[t])
    # print(f'{p_d_q} == {sqrt(mean_squared_error(test,predictions))}')
    
    p,d,q = p_d_q

    model = ARIMA(diff_history, order=p_d_q)
    model_fit = model.fit()
    f_cast = model_fit.forecast(len(test))
    for yhat in f_cast:
        inverted = inverse_difference(history, yhat, diff_interval)
        history.append(inverted)
        predictions.append(inverted)

    rmse = np.sqrt(np.mean((np.array(test)-np.array(predictions))**2))

    anomaly_counts = trends(predictions, test)

    
    # print(f'{p_d_q} ======== {rmse:.2f}')

    # print(test)
    # print(predictions)

    # print(np.array(test) - np.array(predictions))

    # plt.figure(figsize=(10,5))

    # _ = plt.plot(np.array(train))
    # _ = plt.plot(np.append(np.array([np.nan]*len(train)),np.array(test)))
    # _ = plt.plot(np.append(np.array([np.nan]*len(train)),np.array(predictions)))
    # plt.xlabel('time')
    # plt.ylabel('frequency')
    # plt.legend(["history", "test", "prediction"])
    # fpath = os.path.join(BASE_DIR, f'wiki_sarima/word_{word}_sarima.png')
    # plt.savefig(fpath)
    # plt.show()
    
    
    return rmse, anomaly_counts
    # return sqrt(mean_squared_error(test,predictions))


# H0: It is non stationary
# H1: It is stationary
def adfuller_test(data):
    test_result = adfuller(data)
    
    if test_result[1] <= 0.05:
        # print(f'H1: It is stationary')
        # labels = ['ADF Test Statistic', 'p-value', '#Lags used', 'Number of observations used']
        # for value, label in zip(test_result, labels):
            # print(f'{label} -- {value}')
        return True
    else:
        # print(f'H0: It is non stationary')
        return False

def stationarize(data):
    df = pd.DataFrame(data)
    for s in range(12):
        # print(f'word --- {word} --- Results for shifts = {s}')
        if adfuller_test(df.shift(s).dropna()):
            # print()
            return s
        else:
            continue



def find_best_rmse_ARIMA_model(scaled_freq_time_series, words, start, end, test_split, p_d_q_permutation, diff_interval):
    RMSE, anomaly_points = [], []
    for word in words[start : end]:
        data = scaled_freq_time_series[word]

        best_rmse, best_pdq = None, None
        for p_d_q in p_d_q_permutation:
            rmse, anomaly_counts = get_arima_model_rmse(data, p_d_q, test_split, diff_interval) 
            if best_rmse is not None:
                if best_rmse > rmse:
                    best_rmse = rmse
                    best_pdq = p_d_q
            else:
                best_rmse = rmse
                best_pdq = p_d_q

        print(f'word === {word}  , best_rmse === {best_rmse} , best-pdq  === {best_pdq}')    
        RMSE.append(rmse)
        anomaly_points.append(anomaly_counts)

    df = pd.DataFrame(list(zip(words[start : end], RMSE, anomaly_points)), columns=['words', 'rmse', 'trend_points'])
    
    return df


def calculate_arima(*args):
    params = args[0]
    logs = args[1]
    produced_files_list = args[2] 
    
    ### ARIMA parameters ###
    p_p = params['p']
    p_d = params['d']
    p_q = params['q']
    repeat_interval = params['repeate_interval']
    scale = params['scaling']
    test_split = params['test_split']
    
    ### general parameters ###
    frequency_time_series_dict = params['frequency_data']
    words = list(frequency_time_series_dict.keys())
    start = params['start']
    end = params['end']
    RESULT_DIR = params['result_dir_path']
    
    scalers, scaled_freq_time_series = scaling_data(frequency_time_series_dict, scale)
    
    p_d_q_permutation = list(itertools.product(p_p,p_d,p_q))

    calculation_df = find_best_rmse_ARIMA_model(scaled_freq_time_series, words, start, end, test_split, p_d_q_permutation, repeat_interval)

    fpath = os.path.join(RESULT_DIR, f'result_arima_{start}_{end}.csv')
    calculation_df.to_csv(fpath, index=False)
    
    produced_files_list.append(fpath)
    logs.append(f'calculate_arima for {start} to {end} sucessfully completed...')

if __name__ == '__main__':
    pass