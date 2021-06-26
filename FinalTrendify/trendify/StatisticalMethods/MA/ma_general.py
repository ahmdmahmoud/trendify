from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import csv
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
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


def ma_forecast_single_step(history, lag, steps):
    # print(f'len of history == {len(history)}')
    model = ARIMA(history, order=(0,0,lag))
    model_fit = model.fit()
    yhat = model_fit.predict(len(history), len(history)+steps-1)
    # print(f'len of yhat == {len(yhat)}')
    return yhat

def ma_forecast_multi_step(history, lag, n_steps):
    model = ARIMA(history, order=(0,0,lag))
    model_fit = model.fit()
    yhat = model_fit.predict(len(history), len(history)+n_steps)
    return yhat


def ma_rmse(data, test_split, lag, word):
    train_split = 1 - test_split
    train_size = int(len(data)*train_split)
    train, test = data[:train_size], data[train_size:]
    
    predictions = []
    history = [x for x in train]
    # for t in range(len(test)):
    #     yhat = ma_forecast_single_step(history,lag)
    #     predictions.append(yhat)
    #     history.append(test[t])
    
    steps = 10
    for t in range(0,len(test),steps):
        if (t+steps) <= len(test):
            yhat = ma_forecast_single_step(history,lag, steps)
        else:
            yhat = ma_forecast_single_step(history,lag, len(test) - t)

        predictions.extend(yhat)
        history.extend(test[t:t+steps])
    
    # print(len(predictions), len(test))
    # predictions = ma_forecast_multi_step(history, lag, len(test)-1)
    rmse = sqrt(mean_squared_error(test, predictions))

    anomaly_counts = trends(predictions, test)

    # plt.figure(figsize=(10,4))

    # _ = plt.plot(np.array(train))
    # _ = plt.plot(np.append(np.array([np.nan]*len(train)),np.array(test)))
    # _ = plt.plot(np.append(np.array([np.nan]*len(train)),np.array(predictions)))
    # plt.xlabel('time')
    # plt.ylabel('frequency')
    # plt.title(f'word --- {word} ,  rmse --- {rmse: .2f}')
    # plt.legend(["history", "test", "prediction"])
    # plt.show()

    
    return rmse, anomaly_counts




def find_best_rmse_ma(scaled_freq_time_series, words, start, end, test_split, lags_permutations):
    RMSE = []
    anomaly = []
    for word in words[start:end]:
        data = scaled_freq_time_series[word]
        best_rmse = None
        best_lag = None
        for lag in lags_permutations:
            rmse, anomaly_counts = ma_rmse(data, test_split, lag, word)
            if best_rmse is not None:
                if best_rmse > rmse:
                    best_rmse, best_lag = rmse, lag
            else:
                best_rmse, best_lag = rmse, lag
        RMSE.append(best_rmse)
        anomaly.append(anomaly_counts)
        print(f'word === {word} , best rmse {best_rmse} , best lag {best_lag}')
        
    

    df = pd.DataFrame(list(zip(words[start:end], RMSE, anomaly)), columns=['words', 'rmse', 'trend_points'])
    
    return df

    # print(df)
    # fpath = os.path.join(OUTPUT_DIR, f'wiki_data_ma_{start}_{end}.csv')
    # df.to_csv(fpath, index=False)
    # print(f'mean RMSE  === {np.mean(RMSE): .2f}')


def calculate_ma(*args):
    params = args[0]
    logs = args[1] 
    produced_files_list = args[2]
    
    ### MA parameters ###
    lags = params['lags']
    scale = params['scaling']
    test_split = params['test_split']
    
    ### general parameters ###
    frequency_time_series_dict = params['frequency_data']
    words = list(frequency_time_series_dict.keys())
    start = params['start']
    end = params['end']
    RESULT_DIR = params['result_dir_path']
    
    scalers, scaled_freq_time_series = scaling_data(frequency_time_series_dict, scale)
    
    calculation_df = find_best_rmse_ma(scaled_freq_time_series, words, start, end, test_split, lags)

    fpath = os.path.join(RESULT_DIR, f'result_ma_{start}_{end}.csv')
    calculation_df.to_csv(fpath, index=False)
    
    produced_files_list.append(fpath)
    logs.append(f'calculate_ma for {start} to {end}  sucessfully completed...')


if __name__ == '__main__':
    pass