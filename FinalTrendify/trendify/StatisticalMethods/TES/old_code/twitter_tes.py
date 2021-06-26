from math import sqrt
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Data/twitter-data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/twitter-results/TES')

def load_frequency_time_series_csv(csv_file_path):
    with open(csv_file_path, mode='r',encoding="utf-8") as infile:
        # csv_reader = csv.reader(infile, delimiter=',')
        frequency_time_series_dict = {}
        index = 0
        for line in infile.readlines():
            split = line.split(",")
            entity = split[0].strip()
            try:
                frequencies = [int(x.strip()) for x in split[1:]]
                frequency_time_series_dict[entity] = frequencies
            except:
                print ("error")
            index += 1
    return frequency_time_series_dict

twitter_data_filepath = os.path.join(INPUT_DIR, "frequency_time_series_data.csv")


frequency_time_series_dictionary =  load_frequency_time_series_csv(twitter_data_filepath)

min_val, max_val = np.inf, 0
for word in frequency_time_series_dictionary:

    if np.min(frequency_time_series_dictionary[word]) < min_val:
        min_val = np.min(frequency_time_series_dictionary[word])
    if np.max(frequency_time_series_dictionary[word]) > max_val:
        max_val = np.max(frequency_time_series_dictionary[word])

print(min_val, max_val)

def scaling_data(frequency_time_series_dictionary):
    scaled_freq_time_series = {}
    scalers = {}
    for word in frequency_time_series_dictionary:
        data = np.array(frequency_time_series_dictionary[word])
        scaler = MinMaxScaler(feature_range=(1,100))
        transformed = scaler.fit_transform(data.reshape(-1,1))
        scalers[word] = scaler
        scaled_freq_time_series[word] = np.reshape(transformed, len(transformed))

    return scalers, scaled_freq_time_series

scalers, scaled_freq_time_series = scaling_data(frequency_time_series_dictionary)


t, d, s, s_p = ['add'], [True], ['mul'], [24]
t_d_s_sp_permutations  = list(itertools.product(t,d,s,s_p))
def exp_smoothing_forecast_single_step(history, cfg):
    (t,d,s,s_p) = cfg
    model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=s_p)
    model_fit = model.fit(optimized=True)
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]
def exp_smoothing_forecast_multi_step(history, cfg, n_steps):
    (t,d,s,s_p) = cfg
    model = ExponentialSmoothing(history, trend=t, damped_trend=d, seasonal=s, seasonal_periods=s_p)
    model_fit = model.fit(optimized=True)
    yhat = model_fit.predict(len(history), len(history)+n_steps)
    return yhat


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
        


def exp_smoothing_rmse(data, test_split, cfg, word):
    train_split = 1 - test_split
    train_size = int(len(data)*train_split)
    train, test = data[:train_size], data[train_size:]
    
    predictions = []
    history = [x for x in train]
    # for t in range(len(test)):
    #     yhat = exp_smoothing_forecast_single_step(history,cfg)
    #     predictions.append(yhat)
    #     history.append(test[t])
    
    predictions = exp_smoothing_forecast_multi_step(history, cfg, len(test)-1)
    rmse = sqrt(mean_squared_error(test, predictions))

    anomaly_counts = trends(predictions, test)

    # plt.figure(figsize=(20,8))

    # _ = plt.plot(np.array(train))
    # _ = plt.plot(np.append(np.array([np.nan]*len(train)),np.array(test)))
    # _ = plt.plot(np.append(np.array([np.nan]*len(train)),np.array(predictions)))
    # plt.xlabel('time')
    # plt.ylabel('frequency')
    # plt.title(f'word --- {word} ,  rmse --- {rmse: .2f}')
    # plt.legend(["history", "test", "prediction"])
    # plt.show()

    
    return rmse, anomaly_counts

def find_best_rmse_exp_smoothing(words):
    RMSE = []
    anomaly = []
    for word in words:
        data = scaled_freq_time_series[word]
        best_rmse = None
        best_cfg = None
        for cfg in t_d_s_sp_permutations:
            rmse, anomaly_counts = exp_smoothing_rmse(data, 0.3, cfg, word)
            if best_rmse is not None:
                if best_rmse > rmse:
                    best_rmse, best_cfg = rmse, cfg
            else:
                best_rmse, best_cfg = rmse, cfg
        RMSE.append(best_rmse)
        anomaly.append(anomaly_counts)
        print(f'word === {word} , best rmse {best_rmse} , best cfg {best_cfg}')
    

    df = pd.DataFrame(list(zip(words, RMSE, anomaly)), columns=['words', 'RMSE', 'trend_points'])
    print(df)
    fpath = os.path.join(OUTPUT_DIR, "twitter_data_tes_0_1374.csv")
    df.to_csv(fpath, index=False)
    print(f'mean RMSE  === {np.mean(RMSE): .2f}')

words = ['weather'] 
# words = ['weather', 'stupid', 'ones', 'that', 'need', 'advice', 'make', 'me', 'feel', 'warm', 'inside'] 
# words = frequency_time_series_dictionary.keys()
find_best_rmse_exp_smoothing(words)

