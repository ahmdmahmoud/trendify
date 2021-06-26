import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Data/wiki-data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results/SARIMA')

def load_frequency_time_series_csv(csv_file_path):
    frequency_time_series_dict = {}
    with open(csv_file_path) as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            try:
                word, series = row[0], row[1:]
                frequencies = [float(x.strip()) for x in series]
                frequency_time_series_dict[word] = frequencies
            except:
                print("error")
    
    return frequency_time_series_dict

wiki_data_filepath = os.path.join(INPUT_DIR, "original_wiki_384.csv")
# wiki_data_filepath = os.path.join(INPUT_DIR, "sample_wiki_385.csv")


frequency_time_series_dictionary =  load_frequency_time_series_csv(wiki_data_filepath)

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
        scaler = MinMaxScaler(feature_range=(0,100))
        transformed = scaler.fit_transform(data.reshape(-1,1))
        scalers[word] = scaler
        scaled_freq_time_series[word] = np.reshape(transformed, len(transformed))

    return scalers, scaled_freq_time_series

scalers, scaled_freq_time_series = scaling_data(frequency_time_series_dictionary)


def trends(fc, actual):
    df = pd.DataFrame(list(zip(actual, fc)), columns=['actual', 'forecast'])
    df['error'] = df['actual'] - df['forecast']
    df['percentage_change'] = ((df['actual'] - df['forecast']) / df['actual']) * 100
    df['meanval'] = df['error'].rolling(window=32).mean()
    df['deviation'] = df['error'].rolling(window=32).std()
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


def get_arima_model_rmse(data, p_d_q, test_split):
    train_split = 1 - test_split
    train_size = int(train_split*len(data))
    train, test = data[0:train_size], data[train_size:]
    history = [x for x in train]
    predictions = []

    diff_history = difference(history, 32)

    # for t in range(len(test)):
    #     model = sm.tsa.statespace.SARIMAX(diff_history, order=p_d_q, enforce_stationarity=False, enforce_invertibility=False)
    #     model_fit = model.fit()
    #     yhat = model_fit.forecast()[0]
    #     inverse_yhat = inverse_difference(history, yhat, 12)
    #     predictions.append(inverse_yhat)
    #     history.append(test[t])
    # print(f'{p_d_q} == {sqrt(mean_squared_error(test,predictions))}')
    
    p,d,q = p_d_q

    model = sm.tsa.statespace.SARIMAX(diff_history, order=p_d_q, seasonal_order=(0,0,0,32), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    f_cast = model_fit.forecast(len(test))
    for yhat in f_cast:
        inverted = inverse_difference(history, yhat, 32)
        history.append(inverted)
        predictions.append(inverted)

    rmse = np.sqrt(np.mean((np.array(test)-np.array(predictions))**2))

    anomaly_counts = trends(predictions, test)

    
    print(f'{p_d_q} ======== {rmse:.2f}')

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



p = [6]
d = [0]
q = [2]
p_d_q_permutation = list(itertools.product(p,d,q))

def find_best_rmse_SARIMA_model(words, start, end):
    RMSE, anomaly_points = [], []
    for word in words[start : end]:
        data = scaled_freq_time_series[word]
        
        print(f'word ==> {word} ')

        best_rmse, best_pdq = None, None
        for p_d_q in p_d_q_permutation:
            rmse, anomaly_counts = get_arima_model_rmse(data, p_d_q, 0.3) 
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

    print(f'mean rmse --- {np.mean(RMSE)}')
    df = pd.DataFrame(list(zip(words[start : end], RMSE, anomaly_points)), columns=['words', 'rmse', 'trend_points'])
    print(df)
    # print(f'mean rmse ------ {np.mean(df["rmse"])}')
    fpath = os.path.join(OUTPUT_DIR, f'wiki_data_sarima_{start}_{end}.csv')
    df.to_csv(fpath, index=False)

words = ['de_Lacrosse',
 'ar.m_محمد_بركات',
 'de_Österreich-Ungarns_Heer_im_Ersten_Weltkrieg',
 'de_Fantomas_(1964)',
 'de_Dissen_(Umgangssprache)',
 'de.m_ATP_Doha',
 'ar.zero_بري_أولسون',
 'ar.m_ريما_مكتبي',
 'de.m_Apostolisches_Glaubensbekenntnis',
 'ar.m_مدحت_صالح']

# words = pd.read_csv(os.path.join(INPUT_DIR, "list_of_words.csv"))['words'].tolist()

# start = 0
# end = 10295


# t1 = time.time()
# find_best_rmse_SARIMA_model(words, start, end)
# t2 = time.time()

# print(f'total time {t2-t1} sec.')




## multi-processing ##
from multiprocessing import Process

## configurations ##
num_process = 4
total_len = len(words)

start = 0
steps = total_len // num_process


target_method_call = find_best_rmse_SARIMA_model

proc_list = []
for i in range(start, total_len, steps):
    method_args = (words, i, i+steps)
    p = Process(target=target_method_call, args=method_args)
    p.start()
    proc_list.append(p)

for p in proc_list:
    p.join()
