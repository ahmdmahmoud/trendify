import re
import argparse
import csv
import numpy as np
import os
from tensorflow import keras
import pandas as pd
import seaborn as sns
import statistics
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import StandardScaler


def ebay_atlas(data):
    return round(statistics.stdev(data)*3, 2)

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def get_model(X_train):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    return model


def apply_lstm_model(data, word, epochs, batch_size, validation_split, test_split, time_stamp):
    df = pd.DataFrame(data, columns=["frequency"])

    # Divide the data into train test split
    train_size = int(len(df) * (1-test_split))
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    # Scale the frequency in (0, 1) range.
    scaler = StandardScaler()
    scaler = scaler.fit(train[['frequency']])

    train['frequency'] = scaler.transform(train[['frequency']])
    test['frequency'] = scaler.transform(test[['frequency']])

    TIME_STEPS = time_stamp

    X_train, y_train = create_dataset(train[['frequency']], train.frequency, TIME_STEPS)
    X_test, y_test = create_dataset(test[['frequency']], test.frequency, TIME_STEPS)

    # print(X_train.shape, y_train.shape)

    model = get_model(X_train)

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        shuffle=False)

    # plot the loss distribution of the training set
    X_train_pred = model.predict(X_train)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

    # print(f"shape of mae  ==== {train_mae_loss.shape}")

    return train_mae_loss


def find_best_mae_lstm(frequency_time_series_data, words, start, end, epochs, batch_size, validation_split, test_split, THRESHOLDS, time_stamp):
    
    trend_points = {}
    for THS in THRESHOLDS:
        trend_points[str(THS)] = []

    for word in words[start:end]:
        data = frequency_time_series_data[word]
        train_mae_loss = apply_lstm_model(data, word, epochs, batch_size, validation_split, test_split, time_stamp)
        
        for THS in THRESHOLDS:
            trend_points[str(THS)].append((train_mae_loss > THS).sum())

    df = pd.DataFrame(list(zip(words[start:end])), columns=['words'])
    for THS in THRESHOLDS:
        df[f'THRESH_{THS}'] = trend_points[str(THS)]
    
    return df    




def calculate_lstm(*args):
    params = args[0]
    logs = args[1] 
    produced_files_list = args[2]

    ### LSTM parameters ###
    epochs = params['epochs']
    batch_size = params['batch_size']
    validation_split = params['validation_split']
    test_split = params['test_split']
    THRESHOLDS = params['thresholds']
    timestamps = params['timestamp']
    

    ### general parameters ###
    frequency_time_series_dict = params['frequency_data']
    words = list(frequency_time_series_dict.keys())
    start = params['start']
    end = params['end']
    RESULT_DIR = params['result_dir_path']
    
    calculation_df = find_best_mae_lstm(frequency_time_series_dict, words, start, end, epochs, batch_size, validation_split, test_split, THRESHOLDS,timestamps)

    fpath = os.path.join(RESULT_DIR, f'result_lstm_{start}_{end}.csv')
    calculation_df.to_csv(fpath, index=False)

    produced_files_list.append(fpath)
    logs.append(f'calculate_lstm for {start} to {end}  sucessfully completed...')


if __name__ == '__main__':
    pass