import re
import argparse
import csv
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import statistics
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import StandardScaler

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

csv.field_size_limit(100000)


def ebay_atlas(data):
    return round(statistics.stdev(data)*3, 2)


def generate_wiki_data(csv_file_path):

    word_frequency_data = []
    word_mapping_idx = {}

    with open(csv_file_path) as f:
        reader = csv.reader(f, delimiter=',')
        word_idx = 0
        for idx, row in enumerate(reader):
            try:
                word, series = row[0], row[1:]
                word_frequency_data.append([float(x.strip()) for x in series])
                word_mapping_idx[word_idx] = word
                word_idx += 1
            except:
                print("error")

    return word_frequency_data, word_mapping_idx

def generate_twitter_data(csv_file_path):

    word_frequency_data = []
    word_mapping_idx = {}

    with open(csv_file_path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        word_idx = 0
        for row in csv_reader:
            word_frequency_data.append([float(x.strip()) for x in row[1:]])
            word_mapping_idx[word_idx] = row[0]
            word_idx += 1

    return word_frequency_data, word_mapping_idx


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        type=str,
                        help="path to csv dataset file.",
                        dest="csv_file_path",
                        default="../../data/sample_wiki_385.csv")
    parser.add_argument("-t",
                        type=int,
                        help="time stamp.",
                        dest="time_stamp",
                        default="30")
    parser.add_argument("-d",
                        type=str,
                        help="Dataset type (twitter|wiki).",
                        dest="dataset_type",
                        default="twitter")
    args = parser.parse_args()

    if args.dataset_type == "twitter":
    # Get the frequency data and word mapping dict for twitter dataset.
        word_frequency_data, word_mapping_idx = generate_twitter_data(args.csv_file_path)
    elif args.dataset_type == "wiki":
    # Get the wiki dataset dict with word as key and frequencies as value.
        word_frequency_data, word_mapping_idx = generate_wiki_data(args.csv_file_path)


    words = []
    anomaly_points1 = []
    anomaly_points2 = []
    anomaly_points3 = []

    for word_idx in tqdm(range(len(word_frequency_data))):
        try:
            keras.backend.clear_session()
            # Create pandas dataframe out of word frequency data
            print(word_frequency_data[word_idx])
            df = pd.DataFrame(
            word_frequency_data[word_idx], columns=["frequency"])

            # Divide the data into train test split
            train_size = int(len(df) * 0.95)
            test_size = len(df) - train_size
            train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
            print(train.shape, test.shape)

            # Scale the frequency in (0, 1) range.
            scaler = StandardScaler()
            scaler = scaler.fit(train[['frequency']])

            train['frequency'] = scaler.transform(train[['frequency']])
            test['frequency'] = scaler.transform(test[['frequency']])

            TIME_STEPS = args.time_stamp

            X_train, y_train = create_dataset(
            train[['frequency']], train.frequency, TIME_STEPS)
            X_test, y_test = create_dataset(
            test[['frequency']], test.frequency, TIME_STEPS)

            print(X_train.shape, y_train.shape)

            model = get_model(X_train)

            # Train the model
            history = model.fit(
            X_train, y_train,
            epochs=15,
            batch_size=32,
            validation_split=0.1,
            shuffle=False
            )

            # plot the loss distribution of the training set
            X_train_pred = model.predict(X_train)
            train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

            THRESHOLD1 = 1.2
            THRESHOLD2 = 1.5
            THRESHOLD3 = 1.8

            train_score_df = pd.DataFrame(index=train[TIME_STEPS:].index)
            train_score_df['loss'] = train_mae_loss
            train_score_df['threshold1'] = THRESHOLD1
            train_score_df['threshold2'] = THRESHOLD2
            train_score_df['threshold3'] = THRESHOLD3
            train_score_df['anomaly1'] = train_score_df.loss > train_score_df.threshold1
            train_score_df['frequency'] = train[TIME_STEPS:].frequency

            words.append(word_mapping_idx[word_idx])

            # print(train_score_df['anomaly1'].value_counts())
            anomaly_points1.append(
            train_score_df['anomaly1'].value_counts().get(key=True) if train_score_df['anomaly1'].value_counts().get(key=True) else 0)

            train_score_df['anomaly2'] = train_score_df.loss > train_score_df.threshold2

            # print(train_score_df['anomaly2'].value_counts())
            anomaly_points2.append(
            train_score_df['anomaly2'].value_counts().get(key=True) if train_score_df['anomaly2'].value_counts().get(key=True) else 0)

            train_score_df['anomaly3'] = train_score_df.loss > train_score_df.threshold3

            # print(train_score_df['anomaly3'].value_counts())
            anomaly_points3.append(
            train_score_df['anomaly3'].value_counts().get(key=True) if train_score_df['anomaly3'].value_counts().get(key=True) else 0)

        except Exception as e:
            print(e)

    data = {"Word": words, "# Anomaly (Threshold 1.2)": anomaly_points1,
    "# Anomaly (Threshold 1.5)": anomaly_points2, "# Anomaly (Threshold 1.8)": anomaly_points3}
    print(data)
    trend_df = pd.DataFrame(
    data, columns=["Word", "# Anomaly (Threshold 1.2)", "# Anomaly (Threshold 1.5)", "# Anomaly (Threshold1.8)"])
    trend_df.to_csv(f'../../results/{args.dataset_type}_results/lstm/{args.dataset_type}_trending_list.csv')


if __name__ == '__main__':
    main()