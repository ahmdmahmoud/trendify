# -*- coding: utf-8 -*-
import argparse
import os

from random import randint
import numpy as np

from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import pandas as pd

from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import plot_model
from keras import backend
    
from trendify.attention_decoder_example2 import AttentionDecoder
from trendify.attention_utils import get_activations, get_data_recurrent, gzip_file_reader
import trendify.sequence_labelling_utils as sl

import trendify.create_entity_timestamps_csv as entity_timestamps_creation
import trendify.create_frequency_time_series as frequency_time_series_creation

#import trendify.attention_decoder
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

import sys
import time

import csv
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config) 
backend.set_session(sess)

# compares gold and predicted sequence for the early prediction approach, counts the number of the exact matches 
# and the fuzzy matches, e.g. where the method predicted the beginning of a trend but not with the real value (which is almost impossible anyways)
def custom_array_equal(a,b, n=2):
    exact_match_count = 0
    fuzzy_match_count = 0
    if a == b:
        return (n+1,n+1)
    else:
        if len(a) == len(b):
            for i in range(0,len(a)):
                if a[i] == 0 and b[i] != 0:
                    return (0,0)

            max_i = np.argmax(a)
            for j in range(max_i-n, max_i+1):
                if a[j] != 0:
                    difference = a[j] - b[j]
                    if difference <= 0:
                        exact_match_count += 1
                    if difference < a[max_i]:
                        fuzzy_match_count += 1

            return (exact_match_count, fuzzy_match_count)
        else:
            print ("length error")
            return (0,0)
                

# custom metric for keras that calculates the rmse in each epoch
def rmse_metric(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#Plot a diagram of the expected and predicted values
def plot_predicted_values (expected, predicted, parameter_config_string): 
    
    plt.plot(expected[0], color='b')
    plt.plot(predicted[0], color='r')
    
    plt.savefig('/data21/bwerner/moving/Code/trendify/results/graphs/' + parameter_config_string + '.png')
    #plt.savefig('/data21/asaleh/GIT/moving/Code/trendify/results/attention_model_random_numbers_example.png')
    plt.show()

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# Normalizes a sequence to values between 0 and 100 (including 0 and 100)
def normalize_sequence_min_max(sequence_in):
    min_of_frequence = min(sequence_in)
    max_of_frequence = max(sequence_in)
    min_max_delta = max(max_of_frequence - min_of_frequence, 1)
    sequence_in = [int(((x - min_of_frequence) / min_max_delta * 100)) for x in sequence_in]
    return sequence_in

# produces the gold sequence_out for a given sequence_in, this method is for the trend detection method and produces a sequence with zeroes except for the trend value
def get_sequence_out_with_deviation(sequence_in):
    sequence_out = []
    standard_deviation = statistics.stdev(sequence_in)
    median = statistics.median(sequence_in)
    threshold = median + (3* standard_deviation)
    # print ("Standard Deviation: ", standard_deviation)
    # print ("threshold: ", threshold)
    for value in sequence_in:
        if value >= threshold:
            sequence_out.append(value)
        else:
            sequence_out.append(0)
    return sequence_out

# produces the gold sequence_out for a given sequence_in, this method is for the early prediction method and produces a sequence of zeroes except for the trend value 
# and the n values in front of it, if the trend is at least in the third position of the sequence
def get_early_sequence_out_with_deviation(sequence_in, n=2):
    sequence_out = []
    standard_deviation = statistics.stdev(sequence_in)
    median = statistics.median(sequence_in)
    threshold = median + (3* standard_deviation)
    # print ("Standard Deviation: ", standard_deviation)
    # print ("threshold: ", threshold)
    for x in range(0, len(sequence_in)):
        value = sequence_in[x]
        if value >= threshold and x >= n:
            sequence_out.append(value)
            for k in range(x-n, x):
                sequence_out[k] = sequence_in[k]
        else:
            sequence_out.append(0)
    return sequence_out

# boolean check whether a sequence_in has a trend inside it or not using the 3*std.deviation method (ATLAS ebay)
def has_trend(sequence_in):
    # if sequence_in.count(0) > 0.2*len(sequence_in):
    #     return False
    # else:
    standard_deviation = statistics.stdev(sequence_in)
    median = statistics.median(sequence_in)
    threshold = median + (3* standard_deviation)
    for x in range(0, len(sequence_in)):
        value = sequence_in[x]
        if value >= threshold and x > 1:
            #print ("value", value)
            #print ("threshold", threshold)
            return True
    return False

def encoder_decoder(parameter_config, train_X, test_X, train_Y, test_Y, validation_split, results_path, number_of_values_before_atlas=2):
    print ("starting encoder decoder model with parameters: ", parameter_config["parameter_config_string"])
    parameter_config_string = parameter_config["parameter_config_string"]

    with_attention = parameter_config["attention"]

    # configure problem
    n_features = 101
    n_timesteps_in = parameter_config["window_size"]
    lstm_size = parameter_config["lstm_size"]
	
    # define model, using attention or no attention
    if with_attention:
        model = Sequential()
        model.add(LSTM(lstm_size, input_shape=(n_timesteps_in, n_features), return_sequences=True))
        model.add(AttentionDecoder(lstm_size, n_features))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', rmse_metric])
    else:
        model = Sequential()
        model.add(LSTM(lstm_size, input_shape=(n_timesteps_in, n_features)))
        model.add(RepeatVector(n_timesteps_in))
        model.add(LSTM(lstm_size, return_sequences=True))
        model.add(TimeDistributed(Dense(n_features, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', rmse_metric])

    # train LSTM
    model.fit(train_X, train_Y, validation_split=validation_split, epochs=parameter_config["epoch_number"], batch_size=parameter_config["batch_size"])

    # evaluate LSTM
    total, correct = test_X.shape[0], 0
    rmses = []
    gold_trend_count = 0
    correct_trend_count = 0

    correct_exact_3 = 0
    correct_exact_2 = 0
    correct_exact_1 = 0

    correct_fuzzy_3 = 0
    correct_fuzzy_2 = 0
    correct_fuzzy_1 = 0

    false_count = 0

    for l in range(total):
        X = test_X[l]
        y = test_Y[l]
        X = X.reshape((1, X.shape[0], X.shape[1]))
        y = y.reshape((1, y.shape[0], y.shape[1]))
        yhat = model.predict(X, verbose=0)

        if not all([v == 0 for v in one_hot_decode(y[0])]):
            gold_trend_count += 1

        if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
            correct += 1
            if not all([v == 0 for v in one_hot_decode(y[0])]):
                correct_trend_count += 1
        
        exact_score, fuzzy_score = custom_array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0]), number_of_values_before_atlas)
        if exact_score == 3:
            correct_exact_3 += 1
        elif exact_score == 2:
            correct_exact_2 += 1
        elif exact_score == 1:
            correct_exact_1 += 1
        
        if fuzzy_score == 3:
            correct_fuzzy_3 += 1
        elif fuzzy_score == 2:
            correct_fuzzy_2 += 1
        elif fuzzy_score == 1:
            correct_fuzzy_1 += 1
        else:
            false_count += 1
            

        rmse_val = rmse(np.array(one_hot_decode(yhat[0])), np.array(one_hot_decode(y[0])))
        rmses.append(rmse_val)
    print ('Results for: ' + parameter_config_string)
    print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
    print('Trend Accuracy: %.2f%%' % (float(correct_trend_count)/float(gold_trend_count)*100.0))
    print('Gold Trend Count', str(gold_trend_count), 'Correct Trend Count', str(correct_trend_count)) 
    avg_rmse = statistics.mean(rmses)
    print('Avg RMSE: ', avg_rmse)
    print('Exact Score 3: ', correct_exact_3, 'Exact Score 2: ', correct_exact_2, 'Exact Score 1: ', correct_exact_1, 'from total: ', total)
    print('Fuzzy Score 3: ', correct_fuzzy_3, 'Fuzzy Score 2: ', correct_fuzzy_2, 'Fuzzy Score 1: ', correct_fuzzy_1, 'from total: ', total)

    with open(results_path, mode='a') as result_file:
        result_file.write('Parameters: ' + parameter_config_string + '\n' + 
            'Accuracy: %.2f%%' % (float(correct)/float(total)*100.0) + '\n' +
            'Trend Accuracy: %.2f%%' % (float(correct_trend_count)/float(gold_trend_count)*100.0) + '\n'
            'Gold Trend Count ' + str(gold_trend_count) + ', Correct Trend Count ' + str(correct_trend_count) + '\n'
            'Avg RMSE: ' + str(avg_rmse) + '\n' + 
            'Exact Score 3: ' + str(correct_exact_3) + ', Exact Score 2: ' + str(correct_exact_2) + ', Exact Score 1: ' + str(correct_exact_1) + ', from total: ' +  str(total) + '\n'
            'Fuzzy Score 3: ' + str(correct_fuzzy_3) + ', Fuzzy Score 2: ' + str(correct_fuzzy_2) + ', Fuzzy Score 1: ' + str(correct_fuzzy_1) + ', from total: ' +  str(total) + '\n' + '\n')
    
    # predicted and expected arrays 
    predicted_all = []
    expected_all = [] 
    
    # spot check some examples
    for l in range(10):
        X = test_X[l]
        y = test_Y[l]
        X = X.reshape((1, X.shape[0], X.shape[1]))
        y = y.reshape((1, y.shape[0], y.shape[1]))
        yhat = model.predict(X, verbose=0)
        print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
        expected_all.append(one_hot_decode(y[0]))
        predicted_all.append(one_hot_decode(yhat[0]))
    
    # Plot a diagram 
    plot_predicted_values (expected_all, predicted_all, parameter_config_string)

    # Plot the model 
    #plot_model(model, to_file='LSTM_model_without_attention.png')
    #plot_model(model, to_file='/data21/asaleh/GIT/moving/Code/trendify/results/LSTM_model_without_attention_details.png', show_shapes=True, show_layer_names=True)
    

def encoder_decoder_generator(parameter_config, train_generator, valid_generator, validation_split, test_generator, results_path, generator_counts=(10,10,10), not_trendy_test_generator=None, number_of_values_before_atlas=2):
    print ("starting encoder decoder model with parameters: ", parameter_config["parameter_config_string"])
    parameter_config_string = parameter_config["parameter_config_string"]

    with_attention = parameter_config["attention"]

    # configure problem
    n_features = 101
    n_timesteps_in = parameter_config["window_size"]
    lstm_size = parameter_config["lstm_size"]

    # generator counts
    train_count = generator_counts[0]
    valid_count = generator_counts[1]
    test_count = generator_counts[2]

    if not_trendy_test_generator:
        not_trendy_test_count = generator_counts[3]

    # define model
    if with_attention:
        model = Sequential()
        model.add(LSTM(lstm_size, input_shape=(n_timesteps_in, n_features), return_sequences=True))
        model.add(AttentionDecoder(lstm_size, n_features))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', rmse_metric])
    else:
        model = Sequential()
        model.add(LSTM(lstm_size, input_shape=(n_timesteps_in, n_features)))
        model.add(RepeatVector(n_timesteps_in))
        model.add(LSTM(lstm_size, return_sequences=True))
        model.add(TimeDistributed(Dense(n_features, activation='softmax')))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    #forecasting_model = Sequential()
    #forecasting_model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
    #forecasting_model.add(AttentionDecoder(150, n_features))
    #forecasting_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    # train LSTM
    model.fit_generator(train_generator, steps_per_epoch=train_count, validation_data=valid_generator, validation_steps=valid_count, epochs=parameter_config["epoch_number"], 
        workers=1, verbose=1, use_multiprocessing=False)  #steps_per_epoch=206340, validation_steps=68753, ###steps_per_epoch=None
    
    # evaluate LSTM
    total = 0
    correct = 0
    rmses = []
    gold_trend_count = 0
    correct_trend_count = 0

    exact_scores = []
    fuzzy_scores = []

    for _ in range(test_count):
    #for X,y in test_generator:
        total += 1
        X,y = next(test_generator)
        yhat = model.predict(X, verbose=0)
        
        if not all([v == 0 for v in one_hot_decode(y[0])]):
            gold_trend_count += 1

        if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
            correct += 1
            if not all([v == 0 for v in one_hot_decode(y[0])]):
                correct_trend_count += 1

        exact_score, fuzzy_score = custom_array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0]), number_of_values_before_atlas)
        exact_scores.append(exact_score)
        fuzzy_scores.append(fuzzy_score)
        # add exact and fuzzy scores to exact and fuzzy lists and then count occurence of each integer - boom ??? --- profit!!!

        rmse_val = rmse(np.array(one_hot_decode(yhat[0])), np.array(one_hot_decode(y[0])))
        rmses.append(rmse_val)

    max_fuzzy = max(fuzzy_scores)
    exact_dict = {}
    fuzzy_dict = {}
    for k in range(0,max_fuzzy+1):
        exact = exact_scores.count(k)
        exact_dict[k] = exact
        fuzzy = fuzzy_scores.count(k)
        fuzzy_dict[k] = fuzzy

    if not_trendy_test_generator:
        not_trendy_correct_examples = []
        not_trendy_false_examples = []
        total_not_trendy = 0
        correct_not_trendy = 0
        for _ in range(not_trendy_test_count):
        #for X,y in test_generator:
            total_not_trendy += 1
            X,y = next(not_trendy_test_generator)
            yhat = model.predict(X, verbose=0)
            gold = one_hot_decode(y[0])
            prediction = one_hot_decode(yhat[0])
            if array_equal(gold, prediction):
                correct_not_trendy += 1
                if len(not_trendy_correct_examples) < 10:
                    not_trendy_correct_examples.append((gold, prediction))
            elif len(not_trendy_false_examples) < 10:
                not_trendy_false_examples.append((gold, prediction))

        for x in not_trendy_correct_examples:
            print('Correct non trendy examples')
            print('Expected:', x[0], 'Predicted', x[1])

        for x in not_trendy_false_examples:
            print('False non trendy examples')
            print('Expected:', x[0], 'Predicted', x[1])

    print ('Results for: ' + parameter_config_string)
    print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
    print('Trend Accuracy: %.2f%%' % (float(correct_trend_count)/float(gold_trend_count)*100.0))
    print('Gold Trend Count', str(gold_trend_count), 'Correct Trend Count', str(correct_trend_count)) 
    avg_rmse = statistics.mean(rmses)
    print('Avg RMSE: ', avg_rmse)

    ies = [k for k in exact_dict]
    ies = reversed(sorted(ies))
    exact_string = ""
    for i in ies:
        exact_string += 'Exact Score ' + str(i) + ': ' + str(exact_dict[i]) + ' '

    ies = [k for k in fuzzy_dict]
    ies = reversed(sorted(ies))
    fuzzy_string = ""
    for i in ies:
        fuzzy_string += 'Fuzzy Score ' + str(i) + ': ' + str(fuzzy_dict[i]) + ' '

    print(exact_string + 'from total: ' + str(total))
    print(fuzzy_string + 'from total: ' + str(total))
    if not_trendy_test_generator:
        print('Non Trend Accuracy: %.2f%%' % (float(correct_not_trendy)/float(total_not_trendy)*100.0))
        print('Not Trend Count ' + str(total_not_trendy) + ', Correct Not Trend Count ' + str(correct_not_trendy))
    
    with open(results_path, mode='a') as result_file:
        result_string = 'Parameters: ' + parameter_config_string + '\n' \
            + 'Accuracy: %.2f%%' % (float(correct)/float(total)*100.0) + '\n' \
            + 'Trend Accuracy: %.2f%%' % (float(correct_trend_count)/float(gold_trend_count)*100.0) + '\n' \
            + 'Gold Trend Count ' + str(gold_trend_count) + ', Correct Trend Count ' + str(correct_trend_count) + '\n' \
            + 'Avg RMSE: ' + str(avg_rmse) + '\n' \
            + exact_string + 'from total: ' + str(total) + '\n' \
            + fuzzy_string + 'from total: ' + str(total) + '\n'
        if not_trendy_test_generator:
            result_string += 'Non Trend Accuracy: %.2f%%' % (float(correct_not_trendy)/float(total_not_trendy)*100.0) + '\n' \
                + 'Not Trend Count ' + str(total_not_trendy) + ', Correct Not Trend Count ' + str(correct_not_trendy) + '\n' + '\n'
        else:
            result_file += '\n'
        result_file.write(result_string)
    
    # predicted and expected arrays 
    predicted_all = []
    expected_all = [] 
    
    # spot check some examples 
    for _ in range(10):
        X,y = next(test_generator)
        yhat = model.predict(X, verbose=0)
        print('Input:', one_hot_decode(X[0]), 'Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))
        expected_all.append(one_hot_decode(y[0]))
        predicted_all.append(one_hot_decode(yhat[0]))
    
    # Plot a diagram 
    plot_predicted_values (expected_all, predicted_all, parameter_config_string)
    
    #plot_model(model, to_file='/data21/asaleh/GIT/moving/Code/trendify/results/LSTM_model_with_attention.png')
    #plot_model(model, to_file='/data21/asaleh/GIT/moving/Code/trendify/results/LSTM_model_with_attention.png', show_shapes=True, show_layer_names=True)


# method to split a list l into equally sized chunks of n values, the last chunk is not returned if it would be smaller than n
def chunks(l, n):
    for i in range(0, len(l), n):
        if i+n < len(l):
            yield l[i:i + n]

# method to load the frequency time series data from a csv file into a python dictionary
def load_frequency_time_series_csv(csv_file_path):
    with open(csv_file_path, mode='r') as infile:
        csv_reader = csv.reader(infile, delimiter=',')
        frequency_time_series_dict = {}
        for line in csv_reader:
            split = line
            entity = split[0].strip()
            try:
                frequencies = [int(x.strip()) for x in split[1:]]
                frequency_time_series_dict[entity] = frequencies
            except:
                print ("error")
    return frequency_time_series_dict

# preprocesses the wiki_xx data to produce a file, that only contains rows with trends inside them and a file with rows, that do not have a trend inside with the specified window size
# only 20% of the row data are allowed to be zero
def preprocess(input_csv_dir, window_size):
    trend_preprocess_out_path = '/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/trendy_ws_' + str(window_size)
    notrend_preprocess_out_path = '/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/not_trendy_ws_' + str(window_size)
    files = os.listdir(input_csv_dir)
    #files = ['wiki_128.csv']
    with open(trend_preprocess_out_path, mode='w') as trend_outfile:
        with open(notrend_preprocess_out_path, mode='w') as notrend_outfile:
            for csv_file_path in files:
                print (csv_file_path)
                if csv_file_path[:4] == "wiki":
                    with open(input_csv_dir + csv_file_path, mode='r') as infile:
                        csv_reader = csv.reader(infile, delimiter=',')
                        try:
                            for line in csv_reader:
                                #entity = line[0].strip()
                                if len(line) > 1:
                                    frequencies = [int(x.strip()) for x in line[1:]]
                                    for chunk in chunks(frequencies, window_size):
                                        if chunk:
                                            if chunk.count(0) <= 0.2*len(chunk):
                                                trendy = has_trend(normalize_sequence_min_max(chunk))
                                                if trendy:
                                                    chunk_string = ",".join([str(x) for x in chunk])
                                                    trend_outfile.write(chunk_string + '\n')
                                                else:
                                                    chunk_string = ",".join([str(x) for x in chunk])
                                                    notrend_outfile.write(chunk_string + '\n')
                                                    pass
                        except csv.Error as e:
                            print (e)

# calcualtes the rmse
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def wiki_generator(file_path, window_size, batch_size, early_prediction, max_count, not_trendy=False, number_of_values_before_atlas=2):
    internal_counter = 1
    tfile = open(file_path, mode='r')
    X_list = []
    Y_list = []
    lines = []
    for line in tfile:
        lines.append(line)

    i = 0
    while 1:
        if i == max_count-1:
            i = 0
        line = lines[i]
        x = [int(a) for a in line.strip().split(',')]
        x_sequence_out = normalize_sequence_min_max(x)
        if not_trendy:
            y_sequence_out = [0] * len(x_sequence_out)
        elif early_prediction:
            y_sequence_out = get_early_sequence_out_with_deviation(x_sequence_out, number_of_values_before_atlas)
        else:
            y_sequence_out = get_sequence_out_with_deviation(x_sequence_out)

        X_list.append(one_hot_encode(x_sequence_out ,101))
        Y_list.append(one_hot_encode(y_sequence_out ,101))

        X = np.array(X_list)
        #X_shape = X.shape
        Y = np.array(Y_list)
        #Y_shape = Y.shape
        if internal_counter % batch_size == 0:
            X_list = []
            Y_list = []
            #print ('batch at ', internal_counter)
            yield X,Y
            
        internal_counter += 1
        i += 1



# loads and converts the data into train and test data with feature X and label Y sets: train_X, test_X, train_Y, test_Y
def load_data(frequency_time_series_dict, window_size, validation_split, early_prediction, number_of_values_before_atlas=2):
    # ["bieber", "taylor", ...]
    entities = [x for x in frequency_time_series_dict] 

    # [[0,1,4,7,1,2], [1,4,2,3,5,7], ...]
    frequency_time_series_array = [frequency_time_series_dict[x] for x in frequency_time_series_dict]

    sequence_list = []
    # [[0,1], [4,7], [1,2], [1,4], [2,3], [5,7]]

    # optional for generating the trending rows file
    trend_count_dos = 0
    j = 0
    trending_rows = []

    for array in frequency_time_series_array:
        has_trend = False
        for chunk in chunks(array, window_size):
            if chunk:
                sequence_list.append(chunk)

                # test and condition optional for generating the trending rows file
                if early_prediction:
                    test = get_early_sequence_out_with_deviation(normalize_sequence_min_max(chunk), number_of_values_before_atlas)
                else:
                    test = get_sequence_out_with_deviation(normalize_sequence_min_max(chunk))
                if not all([v == 0 for v in test]):
                    entity = entities[j]
                    entity_frequency_list = [entity] + array
                    trending_rows.append(entity_frequency_list)
                    has_trend = True
        
        # optional for generating the trending rows file
        j += 1
        if has_trend:
            trend_count_dos += 1

    # optional for generating the trending rows file
    with open('daily_filter_trending.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in trending_rows:
            writer.writerow(row)
    
    #print ("Trend count", trend_count_dos)
    #time.sleep(30)
        #input_data.append(tmp_array)
        # [[[0,1], [4,7], [1,2]],          [[1,4], [2,3], [5,7]]]

    total = 0
    trend_count = 0
    X_list = []
    Y_list = []
    for x in sequence_list:
        total += 1
        x_sequence_out = normalize_sequence_min_max(x)
        if early_prediction:
            y_sequence_out = get_early_sequence_out_with_deviation(x_sequence_out, number_of_values_before_atlas)
        else:
            y_sequence_out = get_sequence_out_with_deviation(x_sequence_out)
        if not all([v == 0 for v in y_sequence_out]):
            trend_count += 1
            # print (x_sequence_out)
            # print (y_sequence_out)
            X_list.append(one_hot_encode(x_sequence_out ,101))
            Y_list.append(one_hot_encode(y_sequence_out ,101))

    #X_list = [one_hot_encode(normalize_sequence_min_max(x),101) for x in sequence_list]
    X = np.array(X_list)
    X_shape = X.shape
    #Y_list = [one_hot_encode(get_sequence_out_with_deviation(normalize_sequence_min_max(y)) ,101) for y in sequence_list]
    Y = np.array(Y_list)
    Y_shape = Y.shape

    train_split = 1 - validation_split
    train_samples_number = int(X_shape[0] * train_split)
    test_samples_number = X_shape[0] - train_samples_number
    train_X = X
    test_X = X[train_samples_number:]
    train_Y = Y
    test_Y = Y[train_samples_number:]

    return train_X, test_X, train_Y, test_Y, total, trend_count

# generates the different parameter configurations for the Grid search approach, to find the best parameters from the ones given below
def generate_parameter_configs(attention="both", number_of_values_before_atlas=None):
    parameter_configs = []
    if attention == "both":
        attention = [True, False]
    elif attention == "n":
        attention = [False]
    else:
        attention = [True]
    if number_of_values_before_atlas:
        atlas_numbers_possible = [number_of_values_before_atlas]
    else:
        atlas_numbers_possible = [3,4,2]
    window_sizes_possible = [16]#, 32, 64, 128] #, 50, 150] #[5, 10, 30, 50, 100, 200, 300, 400]
    epoch_numbers_possible = [1,5] #, 128, 1024, 4,64] #, 10000] #[5, 10, 30, 50, 100, 500, 1000, 10000]
    batch_sizes_possible = [16] #[1, 2, 5, 10, 50, 100, 200, 500]
    lstm_sizes_possible = [150] #[50, 100, 150, 200, 250]

    for a in attention:
        for atlas_number in atlas_numbers_possible:
            for window_size in window_sizes_possible:
                for epoch_number in epoch_numbers_possible:
                    for batch_size in batch_sizes_possible:
                        for lstm_size in lstm_sizes_possible:
                            config = {}
                            config["attention"] = a
                            config["atlas_number"] = atlas_number
                            config["window_size"] = window_size
                            config["epoch_number"] = epoch_number
                            config["batch_size"] = batch_size
                            config["lstm_size"] = lstm_size
                            config["parameter_config_string"] = "attention_" + str(a) + "-atlasnumber_" + str(atlas_number) + "-windowsize_" + str(window_size) + "-epochs_" + str(epoch_number) + "-batchsize_" + str(batch_size) + "-lstmsize_" + str(lstm_size)
                            parameter_configs.append(config)

    return parameter_configs

def main():
    """ Parses command line arguments and either performs indexing or
    partial doc update operations
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-a", "--attention", type=str, default="y",
                        help="with or without attention (y or n)")

    # arguments for the entity timestamps creation
    parser.add_argument("-tsv", 
                    type=str,
                    help="path to the tsv files of the tweets",
                    dest="tsv_input_path",
                    default="/data21/asaleh/share/Tweet2011/tweets")
    parser.add_argument("-m",
                    type=str,
                    help="NER config model",
                    dest="model",
                    default="config_twitter_wnut16")
    parser.add_argument("-eto", 
                    type=str,
                    help="output path for the resulting entity timestamps csv file",
                    dest="entity_timestamps_csv_output_path",
                    default="./entity_timestamps.csv")

    # arguments for the frequency time series dict creation
    parser.add_argument("-e", 
                    type=str,
                    help="path to a previously created entity timestamps csv file if available",
                    dest="entity_timestamps_csv_path")
    parser.add_argument("-f", 
                    type=str,
                    help="path to a previously created frequency time series data csv file if available",
                    dest="frequency_time_series_csv_path")
    parser.add_argument("-t",
                    type=str,
                    help="timestep option: hour, minute or day",
                    dest="timestep",
                    default="hour")
    parser.add_argument("-th",
                    type=int,
                    help="maximum number of zero frequencies of an entity",
                    dest="threshold",
                    default=0)
    parser.add_argument("-fto", 
                    type=str,
                    help="output path for the resulting frequency time series csv file",
                    dest="frequency_time_series_output_path",
                    default="./frequency_time_series_data.csv")
    parser.add_argument("-w", 
                    type=int,
                    help="window size for the trend detection",
                    dest="window_size",
                    default=16)
    parser.add_argument("-atlas", 
                    type=int,
                    help="specifies how many values before an atlas trend should be used in training and evaluation",
                    dest="number_of_values_before_atlas",
                    default=2)
    parser.add_argument("-r", 
                    type=str,
                    help="path for the result files",
                    dest="results_path",
                    default="/data21/bwerner/moving/Code/trendify/results/test/test.txt")
    parser.add_argument("-ep", "--early_prediction", type=str, default="n",
                        help="with or without early prediction (y or n)")
    parser.add_argument("-pp", "--preprocess", type=str, default="n",
                        help="with or without preprocessing (y or n)")
    parser.add_argument("-wiki", "--wiki", type=str, default="n",
                        help="with or without wiki data (y or n)")

    args = parser.parse_args()

    if args.number_of_values_before_atlas is not None:
        number_of_values_before_atlas = args.number_of_values_before_atlas

    parameter_configs = generate_parameter_configs(attention=args.attention, number_of_values_before_atlas=args.number_of_values_before_atlas )
    validation_split = 0.3

    if args.early_prediction == "y":
        early_prediction = True
    else:
        early_prediction = False 

    if args.preprocess == "y":
        if args.frequency_time_series_csv_path is not None:
            preprocess(args.frequency_time_series_csv_path, parameter_configs[0]["window_size"])
        sys.exit()

    if args.wiki == "y":
        pass
    elif args.frequency_time_series_csv_path is not None:
        print ("Frequency time series CSV is given...loading")
        frequency_time_series_dict = load_frequency_time_series_csv(args.frequency_time_series_csv_path)
        print ("Loading complete!")
    elif args.entity_timestamps_csv_path is not None:
        print ("Entity timestamps CSV is given...")
        print ("Generating frequency time series data with a timestep of " + args.timestep)
        entity_timestamps_path = args.entity_timestamps_csv_path
        frequency_time_series_dict = frequency_time_series_creation.create_frequency_time_series(entity_timestamps_path, timestep=args.timestep, threshold=args.threshold, output=args.frequency_time_series_output_path)
        print ("Generation complete!")
    elif args.tsv_input_path is not None:
        print ("Twitter TSV path is given...generating Entity timestamps data")
        entity_timestamps_creation.sequence_labeling(args.tsv_input_path, model=args.model, output_path=args.entity_timestamps_csv_output_path)
        entity_timestamps_path = args.entity_timestamps_csv_output_path
        print ("Done!")
        print ("Generating frequency time series data with a timestep of " + args.timestep)
        frequency_time_series_dict = frequency_time_series_creation.create_frequency_time_series(entity_timestamps_path, timestep=args.timestep, threshold=args.threshold, output=args.frequency_time_series_output_path)

    for parameter_config in parameter_configs:
        if args.wiki == "y":
            # train_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/train', window_size=parameter_config["window_size"], batch_size=parameter_config["batch_size"], early_prediction=False)
            # valid_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/valid', window_size=parameter_config["window_size"], batch_size=parameter_config["batch_size"], early_prediction=False)
            # test_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/test', window_size=parameter_config["window_size"], batch_size=1, early_prediction=False)
            # train_count = sum(1 for _ in train_generator)
            # valid_count = sum(1 for _ in valid_generator)
            # test_count = sum(1 for _ in test_generator)
            # generator_counts = (train_count, valid_count, test_count)
            generator_counts = (206340,68700,50000,50000) 
            print ('Generator Counts', generator_counts)
            train_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/train', window_size=parameter_config["window_size"], batch_size=parameter_config["batch_size"], early_prediction=early_prediction, max_count=generator_counts[0], number_of_values_before_atlas=number_of_values_before_atlas)
            valid_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/valid', window_size=parameter_config["window_size"], batch_size=parameter_config["batch_size"], early_prediction=early_prediction, max_count=generator_counts[1], number_of_values_before_atlas=number_of_values_before_atlas)
            test_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/test', window_size=parameter_config["window_size"], batch_size=1, early_prediction=early_prediction, max_count=generator_counts[2], number_of_values_before_atlas=number_of_values_before_atlas)
            not_trendy_test_generator = wiki_generator('/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/not_trend_test', window_size=parameter_config["window_size"], batch_size=1, early_prediction=early_prediction, max_count=generator_counts[2], not_trendy=True)
            encoder_decoder_generator(parameter_config, train_generator, valid_generator, validation_split, test_generator, args.results_path, generator_counts = generator_counts, not_trendy_test_generator=not_trendy_test_generator, number_of_values_before_atlas=number_of_values_before_atlas)
        else:
            train_X, test_X, train_Y, test_Y, total, trend_count = load_data(frequency_time_series_dict, parameter_config["window_size"], validation_split, early_prediction, number_of_values_before_atlas=number_of_values_before_atlas)
            parameter_config["parameter_config_string"] += "-trendcount_" + str(trend_count) + "_of_" + str(total) 
            if trend_count > 0:
                encoder_decoder(parameter_config, train_X, test_X, train_Y, test_Y, validation_split, args.results_path, number_of_values_before_atlas=number_of_values_before_atlas)
            else:
                print ("No Trends - abort " + parameter_config["parameter_config_string"])


if __name__ == '__main__':
    main()