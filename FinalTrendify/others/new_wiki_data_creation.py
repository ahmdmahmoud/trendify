import os
import gzip
from datetime import datetime
from time import time
from dateutil.parser import parse
from datetime import timedelta
import time
import csv
import json
import sys
import random
import statistics
import pandas as pd
import numpy as np

import fnmatch, re

from pandas.io import pickle

def normalize_sequence_min_max(sequence_in):
    min_of_frequence = min(sequence_in)
    max_of_frequence = max(sequence_in)
    min_max_delta = max(max_of_frequence - min_of_frequence, 1)
    sequence_in = [int(((x - min_of_frequence) / min_max_delta * 100)) for x in sequence_in]
    return sequence_in

def has_trend(sequence_in):
    sequence_in = normalize_sequence_min_max(sequence_in)
    standard_deviation = statistics.stdev(sequence_in)
    median = statistics.median(sequence_in)
    threshold = median + (3* standard_deviation)
    for x in range(0, len(sequence_in)):
        value = sequence_in[x]
        if value >= threshold and x > 1:
            return True
    return False

def print_in_file(*msg,**kwargs):
    message = " ".join(msg)+"\n"
    if kwargs.get("end"):
        message = kwargs["end"]
    with open("data_wiki_creation_logs.log","a") as file:
        file.write(message)
    print(*message,**kwargs)

def load_wikipedia_stats_dataset(pageviews_directory, output_csv_path, hours_per_file):
    big_dict = {}

    filelist = os.listdir(pageviews_directory)
    with gzip.open(pageviews_directory + filelist[0],'rt') as file:
        for line in file:
            line = line.strip()
            line = line.split(' ')

    number_of_files = len(filelist)
    big_dict = {}
    file_count = 0
    j=0
    current_hours = 1
    alphalist = [0] * hours_per_file

    for filename in filelist:
    #for filename in ["pageviews-20170101-000000.gz", "pageviews-20170101-010000.gz"]: #, "pageviews-20170101-020000.gz", "pageviews-20170101-030000.gz", "pageviews-20170101-040000.gz"]:
        if filename[:2] == "pa":
            with gzip.open(pageviews_directory + filename,'rt') as file:
                for line in file:
                    line = line.strip()
                    line = line.split(' ')
                    if len(line) > 2:
                        entity = str(line[0].strip()) + '_' + str(line[1].strip())
                        frequency = str(line[2].strip())
                        if entity in big_dict:
                            big_dict[entity][j] = frequency
                        else:
                            big_dict[entity] = list(alphalist)
                            big_dict[entity][j] = frequency

            # print_in_file(file_count, number_of_files)
            percentage = round(file_count * 100 / number_of_files,2)
            print_in_file("", end=f"\rPercent Complete: {file_count}/{number_of_files} = {percentage} %")
            if current_hours % hours_per_file == 0 or file_count+1 == number_of_files:
                with open(output_csv_path + "wiki_" + str(current_hours) + '.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for entity in big_dict:
                        freq_list = big_dict[entity]
                        write_line_list = [entity] + freq_list
                        writer.writerow(write_line_list)
                        j = -1
                    big_dict = {}
            file_count += 1
            j+= 1
            current_hours += 1
            # print_in_file(file_count)

    print_in_file("Done creating the wiki files")

# preprocesses the wiki_xx data to produce a file, that only contains rows with trends inside them and a file with rows, that do not have a trend inside with the specified window size
# only "zero_tolerance"*100 % of the row data are allowed to be zero
def preprocess(input_csv_dir, output_dir, zero_tolerance=0.2):
    
    files = os.listdir(input_csv_dir)

    trendy_df = pd.DataFrame()
    not_trendy_df = pd.DataFrame()

    csv_file_name_pattern = 'wiki_*.csv'

    for csv_file_path in files:
        if fnmatch.fnmatch(csv_file_path, csv_file_name_pattern) :
            print_in_file("\n\n{}".format(csv_file_path))
            actual_csv_file_path = os.path.join(input_csv_dir, csv_file_path)

            df_chunks = pd.read_csv(actual_csv_file_path, chunksize=100000)
            chunk_num = 0
            for chunk in df_chunks:
                print_in_file("FILE :::: {}  :::: chunk {}".format(csv_file_path, chunk_num))
                if chunk_num > 5:
                    break
                try: 
                    chunk['Trendy'] = chunk.iloc[:, 1:].apply(lambda row: has_trend(row), axis=1)

                    temp_trendy = chunk.loc[chunk['Trendy'] == True]
                    temp_not_trendy = chunk.loc[chunk['Trendy'] == False]

                    temp_trendy = temp_trendy.drop(['Trendy'], axis=1)
                    temp_not_trendy = temp_not_trendy.drop(['Trendy'], axis=1)

                    trendy_df = trendy_df.append(temp_trendy, ignore_index=True)
                    not_trendy_df = not_trendy_df.append(temp_not_trendy, ignore_index=True)

                except:
                    print("ERROR -> skipped. ")
                chunk_num +=1

    trendy_file_path = os.path.join(output_dir, 'trendy.pkl')
    not_trendy_file_path = os.path.join(output_dir, 'not_trendy.pkl')
    
    trendy_df.to_pickle(trendy_file_path)
    not_trendy_df.to_pickle(not_trendy_file_path)


def create_train_test_valid_split(path, train_split, data_file): 
    df = pd.read_pickle(path + data_file)

    train_mask = np.random.rand(len(df)) < train_split
    train = df[train_mask]
    test = df[~train_mask]

    

    valid_mask = np.random.rand(len(train)) < train_split
    valid = train[~valid_mask]
    train = train[valid_mask]

    print_in_file("train size :::: {}".format(train.shape))
    print_in_file("test size :::: {}".format(test.shape))
    print_in_file("valid size :::: {}".format(valid.shape))
    
    train_file_path = os.path.join(path, 'train.pkl')
    test_file_path = os.path.join(path, 'test.pkl')
    valid_file_path = os.path.join(path, 'valid.pkl')
    train.to_pickle(train_file_path)
    test.to_pickle(test_file_path)
    valid.to_pickle(valid_file_path)



# input_csv_path = os.getcwd() + "/"
# output_csv_path = os.getcwd() + "/"

pageviews_directory = '/data21/asaleh/wikipedia-pageview-stats/2017/2017-01/dumps.wikimedia.org/other/pageviews/2017/2017-01/'

input_csv_path = '/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/'
output_csv_path = '/home/sathar/trendify/trendify/wiki_data/'

hours_per_file = 64
zero_tolerance = 0.2
train_split = 0.8
data_file = 'trendy.pkl'

print("Start preprocessing and generating frequency time series data...")
t1 = time.time()
preprocess(input_csv_path, output_csv_path, zero_tolerance=zero_tolerance)
t2 = time.time()
print_in_file("\n\nPreprocess wall time :::: {} seconds".format(t2-t1))
print_in_file("Start train tets valid generation...")
create_train_test_valid_split(output_csv_path, train_split, data_file)
t3 = time.time()
print_in_file("\n\nTrain-Test-Valid split wall time :::: {} seconds".format(t3-t2))
print_in_file("\n\nDone")
print_in_file("FINISHED")




