import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Data/twitter-data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/twitter-results/EBAY')

def ebay_atlas(data):
    return round(statistics.stdev(data)*3,2)

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

def list_of_words_has_ebay_trends(word_list, start, end):
    
    
    cnt_list = []
    have_ebay_trend = []
    for word in word_list:
        w_series = frequency_time_series_dictionary[word]
        ebay = ebay_atlas(w_series)
        cnt = 0
        for x in w_series:
            if x >= np.array(w_series).mean() + ebay:
               cnt += 1
        if cnt > 2:
            have_ebay_trend.append(True)
        else:
            have_ebay_trend.append(False)

        cnt_list.append(cnt)
        print(f'{word} --- {cnt}')

    _ = plt.hist(cnt_list,bins='auto')
    plt.show()

    df = pd.DataFrame(list(zip(word_list, cnt_list, have_ebay_trend)), columns=['words', 'point_counts', 'ebay_trend'])
    
    fpath = os.path.join(BASE_DIR, f"twitter_data_ebay_{start}_{end}.csv")
    df.to_csv(fpath, index=False)

    # print(f'mean cnts - {np.array(cnt_list).mean()}')


words = list(frequency_time_series_dictionary.keys())
start, end = 0, 1374
list_of_words_has_ebay_trends(words, start, end)