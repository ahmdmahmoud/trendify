import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Data/wiki-data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results/EBAY')

def ebay_atlas(data):
    return round(statistics.stdev(data)*3,2)

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
        if cnt > 10:
            have_ebay_trend.append(True)
        else:
            have_ebay_trend.append(False)

        cnt_list.append(cnt)
        print(f'{word} --- {cnt}')

    _ = plt.hist(cnt_list,bins='auto')
    plt.show()

    df = pd.DataFrame(list(zip(word_list, cnt_list, have_ebay_trend)), columns=['words', 'point_counts', 'ebay_trend'])
    
    fpath = os.path.join(BASE_DIR, f"wiki_data_ebay_{start}_{end}.csv")
    df.to_csv(fpath, index=False)

    # print(f'mean cnts - {np.array(cnt_list).mean()}')


words = list(frequency_time_series_dictionary.keys())
start, end = 0, 41180

list_of_words_has_ebay_trends(words, start, end)

