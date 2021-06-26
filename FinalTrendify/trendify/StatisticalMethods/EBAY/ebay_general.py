import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statistics
import warnings
warnings.filterwarnings("ignore")

def ebay_atlas(data):
    return round(statistics.stdev(data)*3,2)


def list_of_words_has_ebay_trends(freq_time_series, words, start, end):
    
    cnt_list = []
    have_ebay_trend = []
    for word in words[start:end]:
        w_series = freq_time_series[word]
        ebay = ebay_atlas(w_series)
        cnt = 0
        for x in w_series:
            if x >= np.array(w_series).mean() + ebay:
               cnt += 1
       
        cnt_list.append(cnt)
        print(f'{word} --- {cnt}')


    df = pd.DataFrame(list(zip(words[start:end], cnt_list)), columns=['words', 'trend_points'])
    
    return df
    

def calculate_ebay(*args):
    params = args[0]
    logs = args[1] 
    produced_files_list = args[2]
    
    ### general parameters ###
    frequency_time_series_dict = params['frequency_data']
    words = list(frequency_time_series_dict.keys())
    start = params['start']
    end = params['end']
    RESULT_DIR = params['result_dir_path']
    
    calculation_df = list_of_words_has_ebay_trends(frequency_time_series_dict, words, start, end)

    fpath = os.path.join(RESULT_DIR, f'result_ebay_{start}_{end}.csv')
    calculation_df.to_csv(fpath, index=False)
    
    produced_files_list.append(fpath)
    logs.append(f'calculate_ebay  for {start} to {end} sucessfully completed...')



if __name__ == '__main__':
    pass