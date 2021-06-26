import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

import argparse
from multiprocessing import Process, Manager
from trendify.StatisticalMethods.MA import ma_general
from trendify.StatisticalMethods.ARIMA import arima_general
from trendify.StatisticalMethods.EBAY import ebay_general
from trendify.StatisticalMethods.SARIMA import sarima_general
from trendify.StatisticalMethods.TES import tes_general
from trendify.LSTM import lstm_general



## rename the methods ##
## each method represented by calculate_* function name ##
calculate_ebay = ebay_general.calculate_ebay
calculate_ma = ma_general.calculate_ma
calculate_arima = arima_general.calculate_arima
calculate_sarima = sarima_general.calculate_sarima
calculate_tes = tes_general.calculate_tes
calculate_lstm = lstm_general.calculate_lstm

    
"""
? Use : This script will run all the methods on given dataset in the form of frequency-time-series-csv

* Methods

    * EBAY
    * ARIMA
    * MA
    * SARIMA
    * TES
    * LSTM
    
"""

## definig base_directory for wasy traversal ##
BASE_DIR = os.getcwd()

# twitter_data_directory = os.path.join(BASE_DIR, 'Data/twitter-data')
# wiki_data_directory = os.path.join(BASE_DIR, 'Data/wiki-data')

## a cumulative density function ##
## this function will plot the cdf of trend_points to decide what should be threshold ##

def show_cdf_and_decide_threshold(data, method_name, data_column):

    if method_name == "LSTM":
        fig, axes = plt.subplots(1,len(data_column),figsize=(16,8))
        for i, col in enumerate(data_column):
            sns.ecdfplot(data=data, y=col, complementary=True, ax=axes[i])
        plt.title(f'method_name == {method_name}, thresholds == {data_column}')
        plt.show(block=False)
        print(f'method_name == {method_name}, thresholds == {data_column}')
        threshold = input(f'choose threshold value from {data_column} :::   ')
        plt.close()
        sns.ecdfplot(data=data, y=threshold, complementary=True)
        plt.show(block=False)
        trend_points_threshold = float(input(f'enter threshold value for {method_name} in float   :::   '))
        plt.close()
        
        return threshold, trend_points_threshold

    else:
        _ = sns.ecdfplot(data=data, y=data_column, complementary=True)
        plt.title(f'method_name == {method_name}, min == {np.min(data[data_column])}, max == {np.max(data[data_column])}, mean == {np.mean(data[data_column])}')
        plt.show(block=False)
        print(f'method_name == {method_name}, min == {np.min(data[data_column])}, max == {np.max(data[data_column])}, mean == {np.mean(data[data_column])}')
        threshold = float(input(f'enter threshold value for {method_name} in float   :::   '))
        plt.close()
        return threshold


### this function will process results to final results by analysis of trend_points ##


def analyze_and_process_final_results(file_path, method_name):
    
    if method_name == "LSTM":
        df = pd.read_csv(file_path)
        data_column = [col for col in df.columns if col != "words"]
        threshold, trend_points_threshold = show_cdf_and_decide_threshold(df, method_name, data_column)
        print(threshold, trend_points_threshold)
        df_processed = df.copy()
        df_processed[f'{method_name.lower()}_trend'] = np.where(df_processed[threshold] >= trend_points_threshold, True, False)
        df_processed = df_processed.drop(labels=data_column, axis=1)
        return df_processed

    elif method_name == "EBAY":
        labels=['trend_points']
        data_column = 'trend_points'
        df = pd.read_csv(file_path)
        threshold = show_cdf_and_decide_threshold(df, method_name, data_column)
        print(threshold)
        df_processed = df.copy()
        print(f'total trend percent == {(df_processed[df_processed[data_column] >= threshold].count()["words"] / df.shape[0]) * 100}')
        df_processed[f'{method_name.lower()}_trend'] = np.where(df_processed[data_column] >= threshold, True, False)
        df_processed = df_processed.drop(labels=labels, axis=1)
        return df_processed
    else:
        labels=['trend_points', 'rmse']
        data_column = 'trend_points'
        df = pd.read_csv(file_path)
        threshold = show_cdf_and_decide_threshold(df, method_name, data_column)
        print(threshold)
        df_processed = df.copy()
        print(f'total trend percent == {(df_processed[df_processed[data_column] >= threshold].count()["words"] / df.shape[0]) * 100}')
        df_processed[f'{method_name.lower()}_trend'] = np.where(df_processed[data_column] >= threshold, True, False)
        df_processed = df_processed.drop(labels=labels, axis=1)
        return df_processed

    # else:
    #     df = pd.read_csv(file_path)
    #     threshold = show_cdf_and_decide_threshold(df, method_name)
    #     print(threshold)
    #     df_processed = df.copy()
    #     print(f'total trend percent == {(df_processed[df_processed["trend_points"] >= threshold].count()["words"] / df.shape[0]) * 100}')


### combine results function used for combining results in dataframe ###
### can be vertical or horizontal combining ###

def combine_results(file_list, concat_files=False):
    if concat_files == False:
        final_df = pd.DataFrame()
        for file in file_list:
            df = pd.read_csv(file)
            final_df = final_df.append(df)
        return final_df
    else:
        final_df = pd.DataFrame()
        for file in file_list:
            df = pd.read_csv(file)
            if len(final_df) == 0:
                final_df = df.copy()
            else:
                final_df = pd.merge(left=final_df, right=df, left_on="words", right_on="words")
        return final_df


### general purpose data loading function ###

def load_frequency_time_series_data(file_path):
    df = pd.read_csv(file_path)
    df = df.set_index('words', drop=True)
    df = df.transpose()
    frequency_time_series_dict = df.to_dict('list')
    # frequency_time_series_dict = {}
    # frequency_time_series_dict = df.to_dict()
    # for key in frequency_time_series_dict.keys():
    #     frequency_time_series_dict[key] = frequency_time_series_dict[key].values()
    return frequency_time_series_dict


##################################################################################

# ? Main Function : forecasting + trend detection using all methods on given data

##################################################################################





def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name",
                        type=str,
                        help="frequency time series csv data file name",
                        dest="data_file_name",
                        default="sample_10.csv")
    parser.add_argument("--data_dir",
                        type=str,
                        help="data directory name in ./Data",
                        dest="data_dir_name",
                        default="temp-data")   
    
    parser.add_argument("--methods", 
                        nargs="+", 
                        help="list of methods want to apply on the data from ['EBAY', 'MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM']",
                        default=['EBAY', 'MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM'])

    parser.add_argument("--result_dir",
                        type=str,
                        help="result directory name in ./Results",
                        dest="result_dir_name",
                        default="temp-results")

    parser.add_argument("--computation",
                        type=bool,
                        help="True: computation of trend_points + analysis, False: only analysis",
                        dest="do_computation",
                        default=True)
                
    # methods = ['EBAY', 'MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM']

    calculation_functions = {
        'EBAY': calculate_ebay,
        'MA': calculate_ma, 
        'ARIMA': calculate_arima, 
        'SARIMA': calculate_sarima, 
        'TES': calculate_tes, 
        'LSTM': calculate_lstm
    }
    
    ### this dictionary contains different parametes for each method

    calculation_methods_parameters = {
        'EBAY': {
            # nothing as parameters
        },
        'MA': {
            'lags': [2],
            'test_split': 0.2,
            'scaling': (0,10),
        }, 
        'ARIMA': {
            'p': [7],
            'd': [0],
            'q': [1],
            'repeate_interval': 12, 
            'test_split': 0.2,
            'scaling': (0,10),
        },
        'SARIMA': {
            'p': [7],
            'd': [0],
            'q': [1],
            's': [24],
            'repeate_interval': 12,
            'test_split': 0.2,
            'scaling': (0,10),
        },
        'TES': {
            't':['add'],
            'd':[True],
            's':['mul'],
            's_p':[24],
            'test_split': 0.2,
            'scaling': (0,10),
        },
        'LSTM': {
            'epochs':15, 
            'batch_size': 32,
            'validation_split': 0.1,
            'thresholds': [1.2, 1.5, 1.8],
            'timestamp': 30,
            'test_split': 0.05,
        }
    }

    ### parse the arguments ###

    args = parser.parse_args()

    ### directory paths ###
    data_file_path = os.path.join(BASE_DIR, "Data", args.data_dir_name, args.data_file_name)
    result_dir_path = os.path.join(BASE_DIR, "Results", args.result_dir_name)
    
    ### methods to compute ###
    methods = args.methods
    
    ### do_computation flag ###
    ### if False then analysis on already generated results ###
    do_computation = args.do_computation

    if do_computation == True:

        ## load the data into dictionary format from csv ##
        frequency_time_series_dict = load_frequency_time_series_data(data_file_path)

        ## words from the data ##
        words = frequency_time_series_dict.keys()

        FINAL_RESULTS_FILES_LIST = []

        ## iterate over each method and produce results ##
        for method in methods:

            result_dir_method_path = os.path.join(result_dir_path, method)
            if not os.path.exists(result_dir_method_path):
                os.makedirs(result_dir_method_path)


            target_method_call = calculation_functions[method]

            ## method's parameters ##
            method_parameters = calculation_methods_parameters[method]
            
            total_data_len = len(frequency_time_series_dict)
            print(total_data_len)
            num_process = 4
            start = 0
            steps = total_data_len//num_process

            process_list = []

            produced_files_list = []

            ### multi-processing ###
            with Manager() as manager:
                produced_files = manager.list()
                logs = manager.list()

                for i in range(start, total_data_len, steps):
                    method_args = manager.dict()
                    method_args = {
                        'frequency_data': frequency_time_series_dict,
                        'start': i,
                        'end': i+steps,
                        'result_dir_path': result_dir_method_path
                    }
                    method_args.update(method_parameters)

                    p = Process(target=target_method_call, args=(method_args,logs, produced_files))
                    p.start()
                    process_list.append(p)
                
                for p in process_list:
                    p.join()
                
                print("\n".join(logs))
                # with open(os.path.join(result_dir_path, "logs.txt"), "a+") as f:
                #     f.write("\n".join(logs))

                produced_files_list = [f for f in produced_files]

            ### get combine results of multi-processing ###
            final_result_df = combine_results(produced_files_list)
            final_result_df_fpath = os.path.join(result_dir_method_path, f'result_{method.lower()}_{start}_{total_data_len}.csv')
            final_result_df.to_csv(final_result_df_fpath, index=False)

            ### get True/False labeled results for each methods ###
            final_df_processed = analyze_and_process_final_results(final_result_df_fpath, method)
            final_df_processed_fpath = os.path.join(result_dir_method_path, f'FINAL_RESULT_{method}_{start}_{total_data_len}.csv')
            final_df_processed.to_csv(final_df_processed_fpath, index=False)

            FINAL_RESULTS_FILES_LIST.append(final_df_processed_fpath)

        ### combine all methods True/False labeled results in to one ###
        ALL_METHODS_COMBINE = combine_results(FINAL_RESULTS_FILES_LIST, True)
        result_all_methods_path = os.path.join(result_dir_path, "FINAL_ALL_METHODS")
        if not os.path.exists(result_all_methods_path):
            os.makedirs(result_all_methods_path)
        ALL_METHODS_COMBINE_fpath = os.path.join(result_all_methods_path, f'FINAL_{"_".join(methods)}.csv')
        ALL_METHODS_COMBINE.to_csv(ALL_METHODS_COMBINE_fpath, index=False)
    
    else:
        pass





if __name__ == '__main__':
    main()
