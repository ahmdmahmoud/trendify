import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results/EBAY')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results/ARIMA')


def combine_wiki_arima_multi_process_results(directory_path, pair_wise_index_list):
    csv_files = [f'wiki_data_arima_{start}_{end}.csv' for start,end in pair_wise_index_list]
    csv_files = [os.path.join(directory_path, f) for f in csv_files]
    df = pd.DataFrame()
    for f in csv_files:
        t_df = pd.read_csv(f)
        df = df.append(t_df, ignore_index=True)
    
    df.to_csv(os.path.join(OUTPUT_DIR, 'wiki_data_arima_0_41180.csv'), index=False)
    # print(df.shape)
    # print(df.head(10))


def get_pair_wise_indices():
    indices = list(range(0,41181,10295))
    pairwise_indices = []
    for i in range(1, len(indices)):
        pairwise_indices.append((indices[i-1],indices[i]))
    return pairwise_indices

combine_wiki_arima_multi_process_results(OUTPUT_DIR, get_pair_wise_indices())

def analyze_trend_points():
    

    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'wiki_data_arima_0_41180.csv'))
    
    plt.figure(figsize=(10,5))    
    _ = plt.hist(df['trend_points'])
    # plt.show()
    
    
    plus_10_percent = (df[df["trend_points"] >= 10].count()["words"]/df.shape[0]) * 100
    plus_11_percent = (df[df["trend_points"] >= 11].count()["words"]/df.shape[0]) * 100
    plus_12_percent = (df[df["trend_points"] >= 12].count()["words"]/df.shape[0]) * 100
    

    print(plus_10_percent, plus_11_percent, plus_12_percent)

    ans = pd.DataFrame()
    ans["words"] = df["words"].copy()
    
    
    ans["a_trend_10"] = np.where(df["trend_points"] >= 10, True, False)
    ans["a_trend_11"] = np.where(df["trend_points"] >= 11, True, False)
    ans["a_trend_12"] = np.where(df["trend_points"] >= 12, True, False)
    

    print(len(ans.words.unique()))
    ans.to_csv(os.path.join(OUTPUT_DIR, "wiki_arima_trend_list.csv"),index=False)

analyze_trend_points()



def compare_with_ebay():

    a_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'wiki_arima_trend_list.csv'))
    b_df = pd.read_csv(os.path.join(INPUT_DIR, 'wiki_data_ebay_0_41180.csv'))

    ab_df = pd.merge(b_df, a_df, left_on="words", right_on="words", how='inner')
    print(ab_df.columns)

    
    plus_10 = ab_df[(ab_df['a_trend_10'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_11 = ab_df[(ab_df['a_trend_11'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_12 = ab_df[(ab_df['a_trend_12'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    


    ab_df.drop(labels=['point_counts', 'a_trend_10', 'a_trend_11'], axis=1, inplace=True)
    print(ab_df.columns)
    ab_df.columns = ['words', 'ebay_trend', 'arima_trend']
    print(ab_df.columns)
    ab_df.to_csv(os.path.join(OUTPUT_DIR, 'wiki_ebay_arima_processed.csv'), index=False)

compare_with_ebay()



