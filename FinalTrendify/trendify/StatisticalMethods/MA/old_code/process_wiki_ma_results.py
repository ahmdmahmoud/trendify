import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results/EBAY')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results/MA')


def combine_wiki_ma_multi_process_results(directory_path, pair_wise_index_list):
    csv_files = [f'wiki_data_ma_{start}_{end}.csv' for start,end in pair_wise_index_list]
    csv_files = [os.path.join(directory_path, f) for f in csv_files]
    df = pd.DataFrame()
    for f in csv_files:
        t_df = pd.read_csv(f)
        df = df.append(t_df, ignore_index=True)
    
    df.to_csv(os.path.join(OUTPUT_DIR, 'wiki_data_ma_0_41180.csv'), index=False)
    # print(df.shape)
    # print(df.head(10))


def get_pair_wise_indices():
    indices = list(range(0,41181,10295))
    pairwise_indices = []
    for i in range(1, len(indices)):
        pairwise_indices.append((indices[i-1],indices[i]))
    return pairwise_indices

combine_wiki_ma_multi_process_results(OUTPUT_DIR, get_pair_wise_indices())

def analyze_trend_points():
    

    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'wiki_data_ma_0_41180.csv'))

    plt.figure(figsize=(20,10))    
    _ = plt.hist(df['trend_points'])
    # plt.show()

    plus_7_percent = (df[df["trend_points"] >= 7].count()["words"]/df.shape[0]) * 100
    plus_8_percent = (df[df["trend_points"] >= 8].count()["words"]/df.shape[0]) * 100
    plus_9_percent = (df[df["trend_points"] >= 9].count()["words"]/df.shape[0]) * 100
    plus_10_percent = (df[df["trend_points"] >= 10].count()["words"]/df.shape[0]) * 100
    plus_11_percent = (df[df["trend_points"] >= 11].count()["words"]/df.shape[0]) * 100
    plus_12_percent = (df[df["trend_points"] >= 12].count()["words"]/df.shape[0]) * 100

    print(plus_7_percent, plus_8_percent, plus_9_percent, plus_10_percent, plus_11_percent, plus_12_percent)

    ans = pd.DataFrame()
    ans["words"] = df["words"].copy()
    
    ans["s_trend_8"] = np.where(df["trend_points"] >= 8, True, False)
    ans["s_trend_9"] = np.where(df["trend_points"] >= 9, True, False)
    ans["s_trend_10"] = np.where(df["trend_points"] >= 10, True, False)
    ans["s_trend_11"] = np.where(df["trend_points"] >= 11, True, False)
    ans["s_trend_12"] = np.where(df["trend_points"] >= 12, True, False)

    print(len(ans.words.unique()))
    ans.to_csv(os.path.join(OUTPUT_DIR, "wiki_ma_trend_list.csv"),index=False)

analyze_trend_points()




def compare_with_ebay():

    s_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'wiki_ma_trend_list.csv'))
    e_df = pd.read_csv(os.path.join(INPUT_DIR, 'wiki_data_ebay_0_41180.csv'))

    es_df = pd.merge(e_df, s_df, on="words")

    # print(es_df.head())
    plus_8 = es_df[(es_df['s_trend_8'] == True) & (es_df['ebay_trend'] == True)].shape[0] / es_df[es_df['ebay_trend'] == True].shape[0] * 100
    plus_9 = es_df[(es_df['s_trend_9'] == True) & (es_df['ebay_trend'] == True)].shape[0] / es_df[es_df['ebay_trend'] == True].shape[0] * 100
    plus_10 = es_df[(es_df['s_trend_10'] == True) & (es_df['ebay_trend'] == True)].shape[0] / es_df[es_df['ebay_trend'] == True].shape[0] * 100
    plus_11 = es_df[(es_df['s_trend_11'] == True) & (es_df['ebay_trend'] == True)].shape[0] / es_df[es_df['ebay_trend'] == True].shape[0] * 100
    plus_12 = es_df[(es_df['s_trend_12'] == True) & (es_df['ebay_trend'] == True)].shape[0] / es_df[es_df['ebay_trend'] == True].shape[0] * 100

    # print(f'{es_df[(es_df["s_trend_9"] == True) & (es_df["ebay_trend"] == True)].shape[0]}')
    # print(plus_8, plus_9, plus_10, plus_11, plus_12)


    es_df.drop(labels=['point_counts', 's_trend_9', 's_trend_10', 's_trend_11', 's_trend_12'], axis=1, inplace=True)
    print(es_df.columns)
    es_df.columns = ['words', 'ebay_trend', 'ma_trend']
    print(es_df.columns)
    es_df.to_csv(os.path.join(OUTPUT_DIR, 'wiki_ebay_ma_processed.csv'), index=False)

compare_with_ebay()