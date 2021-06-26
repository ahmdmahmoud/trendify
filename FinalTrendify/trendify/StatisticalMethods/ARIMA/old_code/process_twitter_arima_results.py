import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.getcwd()

INPUT_DIR = os.path.join(BASE_DIR, 'Results/twitter-results/EBAY')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Results/twitter-results/ARIMA')


def analyze_trend_points():
    

    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'twitter_data_arima_0_1374.csv'))
    
    plt.figure(figsize=(10,5))    
    _ = plt.hist(df['trend_points'])
    # plt.show()
    
    plus_5_percent = (df[df["trend_points"] > 5].count()["words"]/df.shape[0]) * 100
    plus_6_percent = (df[df["trend_points"] > 6].count()["words"]/df.shape[0]) * 100
    plus_7_percent = (df[df["trend_points"] > 7].count()["words"]/df.shape[0]) * 100
    plus_8_percent = (df[df["trend_points"] > 8].count()["words"]/df.shape[0]) * 100
    plus_9_percent = (df[df["trend_points"] > 9].count()["words"]/df.shape[0]) * 100
    plus_10_percent = (df[df["trend_points"] > 10].count()["words"]/df.shape[0]) * 100
    

    print(plus_5_percent, plus_6_percent, plus_7_percent, plus_8_percent, plus_9_percent, plus_10_percent)

    ans = pd.DataFrame()
    ans["words"] = df["words"].copy()
    
    ans["a_trend_5"] = np.where(df["trend_points"] > 5, True, False)
    ans["a_trend_6"] = np.where(df["trend_points"] > 6, True, False)
    ans["a_trend_7"] = np.where(df["trend_points"] > 7, True, False)
    ans["a_trend_8"] = np.where(df["trend_points"] > 8, True, False)
    ans["a_trend_9"] = np.where(df["trend_points"] > 9, True, False)
    ans["a_trend_10"] = np.where(df["trend_points"] > 10, True, False)
    

    print(len(ans.words.unique()))
    ans.to_csv(os.path.join(OUTPUT_DIR, "twitter_arima_trend_list.csv"),index=False)

analyze_trend_points()



def compare_with_ebay():

    a_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'twitter_arima_trend_list.csv'))
    b_df = pd.read_csv(os.path.join(INPUT_DIR, 'twitter_data_ebay_0_1374.csv'))

    ab_df = pd.merge(b_df, a_df, left_on="words", right_on="words", how='inner')
    print(ab_df.columns)

    plus_5 = ab_df[(ab_df['a_trend_5'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_6 = ab_df[(ab_df['a_trend_6'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_7 = ab_df[(ab_df['a_trend_7'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_8 = ab_df[(ab_df['a_trend_8'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_9 = ab_df[(ab_df['a_trend_9'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    plus_10 = ab_df[(ab_df['a_trend_10'] == True) & (ab_df['ebay_trend'] == True)].shape[0] / ab_df[ab_df['ebay_trend'] == True].shape[0] * 100
    

    # print(f'{ab_df[(ab_df["a_trend_7"] == True) & (ab_df["ebay_trend"] == True)].shape[0]}')
    # print(plus_5, plus_6, plus_7, plus_8, plus_9, plus_10)


    ab_df.drop(labels=['point_counts', 'a_trend_5', 'a_trend_6', 'a_trend_7', 'a_trend_8', 'a_trend_9'], axis=1, inplace=True)
    print(ab_df.columns)
    ab_df.columns = ['words', 'ebay_trend', 'arima_trend']
    print(ab_df.columns)
    ab_df.to_csv(os.path.join(OUTPUT_DIR, 'twitter_ebay_arima_processed.csv'), index=False)

compare_with_ebay()



