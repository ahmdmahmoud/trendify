

# BASE_DIR = os.getcwd()

# TWITTER_INPUT_DIR = os.path.join(BASE_DIR, 'Results/twitter-results')
# WIKI_INPUT_DIR = os.path.join(BASE_DIR, 'Results/wiki-results')




# def final_twitter_results():
#     methods = ['MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM']

#     final_df = pd.DataFrame()

#     for method in methods:
#         if len(final_df) == 0:
#             f_name = f'twitter_data_ebay_{method.lower()}_processed.csv'
#             df = pd.read_csv(f_name)
#             final_df['words'] = df['words']
#             final_df['ebay_trend'] = df['ebay_trend']
#             final_df[f'{method.lower()}_trend'] = df[f'{method.lower()}_trend']

#         else:
#             f_name = f'twitter_data_ebay_{method.lower()}_processed.csv'
#             df = pd.read_csv(f_name)
#             final_df[f'{method.lower()}_trend'] = df[f'{method.lower()}_trend']
    
#     f_out_path = os.path.join(TWITTER_INPUT_DIR,'final_twitter_results.csv')
#     final_df.to_csv(f_out_path, index=False)

# # final_twitter_results()

# def final_wiki_results():
#     methods = ['MA', 'ARIMA', 'SARIMA', 'TES', 'LSTM']

#     final_df = pd.DataFrame()

#     for method in methods:
#         if len(final_df) == 0:
#             f_name = f'wiki_data_ebay_{method.lower()}_processed.csv'
#             df = pd.read_csv(f_name)
#             final_df['words'] = df['words']
#             final_df['ebay_trend'] = df['ebay_trend']
#             final_df[f'{method.lower()}_trend'] = df[f'{method.lower()}_trend']
            
#         else:
#             f_name = f'wiki_data_ebay_{method.lower()}_processed.csv'
#             df = pd.read_csv(f_name)
#             final_df[f'{method.lower()}_trend'] = df[f'{method.lower()}_trend']

#     f_out_path = os.path.join(WIKI_INPUT_DIR,'final_wiki_results.csv')
#     final_df.to_csv(f_out_path, index=False)

# # final_wiki_results()