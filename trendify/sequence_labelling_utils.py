import keras.backend as K
import numpy as np
from random import randint
import pandas as pd 
import re, os
import tweepy
import importlib
from datetime import datetime
from dateutil import parser
from datetime_truncate import * #truncate
import csv

#from trendify.sequence_tagging.model.config import Config
from trendify.sequence_tagging.model.ner_model import NERModel

from nltk.tokenize import TweetTokenizer

#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)

# TODO: improve NER model (what to train on?)
# TODO: preprocess tokens?

# For possible time_unit parameter values see https://pypi.org/project/datetime_truncate/
# e.g. second, minute, 5_minute, hour, day, week, month, quarter...
def sequence_labeling(textFilesDirectory, model="config", time_unit="hour"):
    """
    Sequence labeling. uses a trained model to extract entities(categories)  from some texts
    :param model: the trained bi-LSTM model.
    :param textFilesDirectory: the directory which contains the text files. 
    :return: labeled_entities: a sequence of entities in the text
    """  
    mod = importlib.import_module("trendify.sequence_tagging.model." + model)

    # NER model parameters, includes the path for embeddings 
    #config = Config()
    config = mod.Config()

    # build and restore NER model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model) 
    classifier = model

    print ("DONE")

    # load twitter dataset into dataframe
    # TODO: methods to load other datasets (if needed)
    df = load_twitter_dataset (textFilesDirectory) #, time_unit)

    print ("DONE LOADING DATASET INTO DATAFRAME")
    tknzr = TweetTokenizer(preserve_case=True)

    labeled_entities_dict = {}

    # iterate the date sorted dataset, predict and count labels for certain time intervals and concat the results
    for index, row in df.iterrows():
        timestamp = row["dt"]
        sentence = row["content"]
        predict_and_write_to_dict(sentence, timestamp, labeled_entities_dict, classifier, tokenizer=tknzr)
        #print (labeled_entities_dict)

    if labeled_entities_dict:
        with open('entity_timestamps.csv', 'w') as csvfile:
            # fieldnames = ['entity', 'timestamps']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()
            for entity in labeled_entities_dict:
                csvfile.write(entity + ',' + ",".join(labeled_entities_dict[entity]) + '\n')
                #writer.writerow({'entity': entity, 'timestamps': labeled_entities_dict[entity]})
    #     ndf = pd.DataFrame.from_dict(labeled_entities_dict)
    #     ndf.columns = ['entity', 'timestamps'] 

    # ndf.to_csv("entity_timestamps.csv", sep='\t')
    return labeled_entities_dict

def load_twitter_stream():
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    
def load_twitter_dataset(tsv_directory):#, time_unit="hour"):
    big_df = None
    #for filename in ["smalltest.tsv"]:
    for filename in ["20110123.tsv"]:
    #for filename in os.listdir(tsv_directory):  
        # read tsv into dataframe and drop non-relevant cols
        df = pd.read_csv(tsv_directory + "/" + filename, sep='\t', header=None, names=["id", "retweets", "dt", "user", "content"], parse_dates=["dt"], date_parser=parse_dates)
        df.drop(["id", "retweets", "user"], axis=1, inplace=True)
        # truncate the datetimes here
        #### NOT NEEDED ANYMORE, NEW FORMAT, TRUNCATE LATER ####
        #df["dt"] = df["dt"].apply(lambda x: truncate(x, time_unit))

        # sort dataset by datetime (dt)
        df.sort_values(by='dt', inplace=True)
        if big_df is None:
            big_df = df
        else: 
            big_df = pd.concat([big_df, df])
    return big_df
        
# use NER classifier to predict entity labels, write them to the dict and count them
def predict_and_write_to_dict(sentence, timestamp, labeled_entities_dict, classifier, tokenizer):
    
    # simple tokenizing
    #words_raw = sentence.strip().split(" ")

    # advanced tokenizing (better tokenizer? stemmer, lemmatizer...)
    words_raw = tokenizer.tokenize(sentence)

    result = classifier.predict(words_raw)
    #print (result)
    zipped_list = zip(words_raw, result)
    current_entity = None
    current_label = None
    for pair in zipped_list:
        if pair[1] != "O":
            #print (pair)
            # combine BI-Tags to full entity and save occurence in dict
            if pair[1][:2] == "B-":
                if current_entity and current_label:
                    current_entity = current_entity.lower()
                    #write_pair = (current_entity, current_label)
                    #timestamp = timestamp.to_datetime()
                    print (timestamp)
                    if current_entity in labeled_entities_dict:
                        labeled_entities_dict[current_entity].append(str(timestamp.to_pydatetime()))
                    else:
                        labeled_entities_dict[current_entity] = [str(timestamp.to_pydatetime())]
                current_entity = pair[0]
                current_label = pair[1][2:]

            elif pair[1][:2] == "I-":
                if current_entity and current_label:
                    current_entity = current_entity + " " + pair[0]
                else:
                    current_entity = pair[0]
                    current_label = pair[1][2:]

# dates are now allowed to have hours like 24:02, this methods convert them to 00:02 instead (no 24 for hours allowed in order to parse them) 
def parse_dates(datestring):
    datesplit = datestring.split(" ")
    timesplit = datesplit[3].split(":")
    if timesplit[0] == "24":
        timesplit[0] = "00"
        datesplit[3] = ":".join(timesplit)
        datestring = " ".join(datesplit)
    dt = parser.parse(datestring)
    return dt


