import pandas as pd 
import tweepy
import importlib
from dateutil import parser
from datetime_truncate import * #truncate
from trendify.sequence_tagging.model.ner_model import NERModel
from nltk.tokenize import TweetTokenizer
import argparse
import os

# TODO: twitter stream support
#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)

# TODO: improve NER model (what to train on?)
# TODO: preprocess tokens?

# For possible time_unit parameter values see https://pypi.org/project/datetime_truncate/
# e.g. second, minute, 5_minute, hour, day, week, month, quarter...
def sequence_labeling(textFilesDirectory, model="config", output_path="./entity_timestamps.csv"):
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

    print ("Finished loading NER model...")

    # load twitter dataset into dataframe
    # TODO: methods to load other datasets (if needed)
    df = load_twitter_dataset (textFilesDirectory) #, time_unit)

    print ("Finished loading dataset into Dataframe...")

    tknzr = TweetTokenizer(preserve_case=True)

    labeled_entities_dict = {}

    # iterate the date sorted dataset, predict and count labels for certain time intervals and concat the results
    for index, row in df.iterrows():
        timestamp = row["dt"]
        sentence = row["content"]
        predict_and_write_to_dict(sentence, timestamp, labeled_entities_dict, classifier, tokenizer=tknzr)

    if labeled_entities_dict:
        with open(output_path, 'w') as csvfile:
            for entity in labeled_entities_dict:
                csvfile.write(entity + ',' + ",".join(labeled_entities_dict[entity]) + '\n')

    return labeled_entities_dict

def load_twitter_stream():
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    
def load_twitter_dataset(tsv_directory):#, time_unit="hour"):
    big_df = None
    #for filename in ["test.tsv"]:
    #for filename in ["20110123.tsv"]:
    for filename in os.listdir(tsv_directory):  
        # read tsv into dataframe and drop non-relevant cols
        df = pd.read_csv(tsv_directory + "/" + filename, sep='\t', header=None, names=["id", "retweets", "dt", "user", "content"], parse_dates=["dt"], date_parser=parse_dates)
        df.drop(["id", "retweets", "user"], axis=1, inplace=True)

        # sort dataset by datetime (dt)
        df.sort_values(by='dt', inplace=True)
        if big_df is None:
            big_df = df
        else: 
            big_df = pd.concat([big_df, df])
    return big_df
        

def get_label_entity_pair_sequence(zipped_list, i):
    label = "I" + zipped_list[i][1][1:]
    real_label = zipped_list[i][1][2:]
    entity_value = zipped_list[i][0]
    i += 1
    while (i < len(zipped_list) and zipped_list[i][1] == label):
        entity_value = entity_value + " " + zipped_list[i][0]
        i += 1
    return entity_value, real_label, i

# use NER classifier to predict entity labels, write them to the dict and count them
def predict_and_write_to_dict(sentence, timestamp, labeled_entities_dict, classifier, tokenizer):
    
    # simple tokenizing
    #words_raw = sentence.strip().split(" ")

    # advanced tokenizing (better tokenizer? stemmer, lemmatizer...)
    words_raw = tokenizer.tokenize(sentence)

    result = classifier.predict(words_raw)
    zipped_list = list(zip(words_raw, result))
    zipped_length = len(zipped_list)
    i = 0
    while i < zipped_length:
        pair = zipped_list[i]
        current_label = pair[1]
        if current_label != "O":
            entity_value, label, i = get_label_entity_pair_sequence(zipped_list, i)
            entity_value = entity_value.lower().replace(',', '')
            if entity_value in labeled_entities_dict:
                labeled_entities_dict[entity_value].append(str(timestamp.to_pydatetime()))
            else:
                labeled_entities_dict[entity_value] = [str(timestamp.to_pydatetime())]
        else:
            i += 1

# dates are not allowed to have hours like 24:02, this methods convert them to 00:02 instead (no 24 for hours allowed in order to parse them) 
def parse_dates(datestring):
    datesplit = datestring.split(" ")
    timesplit = datesplit[3].split(":")
    if timesplit[0] == "24":
        timesplit[0] = "00"
        datesplit[3] = ":".join(timesplit)
        datestring = " ".join(datesplit)
    dt = parser.parse(datestring)
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", 
                    type=str,
                    help="path to the tsv files of the tweets",
                    dest="input_path",
                    default="/data21/asaleh/share/Tweet2011/tweets")
    parser.add_argument("-m",
                    type=str,
                    help="NER config model",
                    dest="model",
                    default="config_twitter_wnut16")
    parser.add_argument("-o", 
                    type=str,
                    help="output path for the resulting entity timestamps csv file",
                    dest="output_path",
                    default="./entity_timestamps.csv")
    args = parser.parse_args()

    sequence_labeling(args.input_path, model=args.model, output_path=args.output_path) 
    
    
if __name__ == '__main__':
    main()