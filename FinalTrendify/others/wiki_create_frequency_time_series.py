import os
import gzip
from datetime import datetime
from dateutil.parser import parse
from datetime import timedelta
import time
import csv
import json
import sys
import random
import statistics

def print_in_file(*msg,**kwargs):
    message = " ".join(msg)+"\n"
    if kwargs.get("end"):
        message = kwargs["end"]
    with open("wiki_creation_logs.log","a") as file:
        file.write(message)
    print(*message,**kwargs)

# Normalizes a sequence to values between 0 and 100 (including 0 and 100)
def normalize_sequence_min_max(sequence_in):
    min_of_frequence = min(sequence_in)
    max_of_frequence = max(sequence_in)
    min_max_delta = max(max_of_frequence - min_of_frequence, 1)
    sequence_in = [int(((x - min_of_frequence) / min_max_delta * 100)) for x in sequence_in]
    return sequence_in

# boolean check whether a sequence_in has a trend inside it or not using the 3*std.deviation method (ATLAS ebay)
def has_trend(sequence_in):
    standard_deviation = statistics.stdev(sequence_in)
    median = statistics.median(sequence_in)
    threshold = median + (3* standard_deviation)
    for x in range(0, len(sequence_in)):
        value = sequence_in[x]
        if value >= threshold and x > 1:
            return True
    return False


def load_wikipedia_stats_dataset(pageviews_directory, output_csv_path, hours_per_file):
    big_dict = {}

    filelist = os.listdir(pageviews_directory)
    with gzip.open(pageviews_directory + filelist[0],'rt') as file:
        for line in file:
            line = line.strip()
            line = line.split(' ')

    number_of_files = len(filelist)
    big_dict = {}
    file_count = 0
    j=0
    current_hours = 1
    alphalist = [0] * hours_per_file

    for filename in filelist:
    #for filename in ["pageviews-20170101-000000.gz", "pageviews-20170101-010000.gz"]: #, "pageviews-20170101-020000.gz", "pageviews-20170101-030000.gz", "pageviews-20170101-040000.gz"]:
        if filename[:2] == "pa":
            with gzip.open(pageviews_directory + filename,'rt') as file:
                for line in file:
                    line = line.strip()
                    line = line.split(' ')
                    if len(line) > 2:
                        entity = str(line[0].strip()) + '_' + str(line[1].strip())
                        frequency = str(line[2].strip())
                        if entity in big_dict:
                            big_dict[entity][j] = frequency
                        else:
                            big_dict[entity] = list(alphalist)
                            big_dict[entity][j] = frequency

            # print_in_file(file_count, number_of_files)
            percentage = round(file_count * 100 / number_of_files,2)
            print_in_file("", end=f"\rPercent Complete: {file_count}/{number_of_files} = {percentage} %")
            if current_hours % hours_per_file == 0 or file_count+1 == number_of_files:
                with open(output_csv_path + "wiki_" + str(current_hours) + '.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for entity in big_dict:
                        freq_list = big_dict[entity]
                        write_line_list = [entity] + freq_list
                        writer.writerow(write_line_list)
                        j = -1
                    big_dict = {}
            file_count += 1
            j+= 1
            current_hours += 1
            # print_in_file(file_count)

    print_in_file("Done creating the wiki files")

# preprocesses the wiki_xx data to produce a file, that only contains rows with trends inside them and a file with rows, that do not have a trend inside with the specified window size
# only "zero_tolerance"*100 % of the row data are allowed to be zero
def preprocess(input_csv_dir, zero_tolerance=0.2):
    trend_preprocess_out_path = output_csv_path + 'trendy'
    notrend_preprocess_out_path = output_csv_path + 'not_trendy'
    files = os.listdir(input_csv_dir)
    #files = ['wiki_128.csv']
    with open(trend_preprocess_out_path, mode='w') as trend_outfile:
        with open(notrend_preprocess_out_path, mode='w') as notrend_outfile:
            for csv_file_path in files:
                if csv_file_path[:4] == "wiki":
                    with open(input_csv_dir + csv_file_path, mode='r') as infile:
                        csv_reader = csv.reader(infile, delimiter=',')
                        try:
                            for line in csv_reader:
                                if len(line) > 1:
                                    word_entity, frequencies = line[0], [int(x.strip()) for x in line[1:]]
                                    if frequencies.count(0) <= zero_tolerance*len(frequencies):
                                        trendy = has_trend(normalize_sequence_min_max(frequencies))
                                        if trendy:
                                            frequencies_string = word_entity + ",".join([str(x) for x in frequencies])
                                            trend_outfile.write(frequencies_string + '\n')
                                        else:
                                            frequencies_string = word_entity + ",".join([str(x) for x in frequencies])
                                            notrend_outfile.write(frequencies_string + '\n')
                                            pass
                        except csv.Error as e:
                            print_in_file(e)


def create_train_test_valid_split(path, train_split, data_file): 
    with open(path + data_file) as f:
        lines = f.readlines()
    random.shuffle(lines)

    number = len(lines)
    limit = int(number * train_split)
    limit_2 = limit + int((number - limit)/2)
    i = 0

    with open(path + "train", "w") as f:
        with open(path + "test", "w") as g:
            with open(path + "valid", "w") as h:
                for line in lines:
                    i += 1                
                    if i < limit: 
                        f.write(line)
                    elif limit < i < limit_2:
                        g.write(line)
                    else: 
                        h.write(line)


# parameters to modify
pageviews_directory = '/data21/asaleh/wikipedia-pageview-stats/2017/2017-01/dumps.wikimedia.org/other/pageviews/2017/2017-01/'
input_csv_path = '/data21/asaleh/share/trendify_data/frequency_timeseries_data/wiki_stats/'
output_csv_path = '/home/sathar/trendify/trendify/wiki_data/'
hours_per_file = 64
zero_tolerance = 0.2
train_split = 0.8  # test_split / valid_split = (1-train_split)/2
data_file = 'trendy'  # or not_trendy

#print_in_file("Load and process wiki pageview files...")
#load_wikipedia_stats_dataset(pageviews_directory, output_csv_path, hours_per_file)

print_in_file("Start preprocessing and generating frequency time series data...")
preprocess(input_csv_path, zero_tolerance=zero_tolerance)

print_in_file("Start train tets valid generation...")
create_train_test_valid_split(output_csv_path, train_split, data_file)

print_in_file("Done")
print_in_file("FINISHED")
