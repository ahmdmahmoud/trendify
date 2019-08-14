from dateutil import parser
from datetime_truncate import *
from datetime import timedelta
import argparse
import datetime
import pytz
import time
import random
import os

# TODO: extend spectrum of available timesteps if needed

'''
Creates zero filled dictionary and from a given start to given end date and fill inbetween using a timestep
'''
def create_delta_dict(youngest, oldest, timestep):
    delta_dict = {}
    current_date = oldest
    while (current_date <= youngest):
        delta_dict[current_date] = 0
        # TODO: extend spectrum of available timesteps if needed
        #days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]
        if timestep == "hour":
            current_date = current_date + timedelta(hours=1)
        elif timestep == "day":
            current_date = current_date + timedelta(days=1)
        elif timestep == "minute":
            current_date = current_date + timedelta(minutes=1)
    return delta_dict

def create_frequency_time_series(csv_file_path, timestep="hour", threshold=0, output="./frequency_time_series_data.csv"):
    utc=pytz.UTC
    entity_timestamps_dict = {}
    # read csv file back into dict
    with open(csv_file_path, mode='r') as infile:
        for line in infile:
            split = line.split(',')
            entity = split[0]
            for x in split[1:]:
                parser.parse(x.strip())
            timestamps = [truncate(parser.parse(x.strip()), timestep) for x in split[1:]]
            entity_timestamps_dict[entity] = timestamps

    frequency_time_series_dict = {}
    with open(output, 'w') as csvfile:
        non_zero_count = 0
        zero_count = 0
        for entity in entity_timestamps_dict:
            timestamps = entity_timestamps_dict[entity]

            # TODO: twitter data set specific - integrate better solution
            min_threshold_date = datetime.datetime(2011,1,23)
            min_threshold_date = utc.localize(min_threshold_date) 
            max_threshold_date = datetime.datetime(2011,2,8,23,59,59)
            max_threshold_date = utc.localize(max_threshold_date) 

            timestamps = [ts for ts in timestamps if ts < max_threshold_date and ts >= min_threshold_date]
            if timestamps:
                youngest = max_threshold_date #max(timestamps) #max(ts for ts in timestamps if ts <= max_threshold_date)
                oldest = min_threshold_date #min(timestamps) #min(ts for ts in timestamps if ts >= min_threshold_date)
                timestamp_dict = create_delta_dict(youngest, oldest, timestep)

                for ts in timestamps:
                    if ts in timestamp_dict:
                        timestamp_dict[ts] += 1
                    else:
                        print("error in creation")

                # filter time series that contain timestamps with 0 frequencies
                threshold_zero_count = 0
                non_zero = True
                for ts in timestamp_dict:
                    if non_zero and timestamp_dict[ts] == 0:
                        threshold_zero_count += 1
                        
                if threshold_zero_count > threshold:
                    non_zero = False

                if non_zero:
                    non_zero_count += 1
                    entity_timeseries_array = []
                    total_frequency = 0
                    for ts in timestamp_dict:
                        freq = timestamp_dict[ts]
                        total_frequency += freq
                        entity_timeseries_array.append(str(freq))

                    #if total_frequency >= threshold:
                    frequency_time_series_dict[entity] = [int(x) for x in entity_timeseries_array]
                    csvfile.write(entity + ',' + ",".join(entity_timeseries_array) + '\n')

                else:
                    #print (timestamp_dict)
                    zero_count += 1

    print ("Non zero count", non_zero_count)
    print ("Zero count", zero_count)
    time.sleep(6)

    return frequency_time_series_dict


def create_train_test_valid_split(path, train_split): 
    with open(path) as f:
        lines = f.readlines()
    random.shuffle(lines)

    number = len(lines)
    limit = int(number * train_split)
    limit_2 = limit + int((number - limit)/2)
    i = 0

    directory = os.path.dirname(path)
    with open(os.path.join(directory,"train"), "w") as f:
        with open(os.path.join(directory,"test"), "w") as g:
            with open(os.path.join(directory,"valid"), "w") as h:
                for line in lines:
                    split = line.split(',')
                    line = ",".join(split[1:])
                    i += 1                
                    if i < limit: 
                        f.write(line)
                    elif limit < i < limit_2:
                        g.write(line)
                    else: 
                        h.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", 
                    type=str,
                    help="path to the csv file for the entity timestamps",
                    dest="input_file",
                    default="./entity_timestamps.csv")
    parser.add_argument("-t",
                    type=str,
                    help="timestep option: hour, minute or day",
                    dest="timestep",
                    default="day")
    parser.add_argument("-th",
                    type=int,
                    help="maximum number of zero frequencies of an entity", 
                    dest="threshold",
                    default=0)
    parser.add_argument("-o", 
                    type=str,
                    help="output path for the resulting frequency time series csv file",
                    dest="output_path",
                    default="./frequency_time_series_data.csv")
    args = parser.parse_args()

    train_split = 0.8

    create_frequency_time_series(args.input_file, timestep=args.timestep, threshold=args.threshold, output=args.output_path)
    create_train_test_valid_split(args.output_path, train_split) 
    
    
if __name__ == '__main__':
    main()

