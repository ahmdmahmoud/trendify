from dateutil import parser
from datetime_truncate import *
from datetime import timedelta
import argparse
import datetime
import pytz
import time
import random
import os
import re

#Yago
import psycopg2
import psycopg2.extensions
psycopg2.extensions.register_type(psycopg2.extensions.UNICODE)
psycopg2.extensions.register_type(psycopg2.extensions.UNICODEARRAY)


# TODO: extend spectrum of available timesteps if needed

#YagoEntitiesCounter = 0
# Yago class 
class Yago:
    
    def __init__(self, dbname='yago', user='ahmed', host='localhost', password='ahmed'):
        self.conn = psycopg2.connect("dbname='%s' user='%s' host='%s' password='%s'" % (dbname, user, host, password))
        
    def query(self, subject, relations):
        cursor = self.conn.cursor()
        
        # when no relations are specified, we take all
        if not relations:
            query = "SELECT predicate,object FROM yagofacts WHERE subject='%s'" % (self._double_quote_escape(subject))
        else:
            relations = map(lambda s: "'%s'" % self._double_quote_escape(s), relations)
            relations_sql_string = ",".join(relations)
            query = "SELECT predicate,object FROM yagofacts WHERE subject='%s' AND predicate IN (%s)" % (self._double_quote_escape(subject),relations_sql_string)
            if query: 
                print("Yago can be successfully queried")
                #YagoEntitiesCounter += 1
                #print("Number of entities found in Yago:", YagoEntitiesCounter)
                
        #replace str with unicode if you are using python 2
        cursor.execute(str(query))
        rows = cursor.fetchall()
        
        return rows
    
    def _double_quote_escape(self, s):
        return s.replace("'", "''")



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


'''
Regex to return a list of words containing a given substring
input example:  f("'new york'@eng", "@eng")
output:         ["new york"] 
'''

def f(s, pat):
    pat =  r'"([^"]*)"%s' % re.escape(pat)  #r'\b\S*%s\S*\b' % re.escape(pat)
    return re.findall(pat, s)
    
    #re.findall(r'"([^"]*)"', inputString)


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


def create_frequency_time_series_and_related_yago_entities(csv_file_path, timestep="hour", threshold=0, output="./frequency_time_series_data.csv", yago_output="./frequency_time_series_data_yago.csv"):
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
    with open(output, 'w') as csvfile, open(yago_output, 'w') as yago_csvfile,  open('./results/related_yago_entities.csv', 'w') as related_yago_entities_csvfile:
        #with open(yago_output, 'w') as yago_csvfile:

        non_zero_count = 0
        zero_count = 0
        entities_detected_in_yago_counter = 0
        
        #TODO: after querying the Yago, detect how many of them are in Yago
        
        print("Here we are debbuging the list of entities: ")
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
                    
                    # Only in this case where the timeseries is valid, check if the entity is in yago 
                    yago = Yago()
                    
                    
                    #first letter in upper case
                    #yago_entity = entity
                    #convert the entity to a format that could allow yago to understant it. so mainly replacing spaces with underscors 
                    yago_entity = entity.title()
                    yago_entity = yago_entity.replace(" ", "_")
                    yago_entity = "<"+yago_entity+">"
                    
                    if (yago.query(yago_entity, [])): 
                        entities_detected_in_yago_counter += 1
                        print(entities_detected_in_yago_counter, yago_entity) 
                        #write the frequency time series of yago entity in a yago output file
                        yago_csvfile.write(entity + ',' + ",".join(entity_timeseries_array) + '\n')
                        
                        #related_yago_entities is a long string that contains entities and categories
                        related_yago_entities = yago.query(yago_entity, [])
                        #print("related_yago_entities", related_yago_entities)
                        #exit()
                        
                        ## first pattern
                        extracted_category_pattern = re.compile(" '<wikicat_(.*?)>'")
                        related_yago_wikicat_entities = re.findall(extracted_category_pattern, str(related_yago_entities))
                        #print ("number of matched entities:", len(related_yago_wikicat_entities))
                        # yago contains many duplicates, remove them, then replace _ with spaces
                        related_yago_wikicat_entities= list(set(related_yago_wikicat_entities))
                        related_yago_wikicat_entities= [item.replace("_", " ").replace("'", " ") for item in related_yago_wikicat_entities]
                        
                        ## second pattern 
                        #extracted_category_pattern_2 = re.compile(", '(.*?)@eng'")
                        #related_yago_wikicat_entities_2 = re.findall(extracted_category_pattern_2, str(related_yago_entities))
                        related_yago_wikicat_entities_2 = f(str(related_yago_entities), "@eng")
                        #related_yago_wikicat_entities_2 = re.sub('[!,*)#%(&$_?.^]', '', str(related_yago_wikicat_entities_2)) #remove special characters except spaces
                        related_yago_wikicat_entities_2 = list(set(related_yago_wikicat_entities_2))
                        related_yago_wikicat_entities_2 = [item.replace("_", " ").replace("@eng", "").replace("'", "").replace(")", " ").replace("(", " ").replace("\"", " ") for item in related_yago_wikicat_entities_2]
                        
                        print ("number of matched entities without duplicates:", len(related_yago_wikicat_entities_2))
                        #print ("related wikipedia categories:", related_yago_wikicat_entities_2 ,  "\n" )
                        related_yago_entities_csvfile.write(",".join(related_yago_wikicat_entities+related_yago_wikicat_entities_2) + '\n')
                        #exit()

                else:
                    #print (timestamp_dict)
                    zero_count += 1

    print ("Non zero count", non_zero_count)
    print ("Zero count", zero_count)
    print ("Number of non-zero entities", non_zero_count,   "out of which", entities_detected_in_yago_counter, "are detected in yago")
    
    # exit()
    
        
        
        
        #extracted_category_pattern = re.compile(str(related_yago_entities))
        #result = extracted_category_pattern.search(entities_detected_in_yago)
        #    print ("there is a time series for the related concept")
        #if result:      
        
    
    #time.sleep(6)

    return frequency_time_series_dict


def create_embedded_time_series(yago_output="./frequency_time_series_data_yago.csv", full_time_stamps="./twitter_timestamps-main.csv", related_entities_from_yago="./results/related_yago_entities.csv", embedded_timeseries_yago_entities="./results/embedded_timeseries_yago_entities.csv" ): 

    # Here, we open the file of the non zero entities and the file of the related related_yago_entities_csvfile
    print ("Generating the embedded time series of yago entities \n") 
    entity_timestamps_dict = {}
    with open(full_time_stamps, 'r') as full_entities_csvfile,  open(yago_output, 'r') as yago_csvfile, open(related_entities_from_yago, 'r') as related_yago_entities_csvfile, open(embedded_timeseries_yago_entities, 'w') as embedded_timeseries_yago_entities_csvfile : 
        
        entities_detected_in_yago = [line.split(',') for line in yago_csvfile]
        #entities_detected_in_yago_without_timeseries = entities_detected_in_yago[0]
        #entities_detected_in_yago_timeseries = entities_detected_in_yago[1:]
        related_yago_entities = [line.split(',') for line in related_yago_entities_csvfile]
        
        full_entities = [line.split(',')[0] for line in full_entities_csvfile]
     
     
        #print ("Yago entity:", entities_detected_in_yago[1])
        #print ("its related concepts", related_yago_entities[1])
        #print (len(related_yago_entities[1]))
        
        
        
        #for each yago conept
        #split = line.split(',')
        #entity = split[0]
        #timeseries= split[1:]
        
        
        for i, detected_yago_entity in enumerate(entities_detected_in_yago):             
            # print ("detected_yago_entity", detected_yago_entity)
            # for each related concepts
            # print ("checking if there is timeseries for the related entities of:", i, detected_yago_entity)  
            # print ("len(related_yago_entities[i])", len(related_yago_entities[i])) 
            
            ### return the related entities of this yago entitiy 
            for related_entity in related_yago_entities[i]: 
                #search for related_entity in entities_detected_in_yago_without_timeseries
                # check if there is a time series 
                #print ("********* check if this related_entity has timeseries", related_entity)
                
                if related_entity in full_entities: 
                    if related_entity: # if not an empty string
                        # add to the embedded time series
                        print("checking if there is timeseries for the related entities of:", i, detected_yago_entity)
                        print("yes, the following exists:", related_entity)
                        # print ("found in:", i,  related_yago_entities[i]) 
                #else:
                    #print("no")
            
                #detected_yago_entity_id += 1 
                #exit()
 

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
    parser.add_argument("-yo", 
                    type=str,
                    help="output path for the resulting merged yago frequency time series csv file",
                    dest="yago_output_path",
                    default="./frequency_time_series_data_yago.csv")
                    
    parser.add_argument("-fo", 
                    type=str,
                    help="output path for the full timeseries of all detected entities",
                    dest="full_output_path",
                    default="./twitter_timestamps-main.csv")
                    
    parser.add_argument("-ro", 
                    type=str,
                    help="output path for the related yago entities of each detected entity",
                    dest="related_output_path",
                    default="./results/related_yago_entities.csv")
                    
    parser.add_argument("-ero", 
                    type=str,
                    help="output path for the embedded timeseries of related yago entities of each detected entity",
                    dest="embedded_timeseries_related_output_path",
                    default="./results/embedded_timeseries_yago_entities.csv")
    
    args = parser.parse_args()

    train_split = 0.8

    #create_frequency_time_series(args.input_file, timestep=args.timestep, threshold=args.threshold, output=args.output_path)
    create_frequency_time_series_and_related_yago_entities(args.input_file, timestep=args.timestep, threshold=args.threshold, output=args.output_path, yago_output=args.yago_output_path)
    create_embedded_time_series(yago_output=args.yago_output_path, full_time_stamps=args.full_output_path, related_entities_from_yago=args.related_output_path, embedded_timeseries_yago_entities=args.embedded_timeseries_related_output_path)

    create_train_test_valid_split(args.output_path, train_split) 
    
    
if __name__ == '__main__':
    main()

