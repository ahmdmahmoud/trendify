import keras.backend as K
import numpy as np
from random import randint
import gzip
import pandas as pd 
import re, os 
import time 


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def gzip_file_reader():

	directory = '/data21/asaleh/wikipedia-pageview-stats/2017/2017-01/'
	
	total_pageviews_matrix_en = pd.DataFrame (index=None,  columns = ['language', 'topic', 'views', 'unknown', 'timeStamp' ])
	for file_dir in os.listdir(directory): 
		if file_dir.endswith(".gz"): 
			start_time = time.time()
			print ('Processing the following file: ', directory+file_dir)
			f=gzip.open(directory+file_dir,'rb')
			
			# Extract the time stamp from the file names of wikipedia dataset. The following convention is used "pageviews-date-time.gz" -> e.g. 
			time_stamp_np = re.search('pageviews-(.+?)-(.+?).gz', file_dir)
			if time_stamp_np:
				time_stamp = pd.to_datetime(time_stamp_np.group(1)+time_stamp_np.group(2), format='%Y%m%d:%H:%M:%S', errors='ignore')
				
			# pageviews pandas dataframe
			file_content=f.read().decode("utf-8") 
			pageviews_matrix_np = [ line.split(' ')    for line in file_content.splitlines()] 
			pageviews_matrix = pd.DataFrame ( pageviews_matrix_np, index=None,  columns = ['language', 'topic', 'views', 'unknown'] )
			
			# keep only the content from English pages 
			pageviews_matrix_en = pageviews_matrix[pageviews_matrix['language']=='en']
			
			# Add the time stamp column 
			pageviews_matrix_en['timeStamp'] =  time_stamp
			
			# Append to the total entities array
			total_pageviews_matrix_en = pd.concat([total_pageviews_matrix_en, pageviews_matrix_en])
			
			
			end_time = time.time()
			print("total time for processing this file: ", time.time() - start_time)
			print("Number of records in this file: {0}, \t  total: {1} ".format( len(pageviews_matrix_en), len(total_pageviews_matrix_en)))
	
	
	
	# count the frequency of the entities/date
	print(total_pageviews_matrix_en.groupby(['timeStamp', 'topic']).size())
			
			
			
def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y
    
def get_trend_data(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    #y = np.random.randint(low=0, high=2, size=(n, 1))
    y = [ [np.amax(input_list, axis=0)[0]] for input_list in x ]
    #print (x, y)
    #exit ()
    
    # replaced column number 10 with zero to match the output 
    #x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    #print (x, y)
    #exit ()
    
    return x, np.reshape(y, (n, 1) ) 