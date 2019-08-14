# -*- coding: utf-8 -*-
import argparse
import os

from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import pandas as pd

from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

from trendify.attention_utils import get_activations, get_data_recurrent, get_trend_data

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


INPUT_DIM = 1 #2
TIME_STEPS = 20
# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = True


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units = 32
    lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model
 

 
 
def apply_attention_model(): 

    N = 300000
    # N = 300 -> too few = no training
    
    # Here i will use the output 
    #inputs_1, outputs = get_data_recurrent(N, TIME_STEPS, INPUT_DIM)
    inputs_1, outputs = get_trend_data(N, TIME_STEPS, INPUT_DIM)
    
    #print (inputs_1, outputs)
    #exit()

    
    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm()

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #print(m.summary())

    m.fit([inputs_1], outputs, epochs=1, batch_size=64, validation_split=0.1)

    attention_vectors = []
    for i in range(1):
        testing_inputs_1, testing_outputs = get_trend_data(1, TIME_STEPS, INPUT_DIM)
        print (testing_inputs_1, testing_outputs)
        attention_vector = np.mean(get_activations(m,
                                                   testing_inputs_1,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        #print('attention =', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    # plot part.
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                         title='Attention Mechanism as '
                                                                               'a function of input'
                                                                               ' dimensions.')
    
    plt.savefig('results/attention_model_example.png')
    plt.show()
 

 

def main():
    """ Parses command line arguments and either performs indexing or
    partial doc update operations
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--inputfile", type=str, default="fulltext",
                        help="specify the field for txt files defaults to 'fulltext'")
    
    args = parser.parse_args()
    #if args.inputfile:
    #    print(args.inputfile)
    #    exit(1)
    
    #encoder_decoder_with_attention()
    apply_attention_model ()
    
    
    
    
if __name__ == '__main__':
    main()