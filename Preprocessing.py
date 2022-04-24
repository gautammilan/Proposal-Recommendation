from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer


def encoding(column,text):
    t=Tokenizer()
    t.fit_on_texts(column)#Creating the hash table using words in texts
    sequence=t.texts_to_sequences([text])#now returning the value of hash table for each words
    return sequence

def preprocess(column,text,column_name,padding="NO"):

    encoded_text= encoding(column,text)[0]

    if padding!="NO":
        if len(encoded_text)<5:
                for i in range(5 - len(encoded_text)):
                    encoded_text.append(0)

    array_text= np.array(encoded_text)
    array_text = array_text.reshape([1,array_text.size])

    return  array_text
