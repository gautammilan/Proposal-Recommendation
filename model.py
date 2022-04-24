from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras



dataframe=pd.read_csv('preprocessed_data.csv')
# print(dataframe.head())
y=dataframe['project_is_approved']
X=dataframe.drop(columns='project_is_approved')

from tensorflow.keras.preprocessing.text import Tokenizer

#Integer encoding the text data
def encoding(texts):
    t=Tokenizer()
    t.fit_on_texts(texts)#Creating the hash table using words in texts
    sequence=t.texts_to_sequences(texts)#now returning the value of hash table for each words
    return sequence

def vocabolary_size(sequence1):
    m=0
    for seq in sequence1:
       if m<max(seq):
        m=max(seq)

    return m

#encoding the all the categorical feattures
school_state=encoding(X['school_state'])

teacher_prefix=encoding(X['teacher_prefix'])

project_grade=encoding(X['project_grade_category'])

clean_categories=encoding(X['clean_categories'])

clean_subcategories=encoding(X['clean_subcategories'])


#PADDING ALL THE FEATURES WHICH HAS DIFFERENT LENGTH OF WORDS

clean_categories = tf.keras.preprocessing.sequence.pad_sequences(clean_categories,padding='post')

clean_subcategories = tf.keras.preprocessing.sequence.pad_sequences(clean_subcategories,padding='post')

# print(clean_categories)
#Converting all the list input to array
school_state=np.array(school_state)
teacher_prefix=np.array(teacher_prefix)
project_grade=np.array(project_grade)
clean_categories=np.array(clean_categories)
clean_subcategories=np.array(clean_subcategories)
price=np.array(X['price'])
previous_no_project=np.array(X['teacher_number_of_previously_posted_projects'])

#Reshaping price and previous_no_project to 2d array
price=price.reshape([price.size,1])
previous_no_project=previous_no_project.reshape([previous_no_project.size,1])



def createModel():

 # Defining the multiple_input to the model
 school_state_input = keras.Input(shape=(1),
                                  name='school_state_input')  # the number of features in shape(no of features)
 teacher_prefix_input = keras.Input(shape=(1), name='teacher_prefix_input')
 project_grade_input = keras.Input(shape=(3), name='project_grade_input')
 clean_categories_input = keras.Input(shape=(5), name='clean_categories_input')
 clean_subcategories_input = keras.Input(shape=(5), name='clean_subcategories_input')
 previous_no_project = keras.Input(shape=(1), name='previous_no_project')
 price = keras.Input(shape=(1), name='price')

 # Using Embedding
 school_state_m = layers.Embedding(max(school_state)[0] + 1, 8, input_length=1)(
  school_state_input)  # embedding word of 51 size to space of 8 dimension
 # school_state size (,1,8) has 1 word in each sequence and embedden in 8 dimension
 school_state_m = layers.Flatten()(school_state_m)
 # school_state_m=(,1*8)

 teacher_prefix_m = layers.Embedding(max(teacher_prefix)[0] + 1, 2, input_length=1)(
  teacher_prefix_input)  # embedding word of 5 size to space of 2 dimension
 # teacher_prefix size (,1,2) has 1 word in each sequence and embedden in 2 dimension
 teacher_prefix_m = layers.Flatten()(teacher_prefix_m)
 # teacher_prefix_m=(,1*2)

 project_grade_m = layers.Embedding(vocabolary_size(project_grade) + 1, 5, input_length=3)(
  project_grade_input)  # embedding each word of size 9 to space of 5 dimension
 # project_grade size (,3,5) 3d tensor has 3 words in each sequence and each word is embedded in 5 dimension
 project_grade_m = layers.Flatten()(project_grade_m)
 # project_grade size=(,3*5)

 clean_categories_m = layers.Embedding(vocabolary_size(clean_categories) + 1, 8, input_length=5)(
  clean_categories_input)  # embedding word of size 15 to space of 8 dimension
 # clean_categories size (,5,8) has 5 words in each sequence and each word is embedded in 8 dimension
 clean_categories_m = layers.Flatten()(clean_categories_m)
 # clean_categories size=(,5*8)

 clean_subcategories_m = layers.Embedding(vocabolary_size(clean_subcategories) + 1, 20, input_length=5)(
  clean_subcategories_input)  # embedding word of size 37 to space of 20 dimension
 # clean_subcategories size (,5,20) has 5 words in each sequence and each word is embedded in 20 dimension
 clean_subcategories_m = layers.Flatten()(clean_subcategories_m)
 # clean_subcategories size= (,5*20)

 # Concatenating all the feature ie adding all the features
 features = layers.concatenate(
  [school_state_m, teacher_prefix_m, project_grade_m, clean_categories_m, clean_subcategories_m, price,
   previous_no_project])
 # Feature size=(,167)

 Dense1 = layers.Dense(512, activation='relu')(features)
 # weights=(167,512) output=(,512)

 dropout = layers.Dropout(0.2)(Dense1)

 Dense2 = layers.Dense(128, activation='relu')(dropout)

 output = layers.Dense(1, activation='sigmoid')(Dense2)

 model = keras.Model(inputs=[school_state_input, teacher_prefix_input, project_grade_input, clean_categories_input,
                             clean_subcategories_input, price, previous_no_project], outputs=[output])

 model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
 return model

model= createModel()
# model.summary()

#Visualizing the model
