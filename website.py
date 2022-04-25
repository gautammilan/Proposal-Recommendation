
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from model import createModel,encoding
import pickle
from Preprocessing import preprocess



#LOADING THE PRETRAINED MODEL
@st.cache(suppress_st_warning=True)
def read_dataset():
 dataframe=pd.read_csv('preprocessed_data.csv')
#  y=dataframe['project_is_approved']
#  X=dataframe.drop(columns='project_is_approved')
 return dataframe


@st.cache(suppress_st_warning=True)
def getting_model():
    model= createModel()
    model.load_weights('backup')
    return model

#Reading the dataframe and getting the model
def spaces(a):
  for i in range(a):
    st.text(" ")
  return

X= read_dataset()
model= getting_model()

st.title("""Select a task""")
task= st.selectbox('Task',["Prediction","ViewModel"])


@st.cache(suppress_st_warning=True)
def loading_info():
  with open('total_categories','rb') as f:
    categories= pickle.load(f)

  with open('total_subcategories','rb') as f:
    subcategoies= pickle.load(f)

  with open('states', 'rb') as f:
    states= pickle.load(f)
  return categories,subcategoies,states

categories,subcategoies,states= loading_info()


if task == "Prediction":

  for i in range(4):
    st.text(" ")

  st.header("Please fill the following information")

  #Code for getting the discription of the project
  project_title = st.text_input("Enter the title of the project")
  projectgradecategory = st.selectbox("Select the grade for which the project is targeted", ('grades_3_5', 'grades_9_12', 'grades_prek_2', 'grades_6_8'))
  projectsubjectcategory= st.multiselect("Category of the project: ",categories)

  projectsubjectsubcategory= st.multiselect("SUbCategory of the project(*Optional): ",subcategoies)
  essay= st.text_input("Essay of the applicant")
  teacherpreviousprojects= st.text_input("How many times did you previous submitted proposal for the projects")
  schoolstate = st.selectbox("Enter the state where you school is located: ",states)
  title = st.radio("Select your title:", ('Dr','Mr','Mrs','Ms','Teacher'))


  for i in range(10):
      st.text(" ")

  #Code for getting the description of the resources of the project
  st.header("Please provide information about the resources you need for your project")
  description= name = st.text_input("Description of required resources")
  quantity = st.text_input("Quantity")
  price= st.text_input("Total price of resources")

  #Doing some datapreprocessing
  cleanCategories= '_'.join(projectsubjectcategory)
  cleanSubCategories= '_'.join(projectsubjectsubcategory)


  if(st.button('Submit')):

    #DOING THE PREDICTION
    # allow= True

    print('We are in the preprocessing')
    pred_schoolstate = preprocess(X['school_state'], schoolstate, 'schoolstate')
    pred_title = preprocess(X['teacher_prefix'], title, 'title')
    pred_projectgradecategory = preprocess(X['project_grade_category'], projectgradecategory, 'projectgradecategory')
    print("The project grade category is:",projectgradecategory)
    pred_cleanCategories = preprocess(X['clean_categories'], 'math_science_literacy', cleanCategories, padding="YES")
    pred_cleanSubCategories = preprocess(X['clean_subcategories'], 'geography', cleanSubCategories, padding="YES")

    pred_teacherpreviousprojects = np.array(teacherpreviousprojects).astype('float32')
    pred_teacherpreviousprojects = pred_teacherpreviousprojects.reshape([pred_teacherpreviousprojects.size, 1])

    pred_price = np.array(price).astype('float32')
    pred_price = pred_price.reshape([pred_price.size, 1])

    input_dictionary = {
      'school_state_input': pred_schoolstate,
      'teacher_prefix_input': pred_title,
      'project_grade_input': pred_projectgradecategory,
      'clean_categories_input': pred_cleanCategories,
      'clean_subcategories_input': pred_cleanSubCategories,
      'price': pred_price,
      'previous_no_project': pred_teacherpreviousprojects
    }


    #Getting the prediction from the model
    prediction = model.predict(input_dictionary)
    if prediction>0.5:
      
      Label= 1
    else:
      Label= 0

    st.subheader("The chance that this application will be selected is: ")
    prediction= str(int(prediction[0][0]*100))+'%'
    st.title(prediction)


else:
  spaces(4)
  intro= "DonorsChoose is a United States-based nonprofit organization that allows individuals to donate \ndirectly to public school classroom projects. According to last year data around 500,000 proposals \nwere sent by teachers all around the world to DonorsChoose hoping to get the donation for their \nrespective projects. Organization goes through every single project proposal to select those \nproposals which have a higher nprobability of getting donations. After selecting the project Donor \nChoose organization will display the selected project on their website allowing the donor to go \nthrough the project which they are likely to donate. Here the main problem is going through all the \nproject proposal equired a large number of resources and cost a tremendous amount of money."

  #Defining the introduction
  st.header("Introduction")
  st.text(intro)
  spaces(4)

  #Defining the objective
  objec= "A huge amount of time and money is required to go through all the project proposals, so by applying \ndeep learning model the objective of this project is to automize the task of selecting those proposal \nwhich have high possibility of getting donation. Model will produce an output for every single proposal, \nwhere output '1' indicates that the proposal has high chance of acceptance and '0' indicates the \nrejection probability"
  st.header("Objective")
  st.text(objec)
  spaces(4)

  #Describing the data
  data_intro= "As many teachers around the world send their proposals seeking the grant, the dataset contains the \ninformation about the person sending the proposal and the price of the grant needed. Such as \nschool_state(where the teacher school is located), teacher_prefix(Dr, Mr, Mrs, Ms, teacher), \nproject_grade_category, the previous number of the proposal submitted on that project, \nproject_is_approved, categories, sub_categories, essay, requested price and finally the label \nindicating the acceptance of the proposal."
  st.header("Understanding the Data")
  st.text(data_intro)
  spaces(2)

  if st.checkbox("Click to see the data"):
    st.dataframe(X.head(10))


  #Describing the model
  model_intro= "As many features can be used to train the model, so it was necessary to select an architecture that \ncan incorporate all of these important features. Three are three types of the feature present \nin the dataset like:"
  categorical_intro= "1) Categorical Feature: \nschool_state, teacher_prefix, project_grade_category,  categories, and \nsub_categories are the categorical feature present in the dataset. So it was necessary to add \nan embedding layer to each of these features. Therefore, an individual embedding layer was \nincluded in each of the features depending upon their vocabulary size."
  numeric_intro= "2)Numeric Feature: \nAs for the numeric data they where normalized between 0-1 and then used."
  text_intro= "3)Text: \nOne of the most important features of the dataset is the essay it defines why the grant is \nnecessary for the teacher and how he/she will use the grant. By using BERT a feature vector of \nsize 768 is generated and this generated feature vector is inputted to the model."
  final_intro= "Finally, all of these features are concatenated, similarly few dense layers with dropout are used \nto make sure there is no overfitting, and finally, the output is sent through a sigmoid layer to \nproduce an probability of selection of the submitted proposal."

  st.header("Model Architecture")
  st.text(model_intro)
  st.text(categorical_intro)
  st.text(numeric_intro)
  st.text(text_intro)
  st.text(final_intro)
  
  if st.checkbox("Click to see the model Architecture"):
    st.image('model.png')

  for i in range(4):
    st.text(" ")

  st.header("Evaluation")
  st.text("The task is simple a binary classification, so it was the straightforward choice of selecting \nbinary cross-entropy loss function. Similarly, the dataset is quite a balance which makes it \npossible to use accuracy as the evaluation matrix. The accuracy was quite good on both the training \nand test dataset even after training for just 9 epochs the training and test accuracy reached \n85.5% and 84% respectively. ")
  st.image('accuracy.png',caption='Tensorboard Image')
  st.text("RED:")
  st.text("BLUE:Test")
