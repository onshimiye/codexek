import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd


# Classes

class Dataset:
    def __init__(self, display_name, name, source, description):
        self.display_name = display_name
        self.name = name
        self.source = source
        self.description = description

    def __str__(self):
        return self.name


# Functions

def load_dataset(name):
    try:
        dataset = pd.read_csv('datasets/{}'.format(name))
        return dataset
    except Exception as err:
        print(type(err))    
        print(err.args)     
        
        print("An exception occurred")
        return None
    
def dataset_head(dataset):
    st.write('Top 5 fields of the dataset')
    return dataset.head()

def dataset_tail(dataset):
    st.write('Bottom 5 fields of the dataset')
    return dataset.tail()

def dataset_shape(dataset):
    st.write('Dimensions of the dataset')
    return dataset.shape

def dataset_stats(dataset):
    st.write('Basic statistics of numerical columns/feature of the dataset')
    return dataset.describe()
    
def understand_dataset(dataset):
    
    st.write(dataset_head(dataset))
    st.write(dataset_tail(dataset))
    st.write(dataset_shape(dataset))
    st.write(dataset_stats(dataset))

def new_section(name):
    st.markdown(f"<div id='{''.join(name.split())}_section'></div>", unsafe_allow_html=True)
    st.header(name)


# Initializations

dataset = None

default_datasets = [
    # Dataset('Credit Card Fraud Detection', 'creditcard.csv',"https://www.kaggle.com/mlg-ulb/creditcardfraud",  "The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."),
    Dataset('Titanic: Machine Learning from Disaster', 'titanic.csv',"https://www.kaggle.com/c/titanic/data", 'For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.'),
    Dataset('House Prices: Advanced Regression Techniques', 'house_prices.csv',"https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data", 'Predict houses sales prices'),
    Dataset('Online News Popularity', 'news_popularity.csv',"https://www.kaggle.com/c/hackafricaai/data", 'You have been provided with 30000 news in training set and 9644 in test set. Your goal would be to predict shares given all the feautures in data field.'),
    # Dataset('Santander Customer Satisfaction', 'customer_satisfaction.csv',"https://www.kaggle.com/c/santander-customer-satisfaction/data", 'You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers. The task is to predict the probability that each customer in the test set is an unsatisfied customer.')
]

sections = ["Introduction", "Datasets", "Data Analysis", "Feature Engineering", "Train Model", "Predictions"]


# UI
st.set_page_config(
    page_title="CodExek: Data Analysis and Machine Learning Tutorials Platform",
    page_icon="assets/logo.svg",
    layout='wide',
    initial_sidebar_state="collapsed"
    )

# Sidebar Navigation

st.sidebar.image('assets/logo_flat.svg')

st.sidebar.title("Jump to different sections:")
for each in sections:
    st.sidebar.markdown(f"<a href='#{''.join(each.split())}_section'>{each} </a>", unsafe_allow_html=True)




navbar = open("./navbar.html", 'r', encoding='utf-8').read()

# Introduction
st.markdown(f"<div id='Introduction_section'></div>", unsafe_allow_html=True)

intro = open("./intro.html", 'r', encoding='utf-8').read()
# print(source_code)
components.html(navbar)
components.html(intro, height=1250)

# Datasets
new_section('Datasets')
# st.markdown(f"<div id='Datasets_section'></div>", unsafe_allow_html=True)

# st.header("Datasets")
st.write("You can chose within our available datasets, or upload you own datasets")

dataset_option = st.selectbox('What datasets would you like to continue with',['Default Datasets', 'Custom Datasets'])

if (dataset_option == 'Default Datasets'):
    for each in default_datasets:
        if st.button(each.display_name):
            dataset = load_dataset(each.name)
            st.write('Dataset Description')
            st.write('{}'.format(each.description, each.source))
            st.write('Source: {}'.format( each.source))
            break


elif (dataset_option == 'Custom Datasets'):
    uploaded_file = st.file_uploader('Please upload csv file you would like to use here')

    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)

   

# Data Analysis
new_section('Data Analysis')


if (dataset is None):
    st.write('No Dataset selected yet!')
    st.write('Select one of the default datasets or upload your own to proceed.')
    st.stop()


st.write('Below is a quick understanding of the dataset')
understand_dataset(dataset)


# Visualizations
new_section('Visualizations')

st.write('They are different visualization techniques and they all depend on what kind of data you want to visualize.')
st.write('Below are some of the visualizations available and what kind of data input they can work with')


st.selectbox('Select any of the visualizations you would like to make',['Box-and-whisker', 'Histograms', 'Bar Chart', 'Pie Chart', 'Scatter Plot', 'Time Series'])


# Feature Engineering
new_section('Feature Engineering')
st.write('This section is still under construction')



# Train Model
new_section('Train Model')


# Predictions
new_section('Predictions')