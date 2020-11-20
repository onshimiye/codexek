import streamlit as st
import streamlit.components.v1 as components
import SessionState
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


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

def shuffle_dataset(dataset):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    st.info('Dataset reshuffled!')

def plot_boxandwhisker(dataset, column):
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=dataset[column])
    st.pyplot()

def plot_histograms(dataset, column):
    sns.set_theme(style="whitegrid")
    # sns.boxplot(data=dataset[column])
    sns.histplot(data=dataset[column])
    st.pyplot()

def plot_barchart(dataset, column):
    sns.set_theme(style="whitegrid")
    # sns.boxplot(data=dataset[column])
    sns.barplot(data=dataset[column])
    st.pyplot()

def plot_piechart(dataset, column):
    pie, ax = plt.subplots(figsize=[10,6])
    labels = dataset[column].unique().tolist()
    data = []
    for each in labels:
        data.append(len(dataset[dataset[column] == each]))

    plt.pie(x=data, autopct="%.1f%%", labels=labels, pctdistance=0.5)
    plt.plot()
    st.pyplot()

def correlation_graph(dataset, columns=None):
    if columns == None:
        columns = dataset.columns.values.tolist()
    plt.figure(figsize=(12,10))
    cor = dataset[columns].corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.show()
    st.pyplot()

def label_encode(dataset, columns):
    label_encoder = LabelEncoder()
    for each in columns:
        dataset[each] = label_encoder.fit_transform(dataset[each].astype('str'))
    st.info('Data encoded. Click below to see the new stats')
    understand_dataset(dataset)
    

def column_type(dataset, column):
    return dataset[column].dtype.type

def categorical_columns(dataset):
    res = []
    for each in dataset.columns.values.tolist():
        if str(dataset[each].dtype) == 'object':
            res.append(each)

    return res 

def model_type(df, target):
    if len(df[target].unique().tolist()) > 5:
        return 'linear'
    else:
        return 'logistic'

def suggest_graphs(dataset, column):
    if column is None:
        return None
    else:
        if 'numpy.float' in str(column_type(dataset, column)) or  'numpy.int' in str(column_type(dataset, column)):
            return ['Box-and-whisker', 'Histograms']
        else: return ['Pie Chart',]







# Initializations


default_datasets = [
    # Dataset('Credit Card Fraud Detection', 'creditcard.csv',"https://www.kaggle.com/mlg-ulb/creditcardfraud",  "The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."),
    Dataset('Titanic: Machine Learning from Disaster', 'titanic.csv',"https://www.kaggle.com/c/titanic/data", 'For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.'),
    Dataset('House Prices: Advanced Regression Techniques', 'house_prices.csv',"https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data", 'Predict houses sales prices'),
    Dataset('Online News Popularity', 'news_popularity.csv',"https://www.kaggle.com/c/hackafricaai/data", 'You have been provided with 30000 news in training set and 9644 in test set. Your goal would be to predict shares given all the feautures in data field.'),
    # Dataset('Santander Customer Satisfaction', 'customer_satisfaction.csv',"https://www.kaggle.com/c/santander-customer-satisfaction/data", 'You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers. The task is to predict the probability that each customer in the test set is an unsatisfied customer.')
]

sections = ["Introduction", "Datasets", "Data Analysis", "Feature Engineering", "Train Model", "Predictions"]

sst = SessionState.get(dataset=None)

# UI

# Page setup

st.set_page_config(
    page_title="CodExek: Data Analysis and Machine Learning Tutorials Platform",
    page_icon="assets/logo.svg",
    layout='wide',
    # initial_sidebar_state="collapsed"
    )
st.set_option('deprecation.showPyplotGlobalUse', False)


# Sidebar Navigation

st.sidebar.image('assets/logo_flat.svg')

st.sidebar.title("Section:")
for each in sections:
    st.sidebar.markdown(f"<a href='#{''.join(each.split())}_section' style='text-decoration:none; font-style: normal; font-weight: 500; font-size: 18px; color: #012C3D;'>{each} </a>", unsafe_allow_html=True)




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
            sst.dataset = load_dataset(each.name)
            st.write('Dataset Description')
            st.write('{}'.format(each.description, each.source))
            st.write('Source: {}'.format( each.source))
            break


elif (dataset_option == 'Custom Datasets'):
    uploaded_file = st.file_uploader('Please upload csv file you would like to use here')

    if uploaded_file is not None:
        sst.dataset = pd.read_csv(uploaded_file)

   

# Data Analysis
new_section('Data Analysis')
dataset = sst.dataset


if (dataset is None):
    st.write('No Dataset selected yet!')
    st.write('Select one of the default datasets or upload your own to proceed.')
    st.stop()


st.write('Below is a quick understanding of the dataset')
understand_dataset(dataset)


# Visualizations
new_section('Visualizations')
dataset = sst.dataset

st.write('They are different visualization techniques and they all depend on what kind of data you want to visualize. Select columns you would like to visualize, and we will recomend a graph for it')
rec = st.selectbox('Chose a column you would like to visualize', dataset.columns)

if rec is None:
    st.stop()



graph = st.selectbox('Suggested graphs', suggest_graphs(dataset, rec))

if graph is None:
    st.stop()

if graph is 'Box-and-whisker':
    plot_boxandwhisker(dataset, rec)

elif graph is 'Histograms':
    plot_histograms(dataset, rec)

elif graph is 'Pie Chart':
    plot_piechart(dataset, rec)










# Feature Engineering
new_section('Feature Engineering')
st.write('This section is still under construction')

st.markdown("### Correlation identifications")
st.write('Choose how you would like to display correlational matrix below')


option = st.radio('Want to load specific columns?', ['Yes', 'No'])
if option == 'No':
    correlation_graph(dataset)

else:
    corr_columns = st.multiselect('Which columns you would like to show in correlational matrix?', dataset.columns.values.tolist())
    if len(corr_columns) == 0:
        st.stop() 
    correlation_graph(dataset, columns=corr_columns)

# relevant features extraction


# Categorical data conversion
st.markdown("### Categorical data encoding")
cat_columns = categorical_columns(dataset)
corr_columns = st.multiselect('Which columns you would like to encode to numerical values?', cat_columns)
label_encode(dataset, corr_columns)


# Data normalization and scaling




# Train Model
new_section('Train Model')
dataset = sst.dataset


target = st.selectbox('Please select the target column or the column to predict', dataset.columns.values.tolist())

# identify which model to use 



# Reshuffling
if st.button('Suffle Dataset'):
    shuffle_dataset(dataset)


# Data Splitting 

# test_size = st.number_input('How big do you want your test dataset to be', 0, 50, 20)
# test_size /= 100
# print(test_size)
label = dataset[target]
df = dataset.drop([target], axis=1)
# x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.2, stratify=target)

# if st.button('Train set'):
#     understand_dataset(x_train)
# if st.button('Test set'):
#     understand_dataset(x_test)

# https://stackoverflow.com/a/13623707/13988391




# Model fitting

mt = model_type(dataset, target)
if mt == 'linear':
    st.write('The recommended model for this dataset is Liniear Regression')
    st.write('Possible Tunings')

    fi = st.text_input('fit_intercept', 'True')
    nr = st.text_input('normalize', 'False')
    # nj = st.number_input('n_jobs', 1, min_value=-1, max_value=1)

    clf = LinearRegression(fit_intercept=fi, normalize=nr, n_jobs=None)
    clf.fit(df, label)
    st.info('Model fit successfully')

elif mt == 'logistic':
    st.write('The recommended model for this dataset is Logistic Regression')
    st.write('Possible tunings')
    p = st.text_input('penalty', 'l2')
    fi = st.text_input('fit_intercept', 'True')
    # mi = st.number_input('max_iter', 0,  max_value=1000.0)
    rs = st.text_input('random_state', 'True')
    
    clf = LogisticRegression(random_state=rs, penalty=p, fit_intercept=fi, max_iter=0)
    clf.fit(df, label)
    st.info('Model fit successfully')
    



# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Model accuracy assessment


# Predictions
new_section('Predictions')
dataset = sst.dataset

# Predictions



# exporting model
