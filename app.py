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

from sklearn.model_selection import learning_curve
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.model_selection import validation_curve

from modules.deep_learning import deep_learning


# Classes

class Dataset:
    def __init__(self, display_name, name, source, description):
        self.display_name = display_name
        self.name = name
        self.source = source
        self.description = description

    def __str__(self):
        return self.name

class CorrelationColumn:
    def __init__(self, index, name, value):
        self.index = index
        self.name = name
        self.value = value

    def __str__(self):
        return '{} (corr: {})'.format(self.name, self.value)
    
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
    return cor

def label_encode(dataset, columns):
    label_encoder = LabelEncoder()
    for each in columns:
        dataset[each] = label_encoder.fit_transform(dataset[each].astype('str'))
    st.info('Data encoded. Below are the new stats')
    understand_dataset(dataset)
    

def column_type(dataset, column):
    return dataset[column].dtype.type

def categorical_columns(dataset):
    res = []
    for each in dataset.columns.values.tolist():
        if str(dataset[each].dtype) == 'object':
            res.append(each)

    return res 

def numerical_columns(dataset):
    res = []
    for each in dataset.columns.values.tolist():
        if 'numpy.float' in str(column_type(dataset, each)) or  'numpy.int' in str(column_type(dataset, each)):
            res.append(each)

    return res 

def has_nulls(dataset):
    res = []
    for each in dataset.columns.values.tolist():
        if dataset[each].isnull():
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

sst = SessionState.get(dataset=None, df=None)

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

dataset_option = st.selectbox('What datasets would you like to continue with',['Default Datasets', 'Custom Datasets', 'Deep Learning'])

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

elif (dataset_option == 'Deep Learning'):
    st.write("This shows you how to use deep learning models. It uses a classic example of spam detection model. Enjoy")

    deep_learning()
    st.stop()


   

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

if graph == 'Box-and-whisker':
    plot_boxandwhisker(dataset, rec)

elif graph == 'Histograms':
    plot_histograms(dataset, rec)

elif graph == 'Pie Chart':
    plot_piechart(dataset, rec)





# Feature Engineering
new_section('Feature Engineering')
st.write('This section is still under construction')


# Categorical data conversion
st.markdown("### Categorical data encoding")
cat_columns = categorical_columns(dataset)
corr_columns = st.multiselect('Which columns you would like to encode to numerical values?', cat_columns)
label_encode(dataset, corr_columns)



st.markdown("### Correlation identifications")
st.write('Choose how you would like to display correlational matrix below')


option = st.radio('Want to load specific columns?', ['Yes', 'No'])
if option == 'No':
    cor = correlation_graph(dataset)

else:
    # corr_columns = []
    corr_columns = st.multiselect('Which columns you would like to show in correlational matrix?', dataset.columns.values.tolist())
    if len(corr_columns) == 0:
        st.stop() 
    cor = correlation_graph(dataset, columns=corr_columns)


# relevant features extraction

st.markdown("### Relevant features extraction")

target = st.selectbox('Please select the target column or the column to predict', dataset.columns.values.tolist())

st.write('Below are all columns numerical columns and their correlation to the target column. Please pick those that you believe have a significant correlation value.')
correlation_values = cor[target].reset_index(name='values')

corr_classes = []

for each in correlation_values.itertuples():
    if each.index == target:
        continue
    corr_classes.append(CorrelationColumn(each.Index, each.index, round(each.values, 2)))

selected_columns = st.multiselect('Select columns by their correlation to the target', list(corr_classes))


# st.write(list(set().union(numerical_columns(dataset), corr_columns)))

other_numerical_columns = st.multiselect('Select other numerical columns you would like to train with', list(set().union(numerical_columns(dataset), corr_columns)))


df = dataset[list(set().union([x.name for x in selected_columns], other_numerical_columns, [target]))]
df = df.dropna()
sst.df = df



# Data normalization and scaling




# Train Model
new_section('Train Model')
dataset = sst.dataset
df = sst.df



# identify which model to use 



# Reshuffling
if st.button('Suffle Dataset'):
    shuffle_dataset(dataset)




# Data Splitting 

test_size = st.number_input('How big do you want your test dataset to be', 0, 50, 20)
test_size /= 100
print(test_size)

st.write(df)
label = df[target]
st.write(label)


x = df.drop([target], axis=1)
st.write(x)

x_train, x_test, y_train, y_test = train_test_split(x, label, test_size=test_size)

# st.write(x_train)
# st.write(x_test)

if st.button('Train set'):
    understand_dataset(x_train)
if st.button('Test set'):
    understand_dataset(x_test)

# https://stackoverflow.com/a/13623707/13988391





# Model fitting

st.markdown("### Model Fitting")
mt = model_type(dataset, target)
if mt == 'linear':
    st.write('The recommended model for this dataset is Liniear Regression')
    st.write('Possible tunings. Check the documentation for more details: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html')
    st.write()

    fi = st.selectbox('fit_intercept', [True, False])
    nr = st.selectbox('normalize', [True, False])
    nj = st.number_input('n_jobs', -1,1, 1)

    clf = LinearRegression(fit_intercept=fi, normalize=nr, n_jobs=nj)
    clf.fit(x_train, y_train)
    st.info('Model fit successfully')
    st.write(clf)

elif mt == 'logistic':
    st.write('The recommended model for this dataset is Logistic Regression')
    st.write('Possible tunings. Check the documentation for more details: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html')
    p = st.selectbox('penalty', ['l1', 'l2', 'elasticnet', 'none'])
    fi = st.selectbox('fit_intercept', [True, False])
    dual = st.selectbox('dual', [True, False])
    c = st.number_input('C', 0, 100, 1)
    
    i_s = st.number_input('intercept_scaling', 0, 100, 1)
    rs = st.number_input('random_state', 0, 1, 0)
    mi = st.number_input('max iter', 0, 1000000, 100)

    st.write(len(x_train), len(y_train))
    
    clf = LogisticRegression(dual=dual, C=c, intercept_scaling=i_s, random_state=rs, penalty=p, fit_intercept=fi, max_iter=0)
    clf.fit(x_train, y_train)
    st.info('Model fit successfully')
    st.write(clf)
    


st.markdown("### Model Evaluation")
y_test_pred = clf.predict(x_test)


st.write("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2)) 
st.write("Root mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2)) 
st.write("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2)) 
st.write("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2)) 
if mt == 'logistic':
    st.write("Accuracy Score =", round(sm.accuracy_score(y_test, y_test_pred), 2))
st.write("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))



st.markdown("### Model Optimisation")
st.write('You can go back to the model fitting to tune the parameters for a better accuracy')

if mt == 'linear':
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_test_pred, edgecolors=(0, 0, 0))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()
    st.pyplot()

alphas = np.logspace(-2, 0, 20)
sgd_logit = SGDClassifier(loss='log', n_jobs=-1, random_state=17)
logit_pipe = Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), 
                    ('sgd_logit', sgd_logit)])
val_train, val_test = validation_curve(logit_pipe, x, label,
                                    'sgd_logit__alpha', alphas, cv=5,
                                    scoring='roc_auc')

if mt == 'logistic':
    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, '-', **kwargs)
        plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                        facecolor=lines[0].get_color(), alpha=0.2)

    plot_with_err(alphas, val_train, label='training scores')
    plot_with_err(alphas, val_test, label='validation scores')
    plt.xlabel(r'$\alpha$'); plt.ylabel('ROC AUC')
    plt.legend()
    st.pyplot()





# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

# Model accuracy assessment


# Predictions
new_section('Predictions')
dataset = sst.dataset

# Predictions

arr_to_predict = []
for each in df.columns:
    if each == target: continue
    inpt = st.number_input(each)
    arr_to_predict.append(inpt)

arr_to_predict = np.array([arr_to_predict]).astype(np.float64).round()

st.write(clf.predict(arr_to_predict))

import pickle
import base64

# x = {"my": "data"}

def download_model(model):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download="model.pkl">Download Trained Model .pkl File</a>'
    st.markdown(href, unsafe_allow_html=True)


download_model(model=clf)

# exporting model
