import streamlit as st
import numpy as np
import pandas as pd

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
    st.write('This shows the top 5 fields of a dataset')
    return dataset.head()

def dataset_tail(dataset):
    st.write('This shows the bottom 5 fields of a dataset')
    return dataset.tail()

def dataset_shape(dataset):
    st.write('This shows the dimensions of a dataset')
    return dataset.shape

def dataset_info(dataset):
    st.write('This shows the datatypes and count of empty fields for each column of a dataset')
    return dataset.info()

def dataset_stats(dataset):
    st.write('This shows the basic statistics of numerical columns/feature of a dataset')
    return dataset.describe()
    
def understand_dataset(dataset):
    st.write(dataset_head(dataset))
    st.write(dataset_tail(dataset))
    st.write(dataset_shape(dataset))
    st.write(dataset_info(dataset))
    st.write(dataset_stats(dataset))
    st.write(dataset_info(dataset))

st.title("Trying out Streamlit")

st.write("Choose between the available datasets")

if st.button("Iris"):
    dataset = load_dataset('iris.csv')
    understand_dataset(dataset)

elif st.button('Boston House Prices'):
    dataset = load_dataset('boston_house_prices.csv')
    understand_dataset(dataset)    

#understand_dataset(dataset)


