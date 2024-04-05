import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
import streamlit as st

log_model = pickle.load(open("Logistic_Model.pkl", 'rb'))
MinMaxScaler = pickle.load(open("MinMaxScaler.pkl", 'rb'))
def predict():
    pass


