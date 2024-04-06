import pandas as pd
import pickle

MinMaxScaler = pickle.load(open("pickle_files/MinMaxScaler.pkl","rb"))

def preprocess(data_df):
    data_df = data_df.drop("customerID", axis=1)

    def binaryMap(feature):
        return feature.map({"Yes": 1, "No":0})

    data_df['gender'] = data_df['gender'].map({'Female':0, 'Male':1})

    list_to_binarize = ['Churn', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen', 'Partner']
    data_df[list_to_binarize] = data_df[list_to_binarize].apply(binaryMap)
    data_df = pd.get_dummies(data_df, dtype=int, drop_first=True)

    #Normalizing numerical features
    data_df['tenure'] = MinMaxScaler.transform(data_df[['tenure']])
    data_df['MonthlyCharges'] = MinMaxScaler.transform(data_df[['tenure']])
    data_df['TotalCharges'] = MinMaxScaler.transform(data_df[['tenure']])