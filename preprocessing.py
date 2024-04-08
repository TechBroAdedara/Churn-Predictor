import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

sc = joblib.load("pickle_files/scaler.save")

unique_categories = {
    'InternetService': ['DSL', 'Fiber optic'],
    'Contract' : ['Month-to-month','One year', 'Two year'], 
    'PaymentMethod' : ['Mailed check', 'Electronic check', 'Bank transfer (automatic)',
                       'Credit card (automatic)']
}

def preprocess_data(data_df, unique_categories=unique_categories):
    # Drop unnecessary columns
    #data_df = data_df.drop("customerID", axis=1)
    
    # Define binary mapping function
    def binaryMap(feature):
        return feature.map({"Yes": 1, "No":0})

    
    binary_features_list = ['SeniorCitizen', 'Dependents','PhoneService', 'MultipleLines','OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling']
    data_df[binary_features_list] = data_df[binary_features_list].apply(binaryMap)
    
    # Create dummy variables
    if unique_categories:
        for column, categories in unique_categories.items():
            data_df[column] = pd.Categorical(data_df[column], categories=categories)
    data_df = pd.get_dummies(data_df, dtype=int)
    
    # Normalize numerical features
    data_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = sc.transform(data_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    useful_columns = ['SeniorCitizen',
                        'Dependents',
                        'tenure',
                        'PhoneService',
                        'MultipleLines',
                        'OnlineSecurity',
                        'TechSupport',
                        'StreamingTV',
                        'StreamingMovies',
                        'PaperlessBilling',
                        'MonthlyCharges',
                        'TotalCharges',
                        'InternetService_DSL',
                        'InternetService_Fiber optic',
                        'Contract_Month-to-month',
                        'Contract_Two year',
                        'PaymentMethod_Electronic check']
                    
    data_df = data_df.loc[:,useful_columns]
    
    return data_df
