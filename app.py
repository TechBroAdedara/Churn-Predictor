import pickle
import streamlit as st
from preprocessing import preprocess

log_model = pickle.load(open("pickle_files/Logistic_Model.pkl", 'rb'))
 

def predict():
    pass

def main():
    
    with st.sidebar:
            st.title("ChurnSight")
            st.markdown("""Empower your business with predictive insights. 
                            Anticipate churn. 
                            Retain customers. Thrive.""")
        
    st.subheader("Please input the details of your customer in the respective sections", divider="rainbow")

    #demographic data selection
    st.subheader("Demographic Data")
    dependent = st.selectbox("Dependent:", ("Yes", "No"))
    seniorcitizen = st.selectbox("Senior Citizen:", ("Yes", "No"))

    #Payment Method Selection

    data = {
            'SeniorCitizen':seniorcitizen.lower(),
            'Dependents': dependent.lower()
            }
    
    st.write(data['Dependents'])


if __name__ == "__main__":
    main()