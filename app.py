import joblib
import streamlit as st
from preprocessing import preprocess_data
import pandas as pd

log_model = joblib.load("pickle_files/Logmodel.save")
 


def main():
    
    with st.sidebar:
            st.title("ChurnSight")
            st.markdown("""Empower your business with predictive insights. 
                            Anticipate churn. 
                            Retain customers. Thrive.""")
        
    st.subheader("Please input the details of your customer in the respective sections", divider="rainbow")

    #Payment Method Selection
    st.subheader("Payment")
    PaymentMethod = st.selectbox("Payment Method:", ('Electronic check', 
                                                     'Mailed check', 
                                                     'Bank transfer (automatic)',
                                                    'Credit card (automatic)'))
    PaperlessBilling = st.selectbox("Paperless billing:", ('Yes', 'No'))
    Contract = st.selectbox("Contract type:", ('Month-to-month','One year', 'Two year'))
    
    #Services selection
    st.subheader("Services User has signed up for")
    PhoneService = st.selectbox("User has phone service?", ('Yes', 'No'))
    MultipleLines = st.selectbox("User has multiple lines?", ('Yes','No'))
    InternetService = st.selectbox(" User has internet service?", ('DSL', 'Fiber optic'))
    OnlineSecurity = st.selectbox(" User has online security service?", ('Yes', 'No'))
    TechSupport = st.selectbox("User has tech support services?", ('Yes', 'No'))
    StreamingTV = st.selectbox("User has streaming TV services?", ('Yes', 'No'))
    StreamingMovies = st.selectbox("User has streaming movies services?", ('Yes', 'No'))

    st.subheader("Numerical Data")
    tenure = st.slider("Number of months user has stayed with company", min_value=0, max_value=72, value=0)
    monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
    totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

    data = {
                'tenure':tenure,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod':PaymentMethod,
                'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges,
                }
    df = pd.DataFrame.from_dict([data])

    st.subheader("Overview of input:")
    st.dataframe(df)
    preprocessed_df= (preprocess_data(df))
    if st.button("Make Prediction"):
        prediction = log_model.predict(preprocessed_df)
        if prediction:
             st.warning("Customer will leave")
        else:
             st.warning("Customer will stay")


if __name__ == "__main__":
    main()