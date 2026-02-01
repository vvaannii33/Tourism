import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Vvaannii33/Tourism-Package-Creation", filename="tourism_package_creation_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for travel companies that implements a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies.")
st.write("Kindly enter the tourist details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100)
TypeofContact = st.selectbox("The method by which the customer was contacted", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("The city category", ["Tier 1", "Tier 2","Tier 3"])
Occupation = st.selectbox("Customer's occupation", ["Salaried", "Small Business","Large Business","Free Lancer"])
Gender = st.selectbox("Gender of the customer", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer")
PreferredPropertyStar = st.selectbox("Preferred hotel rating by the customer.", [3.0,4.0,5.0])
MaritalStatus = st.selectbox("Marital status of the customer", ["Married", "Divorced","Unmarried","Single"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually.")
Passport = st.selectbox("Whether the customer holds a valid passport (0: No, 1: Yes)", [0,1])
OwnCar = st.selectbox("Whether the customer owns a car (0: No, 1: Yes).", [0,1])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer.")
MonthlyIncome = st.number_input("Gross monthly income of the customer.")
PitchSatisfactionScore = st.selectbox("Score indicating the customer's satisfaction with the sales pitch.", [1,2,3,4,5])
ProductPitched = st.selectbox("The type of product pitched to the customer.", ["Basic", "Deluxe","Standard","Super Deluxe","King"])
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch")
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer.")


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch

}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "take product" if prediction == 1 else "not take product"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
