 # code :-app.py
from multiprocessing import reduction
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model once
model = joblib.load('best_model.pkl') #not->best_model(2).pkl

gender_encoder = joblib.load('gender_encoder.pkl')
occupation_encoder = joblib.load('occupation_encoder.pkl')
workclass_encoder = joblib.load('workclass_encoder.pkl')
country_encoder = joblib.load('country_encoder.pkl')
experience = joblib.load('experience.pkl')


encoder = joblib.load("encoder.pkl")  # OneHotEncoder used in training

# Set light blue background for the entire page
st.markdown("""
    <style> .stApp {
    background-color: #ADD8E6
    };</style>
    """,unsafe_allow_html=True)

# Set title also given color to it
st.markdown("<h1 style='color: #7A2E2E; text-align: center;'>Employee Salary Predictor using Machine Learning Algorithm</h1>", unsafe_allow_html=True)

# Set header
st.header("Input Employee Details")

# Set all the attributes
age = st.number_input("Select Age", 18, 60)
gender = st.radio("Select Gender", ["Male", "Female"])
education = st.selectbox("Select Education",["Bachelors:13", "Masters:14", "Assoc-voc:11","HS-grad:9","Some-college:10","Assoc-voc:11","Assoc-acdm:12","Prof-school:15","Doctorate:16"])
occupation = st.selectbox("Select Occupation",["Prof-specialty","Craft-repair","Exec-managerial","Adm-clerical","Sales","Other-service","Machine-op-inspct","Transport-moving","Handlers-cleaners","Tech-support","Farming-fishing","Protective-serv","Priv-house-serv","Others "])
workclass = st.selectbox("Select Workclass",["Private","Self-emp-not-inc","Local-gov","State-gov","Self-emp-inc","Federal-gov","NotListed"])
native_country = st.selectbox("Select your Native Country",["Cambodia","Canada","China","Columbia","Cuba","Dominican-Republic","Ecuador","El-Salvador","England","France","Germany","Greece","Guatemala","Haiti","Holand-Netherlands","Honduras","Hong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United-States","Vietnam","Yugoslavia"])
hours_per_week = st.number_input("Select hours per week", 1, 80)
experience = st.number_input("Years of Experience", 0, 40)
# change 3:-...................
educational_num = int(education.split(":")[1]) 

gender_map = {'Male': 0, 'Female': 1}
gender_encoded = gender_map[gender]
gender_encoder = joblib.load('gender_encoder.pkl')


#education_map = {
#   'HS-grad': 0,
#    'Some-college': 1,    
#    'Bachelors': 2,       
#    'Masters': 3,          
#    'Assoc-voc': 4,        
#    'Assoc-acdm': 5,       
#    'Prof-school': 6,       
#    'Doctorate': 7         
#}
#education_encoded = education_map[education]

occupation_map = {
    'Adm-clerical': 0,
    'Craft-repair': 1,
    'Exec-managerial': 2,
    'Farming-fishing': 3,
    'Handlers-cleaners': 4,
    'Machine-op-inspct': 5,
    'Other-service': 6,
    'Priv-house-serv': 7,
    'Prof-specialty': 8,
    'Protective-serv': 9,
    'Sales': 10,
    'Tech-support': 11,
    'Transport-moving': 12
}
occupation_encoded = occupation_map[occupation]

workclass_map = {
    'Federal-gov': 0,
    'Local-gov': 1,
    'Private': 2,
    'Self-emp-inc': 3,
    'Self-emp-not-inc': 4,
    'State-gov': 5,
    'Without-pay': 6
}
workclass_encoded = workclass_map[workclass]


country_map = {
    "Cambodia" : 0,
    "Canada" : 1,
    "China": 2,
    "Columbia": 3,
    "Cuba": 4,
    "Dominican-Republic": 5,
    "Ecuador": 6,
    "El-Salvador": 7,
    "England":8 ,
    "France": 9,
    "Germany": 10,
    "Greece": 11,
    "Guatemala": 12,
    "Haiti":13 ,
    "Holand-Netherlands": 14,
    "Honduras": 15,
    "Hong": 16,
    "Hungary": 17,
    "India": 18,
    "Iran": 19,
    "Ireland": 20,
    "Italy": 21,
    "Jamaica": 22,
    "Japan": 23,
    "Laos": 24,
    "Mexico": 25,
    "Nicaragua": 26,
    "Outlying-US(Guam-USVI-etc)": 27,
    "Peru": 28,
    "Philippines": 29,
    "Poland": 30,
    "Portugal": 31,
    "Puerto-Rico": 32,
    "Scotland": 33,
    "South": 34,
    "Taiwan": 35,
    "Thailand": 36,
    "Trinadad&Tobago": 37,
    "United-States": 38,
    "Vietnam": 39,
    "Yugoslavia": 40,
    "Others": 41
}
native_country_encoded = country_map[native_country]

# Encode user inputs
gender_encoded = gender_encoder.transform([gender])[0] if gender in gender_encoder.classes_ else -1
occupation_encoded = occupation_encoder.transform([occupation])[0]
workclass_encoded = workclass_encoder.transform([workclass])[0]
native_country_encoded = country_encoder.transform([native_country])[0] if native_country in country_encoder.classes_ else -1


# Build Input dataframes
# change 4 :- ................ (in values)
input_df = pd.DataFrame({
    "age": [age],
    "gender": [gender_encoded],
    "educational-num": [educational_num],
    "occupation": [occupation_encoded],
    "workclass": [workclass_encoded],
    "native-country": [native_country_encoded],
    "hours-per-week": [hours_per_week],
    "experience":[experience]
})

st.write("Input Data")
st.write(input_df)

#Change 5:- ...................(whole syntax , read it )
numerical_cols =["age","gender","educational-num","occupation","workclass","native-country","hours-per-week","experience"]
joblib.dump(numerical_cols, 'numerical_cols.pkl')

scaler = joblib.load('scaler.pkl')
numerical_cols = joblib.load('numerical_cols.pkl')


X_scaled = scaler.transform(input_df[numerical_cols])


if st.button("Predict"):
    st.write("Prediction triggered!")
    #Change 6:-.....(replace)
    #features = np.array([[age, gender, educational_num, occupation, workclass, native_country, hours_per_week,experience]])
    #salary = model.predict(features)
    salary = model.predict(X_scaled)
    st.success(f"Predicted Salary: {salary[0]:.2f}")
