import pickle
import numpy as np
import pandas as pd
import streamlit as st 
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# load the model 
model = tf.keras.models.load_model(r'pickle_files\model.h5')

with open(r'pickle_files\ohe_encoder_geography.pkl','rb') as file_obj:
    ohe_encoder_geo = pickle.load(file_obj)

with open(r'pickle_files\label_encoder_gender.pkl','rb') as file_obj:
    label_encoder_gender = pickle.load(file_obj)

with open(r'pickle_files\scaler.pkl','rb') as file_obj:
    scaler = pickle.load(file_obj)

# streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography',ohe_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,99)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

geo_encoded = ohe_encoder_geo.transform([[geography]])
geo_df = pd.DataFrame(geo_encoded,columns=ohe_encoder_geo.get_feature_names_out(['Geography']))

data_df = pd.DataFrame(input_data)
final_df = pd.concat([data_df,geo_df],axis=1)

data_scaled = scaler.transform(final_df)

prediction = model.predict(data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob > 0.5:
    st.write('The customer is likely to Churn')
else:
    st.write("The customer is not likely to Churn")


