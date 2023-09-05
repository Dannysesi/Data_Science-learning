import streamlit as st
import pickle
# import shap
import matplotlib.pyplot as plt
import pandas as pd

# Load the Boston house dataset
boston = pd.read_csv('Data.csv')

# Load the machine learning model from the pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a function to preprocess the user input
def preprocess_input(inputs):
    # Convert the user input to a dataframe
    input_df = pd.DataFrame(inputs, index=[0])
    
    # Add any additional preprocessing steps here
    # ...
    
    return input_df

# Create a function to make a prediction
def predict_price(inputs):
    # Preprocess the inputs
    input_df = preprocess_input(inputs)
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_df)[0]
    
    return prediction

# medv to $1000
def medv_con(a):
    return a * 1000

#convert to local currency naira
def convert(a):
    return a * 459.5


# Create the Streamlit app
st.title('Leinad House Price Predictor')
st.write('---')


# sidebar
st.sidebar.header("Specify Input Parameters")

# Add input widgets for the user to enter the input features
with st.sidebar:
    crim = st.slider('CRIM', min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    zn = st.slider('ZN', min_value=0.0, max_value=100.0, value=0.0, step=1.0)
    indus = st.slider('INDUS', min_value=0.0, max_value=30.0, value=0.0, step=1.0)
    chas = st.selectbox('CHAS', [0, 1])
    nox = st.slider('NOX', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    rm = st.slider('RM', min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    age = st.slider('AGE', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    dis = st.slider('DIS', min_value=0.0, max_value=15.0, value=0.0, step=0.01)
    rad = st.slider('RAD', min_value=0.0, max_value=30.0, value=0.0, step=1.0)
    tax = st.slider('TAX', min_value=0.0, max_value=750.0, value=0.0, step=1.0)
    ptratio = st.slider('PTRATIO', min_value=0.0, max_value=25.0, value=0.0, step=1.0)
    b = st.slider('B', min_value=0.0, max_value=400.0, value=0.0, step=1.0)
    lstat = st.slider('LSTAT', min_value=0.0, max_value=40.0, value=0.0, step=0.5)
    
      
    
# When the user clicks the "Predict" button, make a prediction
if st.button('Predict'):
    # Get the user input as a dictionary
    input_data = {'CRIM': crim, 'ZN': zn, 'INDUS': indus, 'CHAS': chas, 'NOX': nox,
                 'RM': rm, 'AGE': age, 'DIS': dis, 'RAD': rad, 'TAX': tax,
                 'PTRATIO': ptratio, 'B': b, 'LSTAT': lstat}
    features = pd.DataFrame(input_data, index=[0])
    st.write(features)
    # Make a prediction
    prediction = predict_price(input_data)
    
    # Display the prediction to the user
    st.write('The predicted house price is $', round(medv_con(prediction), 2))
    st.write('House Price in local currency Naira is ₦', round(convert(medv_con(prediction)), 3))
