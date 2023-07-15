# Importing libaries to be used 
import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd

# reading hosuing dataset
df = pd.read_csv('Data.csv')

# importing already trained model
with open('model3.pkl', 'rb') as f:
    model = pickle.load(f)

def preprocess_input(inputs):
    # Convert the user input to a dataframe
    input_df = pd.DataFrame(inputs, index=[0])
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
st.title('Leinad Real Estate👷‍♂️👷‍♂️👷‍♂️')
st.header('House Price Prediction System💻')
st.write('---')

# st.markdown(
#     f"""
#     <style>
#     .stApp {
#         background-image: url('ddy.jpg');
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# sidebar
# st.sidebar.header("Specify Input Parameters")

# Add input widgets for the user to enter the input features
# with st.sidebar:
a, b, c, d = st.columns([4, 4, 4, 4])
crim = a.number_input('Crime rate', min_value=0.0, max_value=100.0,  step=1.0)
zn = b.number_input('Land size per plot', min_value=0.0, max_value=100.0,  step=1.0)
indus = c.number_input('Avg no. of Bathrooms', min_value=0.0, max_value=30.0,  step=1.0)
chas = d.selectbox('Charles River', [0, 1])
i, j = st.columns([4,4])
nox = i.slider('Air condition', min_value=0.0, max_value=1.0, value=0.0, step=0.01)
rm = j.slider('Avg no. of Bedrooms', min_value=0.0, max_value=10.0, value=0.0, step=0.5)
age = st.number_input('Home condition', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
u, v = st.columns([4,4])
dis = u.slider('Distance to employment centers', min_value=0.0, max_value=15.0, value=0.0, step=0.01)
rad = v.slider('Acces to federal roads', min_value=0.0, max_value=30.0, value=0.0, step=1.0)
m, n = st.columns([6,6])
tax = m.number_input('Property tax', min_value=0.0, max_value=750.0, value=0.0, step=1.0)
ptratio = n.slider('Pupil-teacher ratio by town', min_value=0.0, max_value=25.0, value=0.0, step=1.0)
q, r = st.columns([4,4])
b = q.number_input('B', min_value=0.0, max_value=400.0, value=0.0, step=1.0)
lstat = r.slider('% lower status of the population', min_value=0.0, max_value=40.0, value=0.0, step=0.5)

# When the user clicks the "Predict" button, make a prediction
st.write('---')
col1, col2, col3 = st.columns([3,1,3])
if col2.button('Predict'):
    # Get the user input as a dictionary
    input_data = {'CRIM': crim, 'ZN': zn, 'INDUS': indus, 'CHAS': chas, 'NOX': nox,
                 'RM': rm, 'AGE': age, 'DIS': dis, 'RAD': rad, 'TAX': tax,
                 'PTRATIO': ptratio, 'B': b, 'LSTAT': lstat}
    features = pd.DataFrame(input_data, index=[0])
    st.write(features)
    # Make a prediction
    prediction = predict_price(input_data)
    
    # Display the prediction to the user
    st.write('---')
    st.write('The predicted house price is $', round(medv_con(prediction), 2))
    st.write('House Price in local currency (Naira) is ₦', round(convert(medv_con(prediction)), 3))

