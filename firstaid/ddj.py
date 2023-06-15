import streamlit as st
import tflearn
import tensorflow as tf


# Load the saved model
model = tf.keras.models.load_model("model2.tflearn")

# Create a Streamlit chat interface
st.title("Chat System")

# Function to generate a response
def generate_response(user_input):
    # Preprocess the user input (if required)
    # ...
    
    # Use the loaded model to generate a response
    # ...
    response = model.predict(...)
    
    # Postprocess the response (if required)
    # ...
    
    return response

# Main interaction loop
while True:
    user_input = st.text_input("User Input", "")
    
    if user_input:
        response = generate_response(user_input)
        st.text_area("Chatbot Response", response)



