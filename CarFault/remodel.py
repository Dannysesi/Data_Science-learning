import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import pandas as pd
import os
import json
import pickle
from tensorflow import keras
import streamlit as st

st.set_page_config(
     page_title='AutoMobile Fault Diagnostics System',
     layout="wide",
     initial_sidebar_state="expanded",
)

with open("intents.json") as file:
	data = json.load(file)




def chat():
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
	    for pattern in intent["patterns"]:
		    wrds = nltk.word_tokenize(pattern)
		    words.extend(wrds)
		    docs_x.append(wrds)
		    docs_y.append(intent["tag"])

	    if intent["tag"] not in labels:
		    labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)


    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
	    bag = []

	    wrds = [stemmer.stem(w) for w in doc]

	    for w in words:
		    if w in wrds:
			    bag.append(1)
		    else:
			    bag.append(0)
	    output_row = out_empty[:]
	    output_row[labels.index(docs_y[x])] = 1

	    training.append(bag)
	    output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
	    model.load("model.tflearn")
    except:
	    model = tflearn.DNN(net)
	    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	    model.save("model.tflearn")

    def bag_of_words(s,words):
	    bag = [0 for _ in range(len(words))]


	    s_words = nltk.word_tokenize(s)
	    s_words = [stemmer.stem(word.lower()) for word in s_words]

	    for se in s_words:
		    for i, w in enumerate(words):
			    if w == se:
				    bag[i] = 1

	    return numpy.array(bag)

    def save_feedback_to_csv(feedback_text):
        feedback_data = {'Feedback': [feedback_text]}
        feedback_df = pd.DataFrame(feedback_data)

        if not os.path.exists('feedback.csv'):
            feedback_df.to_csv('feedback.csv', index=False)
        else:
            feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)

    u,z,k = st.columns([0.8,2,0.5])
    z.title("AutoMobile Fault Diagnostic System")
    q,r,v = st.columns([1.5,2,0.5])
    r.subheader("Your Expert Car Companion")
    st.write('---')
    x, y, n = st.columns([1.3,2,0.5])
    y.write("<span style='font-size: 20px;'>Welcome to the Automobile Fault Diagnostic System!<span>", unsafe_allow_html=True)
    col5, col6, col7 = st.columns([4.6,4,2.5])
    col6.write("<span style='font-size: 20px;'>Say goodbye to the guesswork<span>", unsafe_allow_html=True)
    # st.write("<span style='font-size: 20px;'>Our software utilizes algorithm and diagnostic prompt to pinpoint potential issues accurately. From engine malfunctions to electrical glitches, we've got you covered.<span>", unsafe_allow_html=True)
    st.write('---')

	# Initialize chat history
    if "messages" not in st.session_state:
    	st.session_state.messages = []
	
    # if 'form_data' not in st.session_state:
    # 	st.session_state.form_data = {'name': '', 'email': '', 'message': ''}



    col1, col2, col3, col4 = st.columns([1.5,4,1,1.5])


    inp = col2.text_input("User_Input: ", value="", placeholder='Enter Your Car Fault💭', label_visibility="collapsed")

    w, b, a = st.columns([2.2,4,1.5])

    # Check if the form was submitted
    if col3.button("Send", use_container_width=True):
        # Process the input only when the form is submitted
        if inp.strip():
            results = model.predict([bag_of_words(inp, words)])[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
            st.session_state.messages.append({"role": "user", "content": inp})

        for intent in data["intents"]:
        	for pattern in intent["patterns"]:
        		if inp in pattern:
        			fg = intent["tag"]
        			b.write(f"<span style='font-size: 20px;'>THIS IS MOST LIKELY THE FAULT WITH YOUR CAR {fg}<span>", unsafe_allow_html=True)
        			match_found = True
        			break   


        if results[results_index] > 0.5:
        	for tg in data["intents"]:
        		if tg['tag'] == tag:
        			responses = tg['responses']
    
        	with st.chat_message("assistant"):
        		resp = random.choice(responses)
        		st.markdown(resp + "▌")
        		st.markdown("\n")
        		st.session_state.messages.append({"role": "assistant", "content": resp})
        else:
        	z = "Sorry I didn't get that please make a vaild prompt"
        	st.write("Sorry I didn't get that please make a vaild prompt")
        	st.session_state.messages.append({"role": "assistant", "content": z})
    st.write('---')	
    dail = '+2349011820966'
    st.write(f'For more information please contact {dail}')
    st.write('---') 
    st.write('Leave us a feedback')
    feedback = st.text_area("Feedback", placeholder="Type your feedback here...", label_visibility='collapsed')
    if st.button("Submit Feedback"):
    	save_feedback_to_csv(feedback)
    	st.success("Thank you for your feedback!")

    st.write('---')

	# Display chat messages from history on app rerun
	# for message in st.session_state.messages:
	# 	with st.chat_message(message["role"]):
	# 		st.markdown(message["content"])


if __name__ == '__main__':
    chat()