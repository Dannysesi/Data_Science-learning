import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
from tensorflow import keras
import streamlit as st

with open("intents.json") as file:
	data = json.load(file)

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

tf.compat.v1.reset_default_graph()

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

def chat():
	unique_key = 0
	st.title("AutoMobile Fault Diagnotics System")
	q,r,v = st.columns([1.5,4,1])
	r.subheader("Your Expert Car Companion")
	st.write('---')
	
	st.write("<span style='font-size: 20px;'>Welcome to the Automobile Fault Diagnostics System!<span>", unsafe_allow_html=True)
	u, z = st.columns([2,2])
	u.write("<span style='font-size: 20px;'>Advanced Diagnostics: Say goodbye to the guesswork. Our software utilizes state-of-the-art algorithms and diagnostic tools to pinpoint potential issues accurately. From engine malfunctions to electrical glitches, we've got you covered.<span>", unsafe_allow_html=True)
	z.write("<span style='font-size: 20px;'>Real-time Insights: Get instant access to crucial data and real-time insights about your car's performance. Monitor vital parameters, receive alerts, and make informed decisions to prevent costly breakdowns.<span>", unsafe_allow_html=True)
	st.write('---')

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []
	
	if 'form_data' not in st.session_state:
		st.session_state.form_data = {'name': '', 'email': '', 'message': ''}


	# Accepting user input
	with st.form("chat_input", clear_on_submit=True):
		a, b = st.columns([4, 1])
		inp = a.text_input("User_Input: ", key=unique_key, placeholder='Enter Your Car Fault💭', label_visibility="collapsed")
		match_found = False
		if inp.strip():
			pass

        

		results = model.predict([bag_of_words(inp,words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]
		dail = '+2349025629246'
		b.form_submit_button("Send", use_container_width=True)
		st.session_state.messages.append({"role": "user", "content": inp})

		for intent in data["intents"]:
			for pattern in intent["patterns"]:
				if inp in pattern:
					st.markdown(intent["tag"])
					match_found = True
					break

		if not match_found:
			st.write('Please enter a vaild automobile issue')

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
	st.write(f'For more information please contact {dail}')
	st.write('---')

	# Display chat messages from history on app rerun
	for message in st.session_state.messages:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])


if __name__ == '__main__':
    chat()