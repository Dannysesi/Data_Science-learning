import streamlit as st
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import time


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

	with open('data.pickle','wb') as f:
		pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

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

from requests import Response


unique_key = 0
def chat():
	unique_key = 0
	st.title("First Aid Chatbot Assistant🤖")
	st.write('---')

	# Initialize chat history
	if "messages" not in st.session_state:
		st.session_state.messages = []

	tab1, tab2 = st.tabs(['ChatBot', 'About'])
	with tab1:
	# Accepting user input
		with st.form("chat_input", clear_on_submit=True):
			a, b = st.columns([4, 1])
			inp = a.text_input("User_Input: ", key=unique_key, placeholder='Enter Your First Aid Prompt💭', label_visibility="collapsed")
			# if inp.lower() == "quit":
			if inp.strip():
				pass
			results = model.predict([bag_of_words(inp,words)])[0]
			results_index = numpy.argmax(results)
			tag = labels[results_index]
			dail = '+2349025629246'
			submitted = b.form_submit_button("Send", use_container_width=True)
			st.session_state.messages.append({"role": "user", "content": inp})

			
			if submitted:
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

		st.subheader('Chat History')
		st.write('---')
		# Display chat messages from history on app rerun
		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

	with tab2:
		st.subheader('About First Aid Chatbot')
		st.write('---')

		st.write("<span style='font-size: 20px;'>Welcome to First Aid Chatbot Assistant, your trusted companion for handling emergency situations and providing immediate medical guidance. Our app is designed to empower users with essential first aid knowledge, ensuring quick and accurate responses during critical moments.<span>", unsafe_allow_html=True)
		st.write('---')
		st.write("<span style='font-size: 30px;'>Key Features:<span>", unsafe_allow_html=True)
		st.write("**Interactive Chatbot:** Our intelligent and user-friendly chatbot is at your service 24/7. Simply describe the situation or symptoms, and our AI-powered assistant will guide you through step-by-step first aid procedures.")
		st.write("**Emergency Situations:** Whether it's a minor injury or a life-threatening situation, First Aid Chatbot Assistant is equipped to handle a wide range of emergencies. From burns and cuts to CPR and choking, we've got you covered.")
		st.write("**Illustrative Instructions:** Our app offers easy-to-follow, visual instructions to help you perform first aid procedures confidently and efficiently.")
		st.write("**Stay Informed:** Stay updated with the latest first aid guidelines and protocols, ensuring your knowledge remains up-to-date and reliable.")
		st.write('---')
		st.write("<span style='font-size: 20px;'>Why Choose Our First Aid Chatbot Assistant?<span>", unsafe_allow_html=True)
		st.write("**Instant Access:** With our app, first aid information is just a few taps away, ready to assist you in any medical emergency.")
		st.write("**User-Friendly Interface:** We've designed our app to be intuitive and accessible to users of all ages and backgrounds, making first aid guidance available to everyone.")
		st.write("**Peace of Mind:** Whether you're at home, on the road, or out in nature, having First Aid Chatbot Assistant by your side will give you the confidence to handle unforeseen emergencies effectively.")
		st.write("**Always Evolving:** We continually update our database to incorporate the latest medical guidelines and best practices, ensuring the most reliable and accurate information.")





# if __name__ == '__main__':
# 	chat()

chat()	

