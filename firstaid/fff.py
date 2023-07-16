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

with open("intents.json") as file: # type: ignore
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

	# Accepting user input
	inp = st.chat_input(key=unique_key, placeholder='Enter Your First Aid Prompt💭',)
	with st.chat_message("user"):
		# a, b = st.columns([4, 1])
		
		# unique_key <= 1
		# if inp.lower() == "quit":
		results = model.predict([bag_of_words(inp,words)])[0]
		results_index = numpy.argmax(results)
		tag = labels[results_index]
		dail = '+2349025629246'
		# b.form_submit_button("Send", use_container_width=True)

	# Processing bots response
	if results[results_index] > 0.5:
		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = tg['responses']
		with st.chat_message("assistant"):
			message_placeholder = st.empty()
			full_resp = ""
			resp = random.choice(responses)
			for chunk in resp.split():
				full_resp += chunk + " "
				time.sleep(0.05)
        		# Add a blinking cursor to simulate typing
				message_placeholder.markdown(full_resp + "▌")
			print("\n")
		
	else:
		st.markdown("Sorry I didn't get that")		
	st.write(f'For more information please contact {dail}')

	

if __name__ == '__main__':
	chat()

	

