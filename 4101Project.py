#CSC 4101 (Programming Languages) course assignment studying Neural Networks using Python 3.6.1, Tensorflow, and Keras
#Created Fall 2018 by JJ Juarez and Anita Imoh
#© JJ Juarez and Anita Imoh. Do not redistribute without owner permission. 

import tensorflow as tf
import numpy as npy
import keras.utils as kUtil
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.models import Sequential

token = Tokenizer()

rhyme = """Humpty Dumpty sat on a wall,\n
Humpty Dumpty had a great fall.\n
All the king's horses and all the king's men,\n
Couldn't put Humpty together again.\n"""

def rhymeTData(rhyme): 

	creationTexts = rhyme.lower().split("\n")
	
	token.fit_on_texts(creationTexts)
	numRhyme = len(token.word_index) + 1

	impSeq = []
	for line in creationTexts:
		listX = token.texts_to_sequences([line])[0]
		for x in range(1, len(listX)):
			sequence2 = listX[:x+1]
			impSeq.append(sequence2)

	sequencePrediction = max([len(x) for x in impSeq])
	impSeq = npy.array(pad_sequences(impSeq, maxlen = sequencePrediction, padding = 'pre'))
	prediction, ticket = impSeq[:,:-1],impSeq[:,-1]
	ticket = kUtil.to_categorical(ticket, num_classes = numRhyme)
	return prediction, ticket, sequencePrediction, numRhyme

def nueralNetworkModel(prediction, ticket, sequencePrediction, numRhyme):
	
	model = Sequential()
	model.add(Embedding(numRhyme, 10, input_length = sequencePrediction-1))
	model.add(LSTM(250, return_sequences = True))
	model.add(LSTM(200))
	model.add(Dense(numRhyme, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	network = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 0, mode = 'auto')
	model.fit(prediction, ticket, epochs = 200, verbose = 1, callbacks = [network])
	print(model.summary())
	return model 

def rhymePrediction(rhymePlacement, upcoming, sequencePrediction):
	for _ in range(upcoming):
		listX = token.texts_to_sequences([rhymePlacement])[0]
		listX = pad_sequences([listX], maxlen = sequencePrediction - 1, padding = 'pre')
		predicted = model.predict_classes(listX, verbose = 0)
		endingRhyme = ""
		for word, index in token.word_index.items():
			if index == predicted:
				endingRhyme = word
				break
		rhymePlacement += " " + endingRhyme
	return rhymePlacement


prediction, ticket, sequencePrediction, numRhyme = rhymeTData(rhyme)
model = nueralNetworkModel(prediction, ticket, sequencePrediction, numRhyme)

print("")
print("Starting 4101 Project by JJ Juarez and Anita Imoh")
print("")
print("Testing 'Humpty Dumpty' prediction with next 3 sequences")
print("")
print(rhymePrediction("Humpty Dumpty", 3, sequencePrediction))
print("")
print("Testing 'king's horses' prediction with next 4 sequences")
print("")
print(rhymePrediction("king's horses", 4, sequencePrediction))
print("")
print("Testing 'put Humpty' prediction with the next 6 sequences")
print("")
print(rhymePrediction("put Humpty", 6, sequencePrediction))


