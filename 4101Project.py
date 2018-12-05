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

def rhymeTData(rhyme): #creates and stores the rhyme data for later NN usage

	creationTexts = rhyme.lower().split("\n") #corpus
	
	token.fit_on_texts(creationTexts)
	numRhyme = len(token.word_index) + 1

	impSeq = [] #Keras.array
	for line in creationTexts: #takes the corpus and converts it into a dataset of sentences/sequences
		listX = token.texts_to_sequences([line])[0] 
		for x in range(1, len(listX)): #takes the lengths of the sequences and pads them to be equal
			sequence2 = listX[:x+1]
			impSeq.append(sequence2) #takes the resulting sequence from the data and then appends what it does to store for later use in the model 

	sequencePrediction = max([len(x) for x in impSeq]) #starts the training of the epochs and a small base for the model network
	impSeq = npy.array(pad_sequences(impSeq, maxlen = sequencePrediction, padding = 'pre')) #helps the input of the sequence with padding (keras.io function)
	prediction, ticket = impSeq[:,:-1],impSeq[:,-1]
	ticket = kUtil.to_categorical(ticket, num_classes = numRhyme) #
	return prediction, ticket, sequencePrediction, numRhyme #returns all the predictions, labels, and predictions with the sequence numbers

def nueralNetworkModel(prediction, ticket, sequencePrediction, numRhyme): #Creates the model with some basic Keras.io functions
	
	model = Sequential()
	model.add(Embedding(numRhyme, 10, input_length = sequencePrediction-1))
	model.add(LSTM(250, return_sequences = True)) #runs module and returns the seqeunces to be ran through the module
	model.add(LSTM(200)) #takes the prediction and coordinates with the epochs for how many training modules it has to run
	model.add(Dense(numRhyme, activation = 'softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
	network = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 0, mode = 'auto') #Calling back the functions using Keras.callbacks
	model.fit(prediction, ticket, epochs = 200, verbose = 1, callbacks = [network]) #Uses the call backs and trains for the epochs (200 iterations)
	print(model.summary())
	return model 

def rhymePrediction(rhymePlacement, upcoming, sequencePrediction): #Takes the corpus, tokenizer, sequences, and some Keras.io functions to create the sequence
	for _ in range(upcoming):
		listX = token.texts_to_sequences([rhymePlacement])[0]
		listX = pad_sequences([listX], maxlen = sequencePrediction - 1, padding = 'pre') #prepads the sequences in the model and ready for output
		predicted = model.predict_classes(listX, verbose = 0)
		endingRhyme = ""
		for word, index in token.word_index.items(): #builds the sequence of random generation/prediction and places it in the sequence output
			if index == predicted:
				endingRhyme = word
				break
		rhymePlacement += " " + endingRhyme #if last word, then stop placing words and finish the model
	return rhymePlacement


prediction, ticket, sequencePrediction, numRhyme = rhymeTData(rhyme) #takes all parameters from network and feeds them into the prediciton text
model = nueralNetworkModel(prediction, ticket, sequencePrediction, numRhyme) #output of the model which will show sequences in the language and # possible outcomes 

print("")
print("Starting 4101 Project by JJ Juarez and Anita Imoh")
print("")
print("Testing 'Humpty' prediction with the next 1 sequence")
print("")
print(rhymePrediction("Humpty", 1, sequencePrediction))
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
