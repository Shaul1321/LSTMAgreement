import dynet as dy
import numpy as np
import random
import time
from enum import Enum

import dynet_config
dynet_config.set_gpu()

NUM_LAYERS=1
EMBEDDING_SIZE = 50

from enum import Enum

class Mode(Enum):
     TRAIN = 1
     DEV = 2
     TEST = 3

class RNN(object):

	def __init__(self, in_size, hid_size, out_size, dataGenerator, observer):

		self.in_size = in_size
		self.hid_size = hid_size
		self.out_size = out_size
		self.generator = dataGenerator
		self.observer = observer
		self.create_model()
		

	def create_model(self):

                """build the model, that consists of an embedding layer, a LSTM layer,
		and a softmax layer

		E - an embedding matrix of size (in_size, embedding_size)
		W_ho - a hidden-output matrix, of size (out_size, hid_size)
		builder - a builder for LSTM layer with hid_size units.
		"""

        	self.model =  dy.Model()
		self.E = self.model.add_lookup_parameters((self.in_size, EMBEDDING_SIZE))
		self.W_ho = self.model.add_parameters((self.out_size, self.hid_size))
		self.builder = dy.LSTMBuilder(NUM_LAYERS, EMBEDDING_SIZE, self.hid_size, self.model)
                self.trainer = dy.AdamTrainer(self.model)
        

        def predict(self, sentence, training=True, dropout_rate = 0.4):	

		
		W_ho = dy.parameter(self.W_ho)
		s = self.builder.initial_state()

		#embedd each word in the sentence, and feed it to the LSTM one word at a time.

       	 	for w in sentence:
			w_embedded = dy.lookup(self.E, w)
			if training: w_embedded = dy.dropout(w_embedded, dropout_rate)
			s=s.add_input(w_embedded)

		h = s.output()
		y_pred = dy.softmax(W_ho * h)
		return y_pred


        def train(self, epochs=5):

		n = self.generator.get_train_size()
		print "size of training set: ", n
                print "training..."

		iteration = 0

		for i, training_example in enumerate(self.generator.generate(Mode.TRAIN)):

			iteration+=1


			#stopping criteria
	
                        if i > epochs*n: 
				print "Calcualting accuracy on test set. This may take some time"
				self.test(Mode.TEST)
				return

			# report progress. 

			if i%n == 0:

				iteration = 0
				print "EPOCH {} / {}".format(i/n, epochs)
				print "Calculating accuracy on dev set."
				self.test(Mode.DEV)

			if iteration%(n/5) == 0:
				print "iteration {} / {}".format(iteration, n)

			# build computation graph & predict

			dy.renew_cg()
			sent, true_label, sent_data = training_example

			y_pred = self.predict(sent)

			#backprop
	
			loss = -dy.log(y_pred[true_label])
               		loss.backward()
               		self.trainer.update()


        def test(self, mode):

		if mode == Mode.DEV:
			n = self.generator.get_dev_size()
		else:
			n = self.generator.get_test_size()

 		self.observer.clear()

		for i, training_example in enumerate(self.generator.generate(mode)):

			sent, true_label, sent_data = training_example

			dy.renew_cg()
			y_pred = self.predict(sent, False)
			loss = -dy.log(y_pred[true_label])

			prediction = np.argmax(y_pred.npvalue())
			self.observer.notify(prediction, true_label, loss.scalar_value(), sent_data)


			if i > n: break

	
		self.observer.report_total_accuracy()




