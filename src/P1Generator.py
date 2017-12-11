from DataGenerator import *

class NumberPredictionGenerator(DataGeneratorBase):

 def __init__(self, data_dicts, voc, word2key, key2word):
     DataGeneratorBase.__init__(self, data_dicts, voc,  word2key, key2word)

 
 def create_example(self, data_sample):

        """
        creates a training example for part 1 - the encoded sentence, up to, but not including,
	the present tense verb. used to train a LSTM for a number prediction task: the network is 	  presneted with the sentence up to the present-tense verb, and has to predict its number.

 	returns:
        	x_encoded, y_encoded

    	x_encoded: a list of the words in the sentence, encoded as numbers,
	until the present tense verb.
	
    	y_encoded: 1 if verb pos is VBP, 0 if it's VBZ
        """

	verb_index = int(data_sample['verb_index'])-1
	verb_pos = 1 if data_sample['verb_pos']=='VBP' else 0
	x_example = data_sample['sentence'].split(" ")[:verb_index]
	x_encoded = [self.word2key[word] for word in x_example]
	y_encoded = verb_pos

	return x_encoded, y_encoded


