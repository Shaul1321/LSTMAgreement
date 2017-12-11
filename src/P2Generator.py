from DataGenerator import *
from pattern.en import conjugate
import random

class GrammaticalJudgment_Generator(DataGeneratorBase):

 def __init__(self, data_dicts, voc,  word2key, key2word):
     DataGeneratorBase.__init__(self, data_dicts, voc,  word2key, key2word)
 
 def create_example(self, data_sample):

        """
        creates a training example - the encoded sentence, with a probability of 50%
	of flipping the present verb number to an ungrammatical form. used to train a LSTM for a 	 grammatical judgment task: given a sentence, without special indication of 
	the verb location, predict whether or not the verb agreement is grammatical.


 	returns:
        	x_encoded, y_encoded

    	x_encoded: a list of all words in the sentence, encoded as numbers
    	y_encoded: 1 if the sentence is grammatical, 0 otherwise
        """


	verb_index = int(data_sample['verb_index'])-1
	x_example = data_sample['sentence'].split(" ")
	verb = x_example[verb_index]
	
        #randomely flip to incorrect form.

        flipped = random.random() < 0.5 

	if flipped:

		flipped_verb_number = "2sg" if data_sample['verb_pos'] == 'VBZ' else "3sg"
                ungrammatical_form = conjugate(verb, flipped_verb_number)

		if ungrammatical_form not in self.word2key: #TODO: some forms are not in the vocab - need to check
		    flipped = False

		else:
		    x_example[verb_index] = ungrammatical_form
	
	#encode x,y. y is 1 iff the sentence is grammatical

	x_encoded = [self.word2key[word] for word in x_example]
	y_encoded = 1 if not flipped else 0 

	return x_encoded, y_encoded


