import random

"""
an abstract class for creating training examples, based on the training corpus.
derived classes override create_example abstract method according to the requirements. 
"""

class DataGeneratorBase(object):

 def __init__(self, data_dicts, voc, word2key, key2word):

   self.data_dicts = data_dicts
   self.voc = voc
   self.word2key = word2key
   self.key2word = key2word

   n = len(self.data_dicts)

   random.shuffle(self.data_dicts)
   self.train = self.data_dicts[:int(0.09 * n)]
   self.dev  = self.data_dicts[int(0.09*n):int(0.1*n)]
   self.test = self.data_dicts[int(0.1*n):]

 def get_train_size(self):

   return len(self.train)

 def get_dev_size(self):

   return len(self.dev)

 def get_test_size(self):

   return len(self.test)

 def create_example(self, data_sample):
	"""
	create a single training/prediction example.

	data_sample - a raw data sample from the corpus (contains informatiom regarding a signle 	sentence).


	returns: x_encoded, y_encoded

	x_encoded - the encoded training example
	y_encoded - the true label for that training example
	"""

 	raise NotImplementedError


 def generate(self, mode):

   """
 	a template method for generating a training example. the abstract method create_example
 	is implemented in the derived class, according to the requirements of each task.

	mode - an enum specifying TRAIN/DEV/TEST.
   """


   while True: 

      if mode.name == "TRAIN":
      	data = random.choice(self.train)

      elif mode.name == "DEV":
	data = random.choice(self.dev)

      else:
	data = random.choice(self.test)

      x_encoded, y_encoded = self.create_example(data)
      yield x_encoded,  y_encoded, data

