import csv
import numpy as np
import time
import random


"""
responsible for parsing the vocab and data examples files, creating the dataset for training.
"""

class DataParser(object):

   @classmethod
   def parse(self):
     """
	creates a list of dictionaries containing informatiom about input sentences
        (each sentence is represented by a single dictionary).

 	returns:
        	data_dicts, voc, labels, word2key, key2word

    	data_dicts: a list of dictionaries, each containing information about a single sentence in the 			    dataset, such as the sentence itself, position of the verb, pos of the verb, etc. 			    created on the basis of agr_50_mostcommon_10K.tsv.

    	voc: the vocabulary of the training set, cut to 10000 most common words, plus pos labels.
	     created on the basis of wiki_vocab.txt.

	labels: the set of labels (pos) encountered.

	verbs:	the set of all tuples (present_tense_verb, the verb_pos)

	word2key: a word, integer mapping

	key2word: an integer, word mapping
     """

     print "Creating training data..."

     data_dicts = []
     voc = set()
     labels = set()

     #create vocabulary & labels

     with open("../data/wiki_vocab.txt") as f:
        lines = f.readlines()

     for line in lines[1:]:
          tag = line.split("\t")[1].split(" ")[0]
          labels.add(tag)
    

     #create training examples

     with open("../data/agr_50_mostcommon_10K.tsv") as tsvfile:
      tsvreader = csv.reader(tsvfile, delimiter="\t")

      for i, line in enumerate(tsvreader):

        if i == 0:
           column_names = line
        else:
           data_dict = {}

           for j, val in enumerate(line):
              data_dict[column_names[j]] = val

           for word in data_dict['sentence'].split(" "):
             voc.add(word)

           data_dicts.append(data_dict)

     verbs = set( [(x['sentence'].split(" ")[int(x['verb_index'])-1], x['verb_pos']) for x in data_dicts])

     word2key = dict((char, index) for index, char in enumerate(sorted(voc)))
     key2word = dict((index, char) for index, char in enumerate(sorted(voc)))

     return data_dicts, voc, labels, verbs, word2key, key2word
  


