import dynet as dy
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random


def visaulize(embedding, word2key, data_dicts):
	
		"""
		plot the TSNE projection of the embedded vectors of the verbs in the training set.

		embedding: the trained embedding

		word2key: word, key dictionary

		verbs: a set of tuples (present_tense_verb, the verb_pos)

		"""

		words = []
		embeddings = []
		labels = []
		sample_size = 500

		# collect the embedding of different verbs
	
		words_set = set()
		for data in data_dicts:
			verb_index = int(data['verb_index'])-1
			verb_pos = data['verb_pos']
			verb = data['sentence'].split(" ")[verb_index]

			if verb not in words_set:
				words_set.add(verb)
				words.append(verb)
				verb_encoded = word2key[verb]
				embedded = dy.lookup(embedding, verb_encoded).npvalue()
				embeddings.append(embedded)
				labels.append(verb_pos)

			if len(words) > sample_size: break

		embeddings = np.array(embeddings)


		# calculate TSNE projection & plot

		proj = TSNE(n_components=2).fit_transform(embeddings)

		fig, ax = plt.subplots()
		colors = ["red" if l == "VBP" else "blue" for l in labels]
		xs, ys = proj[:,0], proj[:,1]
		ax.scatter(xs, ys, c=colors, alpha=0.5, label = labels)
		#ax.legend(['VBP', 'VBZ'])
		
		for i, w in enumerate(words):
                    if i%6 == 0:
    			ax.annotate(w, (xs[i],ys[i]))

		plt.show()


