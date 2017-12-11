from collections import Counter
import numpy as np

"""an observer that records & disaplay different statistics regarding the network's performance"""

class StatsCollector(object):

	def __init__(self):
		self.n = 0
		self.good = Counter()
                self.bad = Counter()
		self.good_total = 0.
		self.bad_total  = 0.

		self.losses = []

        def clear(self):
		self.good = Counter()
		self.bad  = Counter()
		self.good_total = 0.
		self.bad_total  = 0.


	def plot_accuracy_vs_distance(self):

		import matplotlib.pyplot as plt

		xs = [x for x in range(16) if (self.good[x]+self.bad[x])>1e-6]
		ys = [self.good[x]/(self.good[x]+self.bad[x]) for x in xs]

		plt.plot(xs,ys)
		plt.xlabel('subject-verb distance')
                plt.ylabel('accuracy')
		plt.ylim(0,1)
		plt.show()
		
	def report_total_accuracy(self):
		acc =  self.good_total/(self.good_total+self.bad_total)
		loss = np.average(self.losses)
		print "accuracy: {}; loss: {}".format(acc, loss)


	def report_distance_accuracy(self):

		  for i in range(len(self.good)):

                        if self.good[i] + self.bad[i] <1e-5: continue

			acc = self.good[i]/(self.good[i]+self.bad[i]) 
			samples = self.good[i] + self.bad[i]
			print "accuracy at distance",i,"is",acc,"with ",samples,"sampels"

        def notify(self, y_pred, y_true, loss, sent_data):
		"""
		counts error rate as a function of distance between the subject and the verb.
		recrods loss.

		y_pred	the prediction of the network
		
		y_true	true label

		loss	the loss in that iteration

		sent_data the dictionary represented the input sentence

		"""

		self.n+=1
		self.losses.append(loss)

		dis = min(int(sent_data['distance']), 15)
		#dis = min(int(sent_data['n_intervening']), 15)
		has_intervening = int(sent_data['n_intervening']) > 0

		if y_pred == y_true:

			if not has_intervening:
				self.good[dis]+=1.
			self.good_total+=1.
		else:
			if not has_intervening:
				self.bad[dis]+=1.

			self.bad_total+=1.







		
	
