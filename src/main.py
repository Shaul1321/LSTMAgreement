from DataParser import *
from P1Generator import *
from P2Generator import *
from RNN import *
import StatsCollector
import EmbeddingVisualizer

import sys

"""
train a LSTM for a given task. The task number is accepted
as commandline argument (currently, part1 or part2).
A task-specific training data generator is created accordingly and injected to the RNN. 
"""

def main():
        part = 1 if "part1" in sys.argv else 2
        plot = "plot" in sys.argv
	print "Running task {}. Plotting: {}".format(part, plot)

	data_dicts, voc, labels, verbs, word2key, key2word = DataParser.parse()

	if part == 1:
		data_generator = NumberPredictionGenerator(data_dicts, voc, word2key, key2word)
	else:
		data_generator = GrammaticalJudgment_Generator(data_dicts, voc, word2key, key2word)

	statsCollector = StatsCollector.StatsCollector()
	rnn = RNN(len(voc), 128, 2, data_generator, statsCollector)
	rnn.train()

	if plot:

		print "calculating TSNE Projection..."
		EmbeddingVisualizer.visaulize(rnn.E, word2key, data_dicts)
		statsCollector.plot_accuracy_vs_distance()


if __name__ == '__main__':
	main()
	  

		
