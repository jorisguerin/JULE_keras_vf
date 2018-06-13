from utils.utils_data import *
from utils.affinity import *
from utils.agglomerative_clustering import *

from termcolor import colored

import numpy as np
import time

class JULE:

	def __init__(self, datafile, modelfile, params):

		self.features_orig, self.true_labels = load_data(datafile)
		print(colored("\nLoaded dataset of shape", "blue"), self.features_orig.shape)

		self.n_samples = self.features_orig.shape[0]

		self.params = params

		self.initialize_clusters()

	def initialize_clusters(self):
		start = time.time()
		_, indices = get_NearestNeighbors(self.features_orig, self.params['k_neighbors_pts'])
		self.labels_current_table = initialize_clusters(indices, self.n_samples)
		print(colored("\nInitialized clusters", "blue"))
		print(colored("    N clusters: %d" % len(self.labels_current_table), "green"))
		print(colored("    Time: %f" % (time.time() - start), "magenta"))