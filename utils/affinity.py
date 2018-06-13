import numpy as np
from sklearn.neighbors import NearestNeighbors

def get_NearestNeighbors(X, k_neighbors):
	# Returns the distances and indices of the K nearest neighbors exluding the data itself
	neighbors = NearestNeighbors(n_neighbors = k_neighbors + 1, algorithm = 'ball_tree', 
								n_jobs = -1).fit(X)
	distances, indices = neighbors.kneighbors(X)
	distances = distances[:, 1:]
	indices = indices[:, 1:]

	return distances, indices