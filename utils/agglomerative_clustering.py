import numpy as np

def initialize_clusters(indices, n_samples):
	cluster_status = -np.ones(n_samples, dtype = np.int32)
	count = 0
	for i_samples in range(n_samples):
		cur_idx = i_samples
		cur_cluster_idxs = []
		while cluster_status[cur_idx] == -1:
			cur_cluster_idxs.append(cur_idx)
			neighbor = indices[cur_idx, 0]
			cluster_status[cur_idx] = -2
			cur_idx = neighbor
			if len(cur_cluster_idxs) > 50:
				break
		if cluster_status[cur_idx] < 0:
			cluster_status[cur_idx] = count
			count += 1
		for j in range(len(cur_cluster_idxs)):
			cluster_status[cur_cluster_idxs[j]] = cluster_status[cur_idx]

	label_indices = []
	for _ in range(count):
		label_indices.append([])
	for i_samples in range(n_samples):
		label_indices[cluster_status[i_samples]].append(i_samples)

	return label_indices