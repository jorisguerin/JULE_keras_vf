import h5py

def load_data(file_path):
	f = h5py.File(file_path)
	features = f["data"][:]
	labels = f["labels"][:]
	f.close()

	return features, labels