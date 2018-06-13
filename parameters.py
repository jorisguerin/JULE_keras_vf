### Infos about the dataset to load ###
datapath = "/data/users_data/joris/DATA/MVTC/JULE_DS"
dataset = "coil-100"
features = "CNN/resnet"

datafile = "%s/%s/%s.h5" % (datapath, dataset, features)

### Infos about the model to load ###
modelspath = "/data/users_data/joris/MODELS/MVTC/JULE_MODELS"
modelfile = "%s/%s/%s/model.py" % (modelspath, dataset, features)

### Unrolling parameters ###
jule_params = {'Nepchs_unrolledPeriod': 20, # Number of training epochs between mergings
'k_neighbors_pts': 20} # Number of neighbors to consider for computing affinity between samples