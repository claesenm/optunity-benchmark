import optunity
import numpy as np
import hpolib_generator as hpo
import cloudpickle as pickle
import sklearn.datasets

name='ionosphere'
num_folds=5
budget = 150

search={'logC': [-8, 1], 'logGamma': [-8, 1]}

positive_digit = 1.0
dataset = sklearn.datasets.load_svmlight_file('data/ionosphere_scale')
data = dataset[0].todense()
data = data + 1.0 * np.random.randn(data.shape[0], data.shape[1])
labels = dataset[1]

n = data.shape[0]

positive_idx = [i for i in range(n) if labels[i] == positive_digit]
negative_idx = [i for i in range(n) if not labels[i] == positive_digit]

original_data = data[positive_idx + negative_idx, ...]
data = original_data # + 10 * np.random.randn(original_data.shape[0], original_data.shape[1])
labels = [True] * len(positive_idx) + [False] * len(negative_idx)

objfun = hpo.make_svm_objfun(data, labels, num_folds)

hpo_search = hpo.svm_search_space(**search)
hpo.setup_hpolib(hpo.negate(objfun), hpo_search, budget, name)
