import optunity
import numpy as np
import hpolib_generator as hpo
import cloudpickle as pickle
from sklearn.datasets import load_digits


positive_digit = 8
name='digits-%d' % positive_digit
num_folds=5
budget = 150

search={'logC': [-8, 1], 'logGamma': [-8, 1]}

digits = load_digits()
n = digits.data.shape[0]

positive_idx = [i for i in range(n) if digits.target[i] == positive_digit]
negative_idx = [i for i in range(n) if not digits.target[i] == positive_digit]

original_data = digits.data[positive_idx + negative_idx, ...]
data = original_data + 10 * np.random.randn(original_data.shape[0], original_data.shape[1])
labels = [True] * len(positive_idx) + [False] * len(negative_idx)

objfun = hpo.make_svm_objfun(data, labels, num_folds)

hpo_search = hpo.svm_search_space(**search)
hpo.setup_hpolib(hpo.negate(objfun), hpo_search, budget, name)
