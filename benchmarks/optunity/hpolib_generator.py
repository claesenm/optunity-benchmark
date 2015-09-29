from __future__ import print_function
import numpy as np
from hyperopt import hp
import math
import optunity
import sklearn.svm
import optunity.metrics
import cloudpickle as pickle
import functools

def hpolib_wrapper(objfun, search, budget, result_on_terminate=0.0):

    # construct the typemap (dict mapping argument names to their types)
    # the typemap is used to reverse type erasure by serialization
    # we use the search space definition to acquire types
    hp2type = {'float': float,
               'switch': str
               }
    typemap = {k: hp2type[v.name] for k, v in search.items()}

    with open('/tmp/data.pkl', 'w') as f:
        pickle.dump({"objfun": objfun, "typemap": typemap, "search": search}, f)

def negate(fun):
    @functools.wraps(fun)
    def wrapped(**kwargs):
        return -fun(**kwargs)
    return wrapped

def make_svm_objfun(data, labels, num_folds):

    @optunity.cross_validated(x=data, y=labels, num_folds=5, regenerate_folds=True)
    def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, logC, logGamma):
        if type(logC) == np.array: logC = logC[0]
        if type(logGamma) == np.array: logGamma = logGamma[0]
        model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc

    return svm_rbf_tuned_auroc

def svm_search_space(logC, logGamma):
    return {'logC': hp.uniform('logC', logC[0], logC[1]),
            'logGamma': hp.uniform('logGamma', logGamma[0], logGamma[1])}

def setup_hpolib(objfun, search, budget, name):
    hpolib_wrapper(objfun, search, budget)
    return None
