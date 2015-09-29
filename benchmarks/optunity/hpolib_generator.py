from __future__ import print_function
import numpy as np
from hyperopt import hp
import subprocess
import math
import optunity
import sklearn.svm
import optunity.metrics
import cloudpickle as pickle
import functools

def write_executable_file(exe_name, pickle_name, objfun, typemap, search, imports,
                    solver_name, problem_name):
    # write objective function and typemap to pickle file
    with open(pickle_name, 'w') as f:
        pickle.dump({"objfun": objfun, "typemap": typemap, "search": search}, f)

    # write the executable file
    with open(exe_name, 'w') as f:
        for i in imports:
            print('import %s' % i, file=f)

        # load objective function and typemap
        print('with open("%s", "r") as f: unpickled = pickle.load(f)' % pickle_name, file=f)
        print('recover_types = lambda dict: {k: unpickled["typemap"][k](v) for k, v in dict.items()}', file=f)

        print("""
if __name__ == "__main__":
    starttime = time.time()
    # Use a library function which parses the command line call
    args, params = benchmark_util.parse_cli()
    typed_params = recover_types(params)
    result = unpickled["objfun"](**typed_params)

    # write external file to track results
    try:
        with open('/tmp/results.pkl', 'r') as f: data = pickle.load(f)
    except (IOError, EOFError):
        data = {"kwargs": [], "results": []}
    with open('/tmp/results.pkl', 'w') as f:
        data["kwargs"].append(typed_params)
        data["results"].append(result)
        pickle.dump(data, f)

    # output result
    duration = time.time() - starttime
    print("Result for ParamILS: SAT, %f, 1, %f, %d, %s" % (abs(duration), result, -1, str(__file__)))
    """, file=f)



def hpolib_wrapper(objfun, search, budget, solver_name, problem_name,
                   exe_name="executable.py",
                   prefix="/data/git/HPOlib/benchmarks/optunity-test/",
                   pickle_name="pickled_data.pkl",
                   config_name="config.cfg",
                   search_name="space.py",
                   hpolib_path="/data/git/HPOlib/",
                   solver_suffix="optimizers/tpe/hyperopt_august2013_mod",
                   result_on_terminate=0.0):

    imports = ['optunity', 'optunity.metrics',
               'sklearn', 'sklearn.svm',
               'numpy as np', 'math',
               'hyperopt as hp']

    if not "cloudpickle as pickle" in imports: imports.append("cloudpickle as pickle")
    if not "time" in imports: imports.append("time")
    if not "sys" in imports: imports.append("sys")
    imports.append("HPOlib.benchmark_util as benchmark_util")

    # construct the typemap (dict mapping argument names to their types)
    # the typemap is used to reverse type erasure by serialization
    # we use the search space definition to acquire types
    hp2type = {'float': float,
               'switch': str
               }
    typemap = {k: hp2type[v.name] for k, v in search.items()}

    print("prefix " + prefix)

    #create config
    with open(prefix + config_name, 'w') as f:
        print("[TPE]\nspace=%s\n" % (search_name), file=f)
        print("[SPEARMINT]\nconfig=params-spearmint.pcs\n", file=f)
        print("[SMAC]\np=%sparams-smac.pcs\n" % prefix, file=f)
        print("[HPOLIB]", file=f)
        print("function=python ../%s" % exe_name, file=f)
        print("number_of_jobs=%d" % budget, file=f)
        print("result_on_terminate=%d" % result_on_terminate, file=f)

    write_executable_file(exe_name=prefix + exe_name,
                    pickle_name=prefix + pickle_name,
                    objfun=objfun,
                    typemap=typemap,
                    search=search,
                    imports=imports,
                    solver_name=solver_name,
                    problem_name=problem_name)

    # write search space file
    with open(prefix + search_name, 'w') as f:
        print('import cloudpickle as pickle', file=f)
        print('with open("%s%s", "r") as f: unpickled = pickle.load(f)' % (prefix, pickle_name), file=f)
        print('space = unpickled["search"]', file=f)


def simple_data():
    npos = 200
    nneg = 200

    delta = 2 * math.pi / npos
    radius = 2
    circle = np.array(([(radius * math.sin(i * delta),
                        radius * math.cos(i * delta))
                        for i in range(npos)]))

    neg = np.random.randn(nneg, 2)
    pos = np.random.randn(npos, 2) + circle

    data = np.vstack((neg, pos))
    labels = np.array([False] * nneg + [True] * npos)
    return data, labels

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
    hpolib_wrapper(objfun, search, budget, "TPE", name)
    return None
