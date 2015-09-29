import optunity
import optunity.metrics
import sklearn
import sklearn.svm
import numpy as np
import math
import hyperopt as hp
import cloudpickle as pickle
import time
import HPOlib.benchmark_util as benchmark_util
with open("/tmp/data.pkl", "r") as f: unpickled = pickle.load(f)
recover_types = lambda dict: {k: unpickled["typemap"][k](v) for k, v in dict.items()}

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

