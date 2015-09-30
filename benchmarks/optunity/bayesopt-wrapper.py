import optunity
import numpy as np
import sklearn
import sklearn.svm
import optunity.metrics
import cloudpickle as pickle
import bayesopt

def prepare_objfun(fun, search):

    def wrapper(args):
        kwargs = {k: v for k, v in zip(search.keys(), args)}
        result = fun(**kwargs)
        try:
            with open('/tmp/results.pkl', 'r') as f: data = pickle.load(f)
        except (IOError, EOFError):
            data = {"kwargs": [], "results": []}
        with open('/tmp/results.pkl', 'w') as f:
            data["kwargs"].append(kwargs)
            data["results"].append(result)
            pickle.dump(data, f)
        return result

    return wrapper


if __name__ == '__main__':
    with open('/tmp/data.pkl', 'r') as f:
        context = pickle.load(f)
        objfun = context['objfun']
        search = context['search']
        typemap = context['typemap']

    objfun = prepare_objfun(objfun, search)

    params = {}
    params['n_iterations'] = 150
#    params['n_iter_relearn'] = 5

    # configuration used throughout experiments
    # search={'logC': [-8, 1], 'logGamma': [-8, 1]}
    n = 2                     # n dimensions
    lb = -8 * np.ones((n,))
    ub = np.ones((n,))

    mvalue, x_out, error = bayesopt.optimize(objfun, n, lb, ub, params)
    print('mvalue: %1.3f' % mvalue)
