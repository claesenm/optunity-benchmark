import optunity
import executable
import sys
import cloudpickle as pickle

name = sys.argv[1]

budget = 150
search={'logC': [-8, 1], 'logGamma': [-8, 1]}

objfun = executable.unpickled['objfun']

def quacking_objfun(**kwargs):
    result = objfun(**kwargs)
    print(str(kwargs) + ' --> %f' % result)
    return result

if __name__ == '__main__':

    pars, info, _ = optunity.maximize(quacking_objfun, num_evals=budget, pmap=optunity.pmap, **search)
    df = optunity.call_log2dataframe(info.call_log)

    with open('results/%s-optunity.pkl' % name, 'w') as f:
        log = {'results': df['value'], 'logC': df['logC'], 'logGamma': df['logGamma']}
        pickle.dump(log, f)

