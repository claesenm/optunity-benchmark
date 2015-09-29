import cloudpickle as pickle
import sys
import operator as op

name = sys.argv[1]
budget = 150

solvers = ['optunity', 'tpe', 'random', 'smac', 'bayesopt']

with open('results/%s-all.pkl' % name, 'r') as f:
    results = pickle.load(f)

performance = {}
for solver in solvers:
    if solver == 'optunity':
        perf = max(results[solver]['results'][:budget])
    else:
        perf = -min(results[solver]['results'][:budget])

    performance[solver] = perf

print("\n\nSUMMARY FOR %s\n" % name)
print("\n".join(map(str, sorted(performance.items(),
                                key=op.itemgetter(1),
                                reverse=True))))

