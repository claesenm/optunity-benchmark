import cloudpickle as pickle
import sys

name = sys.argv[1]

solvers = ['optunity', 'tpe', 'random', 'smac', 'bayesopt']

results = {}

for solver in solvers:
    with open('results/%s-%s.pkl' % (name, solver), 'r') as f:
        results[solver] = pickle.load(f)

with open('results/%s-all.pkl' % name, 'w') as f:
    pickle.dump(results, f)
