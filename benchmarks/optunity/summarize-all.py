import cloudpickle as pickle
import sys
import operator as op
import numpy as np
import pandas

budget = int(sys.argv[1])
try:
    repetition = int(sys.argv[2])
except:
    repetition = None

datasets = ['digits-%d' % i for i in range(10)] + ['covtype-%d' % i for i in range(1, 8)] + ['adult', 'diabetes', 'breast-cancer', 'ionosphere']
datasets = ['digits-%d' % i for i in range(10)] + ['covtype-%d' % i for i in range(1, 8)] + ['diabetes', 'ionosphere']
solvers = ['optunity', 'tpe', 'smac', 'bayesopt', 'random']
solvermap = {'optunity': 'Optunity',
             'tpe': 'Hyperopt',
             'smac': 'SMAC',
             'bayesopt': 'BayesOpt',
             'random': 'random'}

all_results = {}
performance = {}
random90 = {}

for dataset in datasets:
    all_results[dataset] = {}
    if repetition:
        with open('results-repeated/%s-all.pkl-%d' % (dataset, repetition), 'r') as f:
            results = pickle.load(f)
            performance = {}
            for solver in solvers:
                if solver == 'optunity':
                    perf = max(results[solver]['results'][:budget])
                else:
                    perf = -min(results[solver]['results'][:budget])
                all_results[dataset][solver] = perf

                if solver == 'random':
                    srtd = sorted(results[solver]['results'], reverse=True)
                    random90[dataset] = -srtd[int(0.75 * len(srtd))]

    else:
        with open('results/%s-all.pkl' % dataset, 'r') as f:
            results = pickle.load(f)
            performance = {}
            for solver in solvers:
                if solver == 'optunity':
                    perf = max(results[solver]['results'][:budget])
                else:
                    perf = -min(results[solver]['results'][:budget])
                all_results[dataset][solver] = perf

                if solver == 'random':
                    srtd = sorted(results[solver]['results'], reverse=True)
                    random90[dataset] = -srtd[int(0.75 * len(srtd))]

ranks = {solver: [] for solver in solvers}
for dataset in datasets:
    srtd = sorted(all_results[dataset].items(), key=op.itemgetter(1), reverse=True)
    for i, d in enumerate(srtd):
        ranks[d[0]].append(i + 1)

print('solver\taverage rank')
for solver in solvers:
    print('%s\t%1.3f' % (solver, float(sum(ranks[solver])) / len(ranks[solver])))

print('\n')
print('dataset \t' + "\t".join(map(lambda x: solvermap[x], solvers)) + "\tQ_3")
for dataset in datasets:
    print('%s' % dataset + '\t' + '\t'.join(map(lambda x: "%1.2f" % (100 * all_results[dataset][x]), solvers)) + "\t%1.2f" % (100.0 * random90[dataset]))

print('average rank \t' + "\t".join(map(lambda x: "%1.3f" % (float(sum(ranks[x])) / len(ranks[x])), solvers)) + "\tN/A")
