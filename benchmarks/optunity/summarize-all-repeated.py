import cloudpickle as pickle
import sys
import operator as op
import numpy as np
import pandas

budget = int(sys.argv[1])
start_index = int(sys.argv[2])
stop_index = int(sys.argv[3])

datasets = ['digits-%d' % i for i in range(10)] + ['covtype-%d' % i for i in range(1, 8)] + ['adult', 'diabetes', 'breast-cancer', 'ionosphere']
datasets = ['digits-%d' % i for i in range(10)] + ['covtype-%d' % i for i in range(1, 8)] + ['diabetes', 'ionosphere']
solvers = ['optunity', 'tpe', 'smac', 'bayesopt', 'random']
solvermap = {'optunity': 'Optunity',
             'tpe': 'Hyperopt',
             'smac': 'SMAC',
             'bayesopt': 'BayesOpt',
             'random': 'random'}

all_results = {k: [] for k in datasets}
performance = {}
random90 = []

for repetition in range(start_index, stop_index + 1):
    random90.append({})
    for dataset in datasets:
        all_results[dataset].append({})
        with open('results-repeated/%s-all.pkl-%d' % (dataset, repetition), 'r') as f:
            results = pickle.load(f)
            performance = {}
            for solver in solvers:
                if solver == 'optunity':
                    perf = max(results[solver]['results'][:budget])
                else:
                    perf = -min(results[solver]['results'][:budget])
                all_results[dataset][-1][solver] = perf

                if solver == 'random':
                    srtd = sorted(results[solver]['results'][:budget], reverse=True)
                    random90[-1][dataset] = -srtd[int(0.75 * len(srtd))]


ranks = {solver: {data: [] for data in datasets} for solver in solvers}
bests = {dataset: [] for dataset in datasets}
for dataset in datasets:
    for idx in range(len(all_results[dataset])):
        srtd = sorted(all_results[dataset][idx].items(), key=op.itemgetter(1), reverse=True)
        for i, d in enumerate(srtd):
            ranks[d[0]][dataset].append(i + 1)
        bests[dataset].append(max(all_results[dataset][idx].values()))

mean = lambda x: float(sum(x)) / len(x)

def stringify_rank(value, lst):
    if value == max(lst): return "{\\color{gray}%1.2f}" % value
    elif value == min(lst): return "\\textbf{%1.2f}" % value
    else: return "%1.2f" % value

def stringify_regret(value, lst):
    if value == max(lst): return "{\\color{gray}%1.3f}" % (100 * value)
    elif value == min(lst): return "\\textbf{%1.3f}" % (100 * value)
    else: return "%1.3f" % (100 * value)


print('solver\taverage rank')
overall_ranks = {solver: [] for solver in solvers}
avg_rank = {dataset: {} for dataset in datasets}
avg_regret = {dataset: {} for dataset in datasets}
avg_bests = {dataset: mean(bests[dataset]) for dataset in datasets}
avg_q3s = {dataset: mean(map(lambda x: x[dataset], random90))
           for dataset in datasets}
for dataset in datasets:
    for solver in solvers:
        avg_regret[dataset][solver] = sum(map(lambda idx: bests[dataset][idx] - all_results[dataset][idx][solver],
                                              range(stop_index-start_index))) / (stop_index - start_index)
        avg_rank[dataset][solver] = mean(ranks[solver][dataset])

        overall_ranks[solver].append(avg_rank[dataset][solver])
        print('%s\t%s\t%1.3f' % (dataset, solver, overall_ranks[solver][-1]))

for solver in solvers:
    print('%s\t%1.3f' % (solver, mean(overall_ranks[solver])))

for dataset in datasets:
    string = '\\texttt{%s} & %1.2f & %1.2f & & ' % (dataset, 100 * avg_bests[dataset], 100 * avg_q3s[dataset])
    string += ' & & '.join(map(lambda solver: '%s & %s' % (stringify_rank(avg_rank[dataset][solver],
                                                                          avg_rank[dataset].values()),
                                                           stringify_regret(avg_regret[dataset][solver],
                                                                            avg_regret[dataset].values())),
                               solvers))
    print(string + '\\\\')

regret_per_solver = {solver: mean([avg_regret[data][solver] for data in datasets]) for solver in solvers}

allmeansranks = map(mean, overall_ranks.values())

print('\\midrule')
print('average & N/A & N/A & & %s \\\\' % ' & & '.join(["%s & %s" % (stringify_rank(mean(overall_ranks[solver]), allmeansranks),
                                                                     stringify_regret(regret_per_solver[solver], regret_per_solver.values()))
                                                        for solver in solvers]))

#print('\n')
#print('dataset \t' + "\t".join(map(lambda x: solvermap[x], solvers)) + "\tQ_3")
#for dataset in datasets:
#    print('%s' % dataset + '\t' + '\t'.join(map(lambda x: "%1.2f" % (100 * all_results[dataset][x]), solvers)) + "\t%1.2f" % (100.0 * random90[dataset]))

#print('average rank \t' + "\t".join(map(lambda x: "%1.3f" % (float(sum(ranks[x])) / len(ranks[x])), solvers)) + "\tN/A")
