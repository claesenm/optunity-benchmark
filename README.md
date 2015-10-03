Optunity benchmark (based on HPOlib)
==========================================

This repo contains the benchmark comparing [Optunity](http://git.optunity.net) to SMAC, Hyperopt, BayesOpt and random search.
The experimental setup is based on HPOlib, which provides SMAC, Hyperopt and random search.

HPOlib is licensed as GPL. All benchmark-specific code (in `benchmarks/optunity/`) is licensed under the [CRAPL](http://matt.might.net/articles/crapl/) and BSD simultaneously (TL;DR: feel free to use it for whatever but buyer beware).

To run the benchmark, a couple of Python libraries must be installed, in addition to HPOlib, 
namely `optunity`, `hyperopt`, `bayesopt`, `sklearn` and `cloudpickle`.

The benchmark itself is available in the `benchmarks/optunity` subfolder, and contains the following key elements:

- `run-experiment.sh <problem>` runs a full experiment for the specified problem (`digits-[1-9]`, `covtype-[1-7]`, `ionosphere` or `diabetes`)
- `run-all.sh` runs all experiments over 19 data sets, no commandline arguments
- `summarize.py problem budget` Python script to look at the results of a benchmark run for specified problem at the specified evaluation budget
- `summarize-all.py budget [repetition]` Python script to look at the results of a run of the full benchmark suite. 
    If repetition is omitted, the results of a user-generated run are reported, which must be a full benchmark (cfr. `run-all.sh`).
    If repetition is specified (1-5), the full results of that run as reported in the paper are shown.
- `summarize-all-repeated.py budget start stop` Python script to summarize all repeated runs from *start* to *stop* at given budget. 
    Calling `python summarize-all-repeated.py 75|150 1 5` will produce the latex tables as shown in the paper.

Additionally, a bunch of useful `make` recipes are available (specifically `make clean`, since HPOlib likes to generate a lot of folders).

Briefly, experiments are based on an automatically generated file `/tmp/data.pkl`, 
which contains the objective function to be used by all optimizers and some meta-data.
Function evaluations of all optimizers (except Optunity) are logged via side-effects in `/tmp/results.pkl` and later 
moved into the `benchmarks/optunity/results` folder. This approach is dirty but intuitive and effective.

The `benchmarks/optunity/results` folder will contain the following files after running a benchmark:

- `<problem>-[tpe|bayesopt|optunity|random|smac].pkl`: full trace of running given optimizer.
- `<problem>-data.pkl`: the objective function that was used and some meta-data.
- `<problem>-all.pkl`: a merger of all optimizer traces, used by `summarize.py` and `summarize-all.py`.
