smac-experiment: config.cfg params-smac.pcs
	HPOlib-run -o ../../optimizers/smac/smac

tpe-experiment: config.cfg space.py
	HPOlib-run -o ../../optimizers/tpe/hyperopt_august2013_mod

random-experiment: config.cfg space.py
	HPOlib-run -o ../../optimizers/tpe/random_hyperopt_august2013_mod

spearmint-experiment: config.cfg params-spearmint.pcs
	HPOlib-run -o ../../optimizers/spearmint/spearmint_april2013_mod

params-smac.pcs: space.py
	python ../../scripts/HPOlib-convert --from tpe --to smac -f space.py -s params-smac.pcs

params-spearmint.pcs: space.py
	python ../../scripts/HPOlib-convert --from tpe --to spearmint -f space.py -s params-spearmint.pcs

.PHONY: clean

clean:
	rm -f params-smac.pcs params-spearmint.pcs
	rm -rf smac_*
	rm -rf spearmint_*
	rm -rf hyperopt_*
	rm -rf random_*

results/%-optunity.pkl: run-optunity.py /tmp/data.pkl
	python run-optunity.py $*

results/%-tpe.pkl: executable.py /tmp/data.pkl
	rm -f /tmp/results.pkl
	make tpe-experiment
	mv /tmp/results.pkl $@

results/%-smac.pkl: executable.py /tmp/data.pkl
	rm -f /tmp/results.pkl
	make smac-experiment
	mv /tmp/results.pkl $@

results/%-spearmint.pkl: executable.py /tmp/data.pkl
	rm -f /tmp/results.pkl
	make spearmint-experiment
	mv /tmp/results.pkl $@

results/%-random.pkl: executable.py /tmp/data.pkl
	rm -f /tmp/results.pkl
	make random-experiment
	mv /tmp/results.pkl $@

results/%-all.pkl: fuse.py %.py results/%-optunity.pkl results/%-tpe.pkl results/%-random.pkl results/%-smac.pkl results/%-bayesopt.pkl
	python fuse.py $*
	cp /tmp/data.pkl results/$*-data.pkl
	python summarize.py $*

results/%-bayesopt.pkl: bayesopt-wrapper.py /tmp/data.pkl
	rm -f /tmp/results.pkl
	python bayesopt-wrapper.py $*
	mv /tmp/results.pkl $@

.PRECIOUS: results/%-optunity.pkl results/%-tpe.pkl results/%-random.pkl results/%-smac.pkl results/%-bayesopt.pkl
