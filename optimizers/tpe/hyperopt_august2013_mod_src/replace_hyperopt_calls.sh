folders="config_parser format_converter Plotting ."
recursive_folders="benchmarks tests"

for folder in $folders
  do
    find $folder -maxdepth 1 -name "*.py" | xargs sed -i 's/^import hyperopt$/import optimizers.hyperopt_august2013_mod.hyperopt as hyperopt/g'
    find $folder -maxdepth 1 -name "*.py" | xargs sed -i 's/^from hyperopt import hp$/from optimizers.hyperopt_august2013_mod.hyperopt import hp/g'
  done

for folder in $recursive_folders:
  do
    find $folder -name "*.py" | xargs sed -i 's/import hyperopt/import optimizers.hyperopt_august2013_mod.hyperopt as hyperopt/g'
    find $folder -name "*.py" | xargs sed -i 's/^from hyperopt import hp$/from optimizers.hyperopt_august2013_mod.hyperopt import hp/g'
  done
