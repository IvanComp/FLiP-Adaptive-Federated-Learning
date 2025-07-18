./run_experiments.sh --repl $1 --config_name no-selector --iid 0 --high 8 --low 4
./run_experiments.sh --repl $1 --config_name random-selector --iid 0 --high 8 --low 4
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 0 --high 8 --low 4 --threshold 0.3
./run_experiments.sh --repl $1 --config_name always-selector --iid 100 --high 12 --low 0
./run_experiments.sh --repl $1 --config_name always-selector --iid 0 --high 12 --low 0