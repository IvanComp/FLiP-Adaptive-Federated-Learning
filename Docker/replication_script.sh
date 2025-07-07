./run_experiments.sh --repl $1 --config_name no-selector --iid 40 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 40 --high 5 --low 5 --threshold 0.35
./run_experiments.sh --repl $1 --config_name tree-adaptive --iid 40 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name always-selector --iid 40 --high 10 --low 0