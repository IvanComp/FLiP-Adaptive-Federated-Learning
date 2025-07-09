./run_experiments.sh --repl $1 --config_name no-selector --iid 100 --high 10 --low 10
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 100 --high 10 --low 10 --threshold 0.3
./run_experiments.sh --repl $1 --config_name tree-adaptive --iid 100 --high 10 --low 10
./run_experiments.sh --repl $1 --config_name always-selector --iid 100 --high 20 --low 0
./run_experiments.sh --repl $1 --config_name no-selector --iid 0 --high 10 --low 10
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 0 --high 10 --low 10 --threshold 0.3
./run_experiments.sh --repl $1 --config_name tree-adaptive --iid 0 --high 10 --low 10
./run_experiments.sh --repl $1 --config_name always-selector --iid 0 --high 20 --low 0