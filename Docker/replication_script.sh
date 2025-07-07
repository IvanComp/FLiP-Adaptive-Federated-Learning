./run_experiments.sh --repl $1 --config_name no-selector --iid 100 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 100 --high 5 --low 5 --threshold 0.4
./run_experiments.sh --repl $1 --config_name tree-adaptive --iid 100 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name always-selector --iid 100 --high 10 --low 0 
./run_experiments.sh --repl $1 --config_name no-selector --iid 40 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 40 --high 5 --low 5 --threshold 0.35
./run_experiments.sh --repl $1 --config_name tree-adaptive --iid 40 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name always-selector --iid 40 --high 10 --low 0
./run_experiments.sh --repl $1 --config_name no-selector --iid 0 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name fixed-adaptive --iid 0 --high 5 --low 5 --threshold 0.3
./run_experiments.sh --repl $1 --config_name tree-adaptive --iid 0 --high 5 --low 5
./run_experiments.sh --repl $1 --config_name always-selector --iid 0 --high 10 --low 0