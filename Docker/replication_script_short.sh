./run_experiments.sh --repl $1 --config_name no-selector --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name no-selector --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name no-hdh --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name no-hdh --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name no-compressor --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name no-compressor-delay --iid 0 --high 5 --low 5 --data same --delay yes
./run_experiments.sh --repl $1 --config_name always-selector --iid 0 --high 10 --low 0 --data same --delay no
./run_experiments.sh --repl $1 --config_name always-selector --iid 0 --high 10 --low 0 --data new --delay no
./run_experiments.sh --repl $1 --config_name always-hdh --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name always-hdh --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name always-compressor --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name always-compressor-delay --iid 0 --high 5 --low 5 --data same --delay yes
./run_experiments.sh --repl $1 --config_name random-selector --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name random-selector --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name random-hdh --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name random-hdh --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name random-compressor --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name random-compressor-delay --iid 0 --high 5 --low 5 --data same --delay yes
./run_experiments.sh --repl $1 --config_name fixed-selector --iid 0 --high 5 --low 5 --data same --delay no --threshold 0.3
./run_experiments.sh --repl $1 --config_name fixed-selector --iid 0 --high 5 --low 5 --data new --delay no --threshold 0.3
./run_experiments.sh --repl $1 --config_name fixed-hdh --iid 0 --high 5 --low 5 --data same --delay no --threshold 0.1
./run_experiments.sh --repl $1 --config_name fixed-hdh --iid 0 --high 5 --low 5 --data new --delay no --threshold 0.1
./run_experiments.sh --repl $1 --config_name fixed-compressor --iid 0 --high 5 --low 5 --data same --delay no --threshold 90
./run_experiments.sh --repl $1 --config_name fixed-compressor-delay --iid 0 --high 5 --low 5 --data same --delay yes --threshold 120
./run_experiments.sh --repl $1 --config_name tree-selector --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name tree-selector --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name tree-hdh --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name tree-hdh --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name tree-compressor --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name tree-compressor-delay --iid 0 --high 5 --low 5 --data same --delay yes
./run_experiments.sh --repl $1 --config_name bo-selector --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name bo-selector --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name bo-hdh --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name bo-hdh --iid 0 --high 5 --low 5 --data new --delay no
./run_experiments.sh --repl $1 --config_name bo-compressor --iid 0 --high 5 --low 5 --data same --delay no
./run_experiments.sh --repl $1 --config_name bo-compressor-delay --iid 0 --high 5 --low 5 --data same --delay yes
