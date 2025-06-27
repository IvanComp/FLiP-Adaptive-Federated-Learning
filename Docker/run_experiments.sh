for i in $(seq 1 $1)
do
	sudo docker compose -f 'docker-compose.dynamic.yml' 'up'
	mkdir results/vm-iid/always-selector_$i
	cp -a performance/* results/vm-iid/always-selector_$i/
	python reset_config.py
done
