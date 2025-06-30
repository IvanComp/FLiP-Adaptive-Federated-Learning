for i in $(seq 1 $1)
do
	python3 reset_config.py
	sudo docker compose -f 'docker-compose.dynamic.yml' 'up'
	mkdir results/vm-iid/$2/$3_$i
	cp -a performance/* results/vm-iid/$2/$3_$i/
done
