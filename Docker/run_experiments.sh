# Defaults
repl=""
config_name=""
iid=""
high=""
low=""
threshold=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --repl)
      repl="$2"
      shift 2
      ;;
    --config_name)
      config_name="$2"
      shift 2
      ;;
    --iid)
      iid="$2"
      shift 2
      ;;
    --high)
      high="$2"
      shift 2
      ;;
    --low)
      low="$2"
      shift 2
      ;;
	--threshold)
	  threshold="$2"
	  shift 2
	  ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Type check: ensure REPL is an integer
if ! [[ "$repl" =~ ^[0-9]+$ ]]; then
  echo "Error: --repl must be an integer (got '$repl')" >&2
  exit 1
fi

for i in $(seq 6 $((6 + repl)))
do
	python3 setup.py $config_name $iid $high $low $threshold
	sudo docker system prune -f
	sudo docker compose -f 'docker-compose.dynamic.yml' 'up'
	mkdir results/vm/${high}high-${low}low/${iid}iid/${config_name}_$i
	cp -a performance/* results/vm/${high}high-${low}low/${iid}iid/${config_name}_$i/
done
