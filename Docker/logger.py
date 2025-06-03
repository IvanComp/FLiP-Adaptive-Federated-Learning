from flwr.common.logger import log as flwr_log

LOG_PATH = 'performance/output.log'


def log(level, msg: str):
    flwr_log(level, msg)
    with open(LOG_PATH, 'a') as f:
        f.write(str(msg) + '\n')
