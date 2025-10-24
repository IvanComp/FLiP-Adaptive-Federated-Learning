# AP4Fed Docker Package

This folder contains the Dockerized reference implementation of AP4Fed (Architectural Patterns for Federated Learning). It includes a Flower server and multiple Flower clients orchestrated via Docker Compose, plus an adaptation subsystem that turns architectural patterns on/off and tunes their parameters across FL rounds based on observed metrics.

- Project root: [../README.md](../README.md)

## Table of contents
- [Key goals](#key-goals)
- [Folder at a glance](#folder-at-a-glance)
- [Architectural Patterns supported](#architectural-patterns-supported)
- [How adaptation works](#how-adaptation-works)
- [AdaptationManager lifecycle](#adaptationmanager-lifecycle)
- [Activation strategies](#activation-strategies)
- [Configuration](#configuration)
- [Notes](#notes)
- [Quickstart](#quickstart)
- [Observe adaptation](#observe-adaptation)
- [Example: enabling client_selector](#example-enabling-client_selector-when-validation-accuracy-stalls)
- [Extending adaptation](#extending-adaptation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Key goals
- Run a repeatable FL experiment with containerized server/clients
- Enable/disable AP4Fed patterns per round
- Adapt the configuration automatically (adaptation) using several strategies

## Folder at a glance
- [server.py](./server.py): Flower server with multi-model strategy, aggregation, metrics collection, and adaptation loop
- [client.py](./client.py): Flower client that trains/evaluates and reports metrics (including timing, comm. stats, distribution info)
- [adaptation.py](./adaptation.py): AdaptationManager orchestrating which patterns are enabled next round and with which params
- [adaptation_utils/strategy.py](./adaptation_utils/strategy.py): Activation criteria strategies used by the AdaptationManager
- [configuration/](./configuration/): JSON config files ([config.json](./configuration/config.json) is the active one)
- [logger.py](./logger.py): Logs to console and [performance/output.log](./performance/output.log)
- [setup.py](./setup.py): Utility to generate configuration/config.json and docker-compose.dynamic.yml from templates
- docker-compose files: [docker-compose.dynamic.yml](./docker-compose.dynamic.yml), template in [template.docker-compose.dynamic.yml](./template.docker-compose.dynamic.yml) if present

## Architectural Patterns supported
The following pattern keys are recognized throughout the Docker package (also used in [configuration/config.json](./configuration/config.json)):
- client_selector
- client_cluster
- message_compressor
- model_co-versioning_registry
- multi-task_model_trainer
- heterogeneous_data_handler

Each pattern entry has the form:

```jsonc
{
  "<pattern_name>": {
    "enabled": true,
    "params": { /* optional, strategy-specific */ }
  }
}
```

## How adaptation works
- The server aggregates results each round and maintains a metrics_history structure.
- After aggregation, the server calls `AdaptationManager.config_next_round(metrics_history, last_round_time)`.
- AdaptationManager loads activation criteria from [configuration/config.json](./configuration/config.json) and evaluates whether each pattern should be enabled next round and which parameters should be set.
- The server applies the returned configuration immediately for the following round.

See implementation: [adaptation.py](./adaptation.py) and usage in [server.py](./server.py).

## AdaptationManager lifecycle
1) Init
- Loads CONFIG from [configuration/config.json](./configuration/config.json)
- Reads patterns list and `activation_criteria`
- Builds one or more ActivationCriterion instances via `adaptation_utils.strategy.get_activation_criteria`
- Caches the current pattern config and mirrors changes back into `configuration/config.json`

2) Per round
- Receives aggregated metrics and last round timing
- For every pattern listed under `activation_criteria`, executes its `ActivationCriterion.activate_pattern(args)`
  - args includes: model_type, metrics, and time of last round
- For each pattern, returns `(activate: bool, params: dict | None, explanation: str)`
- Updates the in-memory config and writes it back to `configuration/config.json`
- Returns the new patterns block to the server

## Activation strategies
Each strategy inspects a metric and decides whether a pattern should be enabled next round. See code: [adaptation_utils/strategy.py](./adaptation_utils/strategy.py).

- RandomActivationCriterion
  - Randomly toggles the pattern on/off. Optionally selects a subset of clients (`enabled_clients`) when ON.

- FixedGlobalThresholdActivationCriterion
  - Compares a globally aggregated metric to a fixed value.
  - metric_type: "increasing" -> enable when metric < value; "decreasing" -> enable when metric > value.

- FixedLocalThresholdActivationCriterion
  - Compares each client’s local metric to a fixed value; can enable pattern when some or all clients cross the threshold.

- PredictorBasedActivationCriterion
  - Loads a pickled predictive model (`threshold.predictor.model_path`) and uses it to compute an activation threshold from aggregated metrics.
  - `metric_type` applies as with FixedGlobal.

- PredictorBasedLocalActivationCriterion
  - As above, but driven by per-client signals.

- BayesianOptimizationActivationCriterion
  - Uses `skopt.gp_minimize` to explore a policy space to improve a target metric/time tradeoff based on observed history.

Note: `strategy.py` provides utility functions to derive environment descriptors (e.g., ratio of IID clients, high/low spec mix) that strategies may use.

## Configuration
At minimum, [configuration/config.json](./configuration/config.json) contains:

```jsonc
{
  "rounds": 20,
  "clients": 4,
  "adaptation": true,
  "patterns": {
    "client_selector": { "enabled": false, "params": {} },
    "client_cluster": { "enabled": false },
    "message_compressor": { "enabled": false },
    "model_co-versioning_registry": { "enabled": false },
    "multi-task_model_trainer": { "enabled": false },
    "heterogeneous_data_handler": { "enabled": false }
  },
  "client_details": [
    {
      "client_id": 1,
      "cpu": 2,                  // used to classify high/low spec
      "ram": 4,
      "dataset": "CIFAR-10",
      "data_distribution_type": "IID", // or "non-IID"
      "data_persistence_type": "New Data", // or "Same Data" | "Remove Data"
      "model": "CNN 16k",
      "epochs": 1
    }
    // ... more clients
  ],
  "activation_criteria": {
    "metrics": [
      {
        "name": "<strategy_name>",
        "pattern": "<pattern_name>",
        "metric": "<metric_key>",
        "threshold": {
          "type": "increasing", // or "decreasing"
          "calculation_method": "random" | "fixed-global" | "fixed-local" | "predictor_based" | "predictor-local" | "bayesian_optimization",
          // if calculation_method is fixed-*
          "value": 0.7,
          // if calculation_method is predictor*
          "predictor": {
            "model_path": "path/to/model.pkl"
          }
        }
      }
    ]
  }
}
```

## Notes
- The Docker package currently assumes all clients run the same model type for adaptation decisions when computing `model_type`. Mixed models may require extending `get_model_type` in [adaptation.py](./adaptation.py) and the strategies.
- Adaptation writes back effective pattern toggles and params to [configuration/config.json](./configuration/config.json) at each round (see `AdaptationManager.update_json`). This helps with reproducibility and debugging.
- Metrics available to adaptation come from server-side aggregation (`metrics_history`). Typical keys include train/val losses, accuracy, f1, mae, timing, and communication stats. See aggregation in [server.py](./server.py).

## Quickstart
1) Run a single experiment (recommended)
- From the Docker/ directory, use the helper script which prepares config, builds, runs, and archives results automatically:

```bash
# Make scripts executable (once)
chmod +x run_experiments.sh replication_script_full.sh

# Syntax
./run_experiments.sh \
  --repl <N> \
  --config_name <config> \
  --iid <percent_iid> \
  --high <num_high_spec_clients> \
  --low <num_low_spec_clients> \
  --data <new|same> \
  [--threshold <value>]

# Example: 0% IID, 2 high + 4 low, persisted data, with threshold
./run_experiments.sh --repl 6 --config_name fixed-hdh --iid 0 --high 2 --low 4 --data same --threshold 0.1
```
- Notes:
  - --repl controls how many replications to run; the script iterates from 6 up to N, so set N >= 6 for at least one run.
  - Internally, the script calls [setup.py](./setup.py), prunes Docker cache, composes up with rebuild, and copies logs from [performance/](./performance/) to results/.

2) Run the full replication suite
- To reproduce the paper’s grid of scenarios, use:

```bash
# Pass the maximum replication index (>= 6)
./replication_script_full.sh 6
```
- This invokes run_experiments.sh multiple times with the combinations shown inside [replication_script.sh](./replication_script.sh) (e.g., configs: no-hdh, random-hdh, always-hdh, fixed-hdh, tree-hdh, bo-hdh; various high/low mixes; data persistence modes; optional --threshold).

3) Where results and logs go
- Runtime logs: [performance/output.log](./performance/output.log) and console.
- Per-run archives: results/vm/<data>/<H>high-<L>low/<iid>iid/<config_name>_<rep_idx>/ with copies of performance/* from each execution.

Tip: If you prefer fully manual control, you can still use [setup.py](./setup.py) to generate configuration/config.json and docker-compose.dynamic.yml, then run docker compose yourself; however, the scripts above ensure correct parameters and reproducibility.

## Observe adaptation
- After each round, look for lines like:
  - `Adaptation Enabled ✅`
  - `<pattern_name> activated ✅` or `de-activated ❌`
- The [configuration/config.json](./configuration/config.json) file will be updated with the current `enabled`/`params` for each pattern.

## Example: enabling client_selector when validation accuracy stalls
In [configuration/config.json](./configuration/config.json):

```jsonc
{
  "adaptation": true,
  "activation_criteria": {
    "metrics": [
      {
        "name": "fixed-global-val-acc",
        "pattern": "client_selector",
        "metric": "val_accuracy",
        "threshold": {
          "type": "increasing",
          "calculation_method": "fixed-global",
          "value": 0.70
        }
      }
    ]
  }
}
```
Interpretation: If the aggregated `val_accuracy` is below 0.70, `client_selector` will be enabled next round; otherwise disabled.

## Extending adaptation
- Add a new strategy: implement a class in [adaptation_utils/strategy.py](./adaptation_utils/strategy.py) that extends `ActivationCriterion` and implement `activate_pattern(self, args)`.
- Register it in `get_activation_criteria` by branching on a new `calculation_method` string.
- Add the strategy to your [configuration/config.json](./configuration/config.json) under `activation_criteria`.

## Troubleshooting
- No effect from adaptation: verify `"adaptation": true`, check that `activation_criteria.metrics[0]` points to a pattern that is set enabled in default config to let AdaptationManager manage it.
- Predictor path errors: ensure `threshold.predictor.model_path` exists inside the container (mount or bake into image).
- Mixed model types: current code assumes a single model type across clients; verify your `client_details` or extend `get_model_type` and strategy logic.
- Logs not found: ensure [performance/](./performance/) directory exists and is mounted; [logger.py](./logger.py) writes to `performance/output.log`.

## License
See repository-level license (e.g., [../LICENSE](../LICENSE) if present).
