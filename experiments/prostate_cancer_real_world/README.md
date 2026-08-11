# Real-world prostate-cancer experiment for FLiP

This folder contains the isolated Docker/Flower experiment that evaluates FLiP
on the real cross-hospital prostate-cancer scenario from Kazlouski et al.,
*Towards Practical Federated Learning and Evaluation for Medical Prediction
Models*. Nothing under the main project `Docker/` directory is modified.

## Distributed topology

The canonical experiment is a real Flower deployment composed of:

- one Flower server container, which never mounts patient data;
- 11 Flower client containers, one per hospital silo;
- one private Docker bridge network;
- a read-only bind mount containing exactly one silo file in each client;
- real local scikit-learn training and sample-weighted FedAvg;
- an evaluation RPC after each aggregation, involving all 11 clients.

The original processed `data.pkl` is already a dictionary of hospital-specific
DataFrames. `prepare_docker_data.py` preserves those boundaries and writes 11
separate `.npz` files. It never pools or redistributes patient rows. Turkey and
Finland are excluded as in the reproduced clinical scenario, leaving 5,624
patients. Each silo receives its own stratified 80/20 train/test split.

| Client | Silo | Patients | CPU limit |
|---:|---|---:|---:|
| 1 | Spain | 1,640 | 2 |
| 2 | Korea | 999 | 1 |
| 3 | Germany1 | 774 | 2 |
| 4 | China2 | 530 | 1 |
| 5 | China1 | 312 | 2 |
| 6 | USA2 | 310 | 1 |
| 7 | USA1 | 280 | 2 |
| 8 | Netherlands | 266 | 1 |
| 9 | Italy | 218 | 2 |
| 10 | Germany2 | 162 | 1 |
| 11 | UK | 133 | 2 |

The clinical source does not report computing resources. The 1/2 CPU assignment
therefore follows the FLiP experimental setup and is a systems condition, not a
clinical attribute.

## Protocol and patterns

The experiment retains the clinical preprocessing, PSA spline knots, PI-RADS
encoding, label, seed 1234, L2 logistic regression with L-BFGS, and the fixed
stratified split. It runs 20 FL rounds and one repetition.

The Docker deployment supports the FLiP methods `never`, `random`, `always`, and
`fliponline`, applying the following patterns round by round:

- Client Selector: a resource threshold of 2 CPUs;
- Heterogeneous Data Handler: random minority oversampling for tabular clinical
  data;
- Message Compressor: real zlib/LZ77 level-1 payloads transferred through gRPC.

`fliponline` imports the main project's
`ContextualBanditActivationCriterion` and
`ContextualBanditLocalActivationCriterion` directly from
`../../Docker/adaptation_utils/strategy.py`. Docker Compose mounts that main
directory read-only in the server container.

The unstable condition does not use a virtual clock. Every selected client
actually sleeps for an independently generated integer delay from 0 through 20
seconds before returning its model update, matching FLiP's Docker experiments.
The stable condition injects no delay. The Docker CPU limits are also enforced
on the running containers.

The clinical paper performs one federated logistic-regression fit. FLiP needs
multiple observations to adapt, so the same local operation is repeated for 20
rounds and warm-started from the current global coefficients. The main FLiP HDH
uses image/text GANs; those cannot generate valid tabular clinical rows, so the
explicit tabular adapter balances the minority label by resampling existing
local observations without inventing clinical values.

## Running the experiment

Docker Desktop (or Docker Engine with Compose) must be running. Prepare and run
the requested experiment with:

```bash
cd experiments/prostate_cancer_real_world
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_docker_experiment.py \
  --method fliponline \
  --network unstable \
  --rounds 20 \
  --repetition 1
python validate_docker_results.py unstable_fliponline_rep1
```

The runner downloads and verifies the authors' `data.pkl`, materializes the 11
isolated client files, builds the image, starts all 12 containers, waits for the
server to finish, and tears the containers down. Completed results are never
silently overwritten.

To execute the complete comparison used by the FLiP experiments:

```bash
python run_docker_campaign.py
```

This runs four methods under stable and unstable network conditions, 20 rounds
each, with one repetition. Already completed scenarios with matching round
counts are skipped.

For a fast connectivity test:

```bash
python run_docker_experiment.py \
  --method never --network stable --rounds 2 --repetition 99
```

## Outputs

Every run writes to `docker_results/<network>_<method>_rep<repetition>/`:

- `manifest.json`: topology, method, network, clients, and CPU assignments;
- `round_metrics.csv`: global AUC, F1, accuracy, loss, time, communication,
  traffic, and pattern state;
- `client_round_metrics.csv`: one row for every client and every round,
  including participation, delay, training time, traffic, and local metrics;
- `local_metrics.csv`: global-model evaluation on every hospital test split;
- `summary.json`: final and cumulative FLiP metrics.

The full campaign also creates combined CSVs and plots in `docker_results/`.
Because the requested campaign has one repetition, its results are descriptive
and must not be presented with confidence intervals or inferential claims.

## Local reference implementation

`run_experiment.py` and `run_campaign.py` retain the earlier single-process
reference implementation for debugging numerical behavior. They are not the
canonical experiment and must not be used as evidence of a distributed run.
Only the `run_docker_*` entry points create the 11 real client containers.

## Data and clinical-use notice

The source repository distributes the extracted, processed `data.pkl`, while
the underlying raw datasets retain their original licenses. Review those
licenses before redistributing data. This artifact is an experimental
replication and is not a medical device or a clinical decision tool.
