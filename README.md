---
Version: [1.5.0]
Main Libraries: [Flower, PyTorch, Torchvision]
Testing Datasets: [torchvision.datasets, torchtext.datasets]
Testing Models: [torchvision.models (+120)]
---

# FLiP

This repository contains AP4Fed (Architectural Patterns for Federated Learning), a Flower-based reference implementation equipped with an adaptation subsystem that enables and disables architectural patterns and tunes their parameters across FL rounds. The implementation builds on the ideas introduced in FLiP [1], where architectural patterns are toggled at runtime (round-by-round) to better match evolving system conditions and performance goals. 

# References

[1] Baresi, L., Compagnucci, I., Lestingi, L., & Trubiani, C. (2026). Adaptive Toggling of Architectural Patterns for Federated Learning.
Accepted for the International Conference on Software Engineering for Adaptive and Self-Managing Systems (SEAMS) 2026. Rio de Janeiro, Brazil, 13–14 April 2026.

```
@inproceedings{BaresiSEAMSFlIP,
  author    = {Baresi, Luciano and Compagnucci, Ivan and Lestingi, Livia and Trubiani, Catia},
  title     = {{Adaptive Toggling of Architectural Patterns for Federated Learning}},
  booktitle = {International Conference on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  date      = {13--14 April 2026}
}

```

_JSS__Adaptive_FL_Patterns__ECS…

- Detailed Dockerized implementation and adaptation docs: [Docker/README.md](Docker/README.md)

## Results analysis (Docker/results)
The [Docker/results](Docker/results) folder contains the experimental results. Below is a brief description of scripts processing results to address three research questions.

### RQ1 — FLiP Effectiveness ([Docker/results/rq1.py](Docker/results/rq1.py))
- What it does: Performs RQ1 comparative analysis across architectural patterns and activation strategies (e.g., selector, heterogeneous_data_handler (hdh), compressor). It aggregates metrics from replicated runs, filters scenarios (e.g., N_high vs N_low mixes), generates plots, and runs non-parametric statistical tests.
- Methods: Mann–Whitney U test for pairwise comparisons and Vargha–Delaney A effect size. Includes outlier filtering and per-round vs cumulative metric views.
- Outputs: Prints statistical summaries to console and, if enabled, saves figures under `Docker/results/plots/rq1/`.
- Quickstart:
  ```bash
  # From the repository root
  python3 Docker/results/rq1.py
  ```

### RQ2 — Offline Overhead ([Docker/results/rq2.ipynb](Docker/results/rq2.ipynb))
- What it does: Evaluates the quality of predictor-based components used in adaptation by fitting/assessing simple predictive models per pattern (selector, hdh, compressor).
- Outputs: Saves regression quality plots (RMSE and R-squared) under `Docker/results/plots/rq2/` (e.g., `rq2_rmse_selector.pdf`, `rq2_rsquared_selector.pdf`, analogous for hdh and compressor).
- Quickstart:
  ```bash
  # Option A: Execute the notebook headlessly and save an executed copy
  jupyter nbconvert --to notebook --execute Docker/results/rq2.ipynb \
    --output Docker/results/rq2.executed.ipynb

  # Option B: Open interactively
  jupyter notebook Docker/results/rq2.ipynb
  ```

### RQ3 — Online Overhead ([Docker/results/rq3.ipynb](Docker/results/rq3.ipynb))
- What it does: Analyzes time-related behavior across configurations/patterns, e.g., round duration and communication/training time components, and compares distributions across scenarios.
- Outputs: Saves boxplots under `Docker/results/plots/rq3/` (e.g., `rq3_time_boxplot.pdf`).
- Quickstart:
  ```bash
  # Option A: Execute the notebook headlessly and save an executed copy
  jupyter nbconvert --to notebook --execute Docker/results/rq3.ipynb \
    --output Docker/results/rq3.executed.ipynb

  # Option B: Open interactively
  jupyter notebook Docker/results/rq3.ipynb
  ```

If you are looking to run the experiments and produce the input data for these analyses, see [Docker/README.md](Docker/README.md).
