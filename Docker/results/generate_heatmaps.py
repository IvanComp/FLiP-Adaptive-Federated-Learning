import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from skopt import gp_minimize
from skopt.space import Integer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "vm/fl_architectural_dataset.csv"
OUTPUT_DIR = "plots/policies"

# The notebook has used different filenames across experimental versions.
# The script tries these patterns in order and selects the most recently
# modified matching file.
PREDICTOR_MODEL_PATTERNS = [
    "../predictors/f1overtime_tree_selector_3040.pkl"
]

BO_MODEL_CANDIDATES = [
    "../predictors/bo_model.pkl"
]

BO_SCALER_CANDIDATES = [
    "../predictors/bo_scaler.pkl"
]

# Common axes used by the current Figure 4.
ROUNDS = np.arange(1, 21)
F1_OVER_TIME = np.arange(0.0001, 0.0100, 0.0005)

# Fixed-rule threshold used in the current visualization.
FIXED_THRESHOLD = 0.005

# Fixed context used to project the predictor/BO policies onto the
# two dimensions shown in Figure 4. Adjust these if a different
# representative federation configuration is desired.
N_HIGH = 5
N_LOW = 5
IID_VALUE = 100

# Settings inherited from the existing online-learning visualization.
ONLINE_MODEL_TYPE = "CNN 16k"
ONLINE_DATASETS = ("CIFAR10", "CIFAR-10")
ONLINE_ROUND_WINDOW = 1
ONLINE_EFFICIENCY_WINDOW = 0.001

# If no online observation is available around a heat-map cell, preserve
# the behavior of the original script and display the selector as active.
ONLINE_EMPTY_CELL_STATE = 1

# BO visualization settings, matching the notebook implementation.
BO_N_CALLS = 10
BO_RANDOM_STATE = 42


# ============================================================
# FILE RESOLUTION
# ============================================================

def newest_matching_file(patterns):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            "Could not find any file matching:\n  " + "\n  ".join(patterns)
        )

    return max(matches, key=os.path.getmtime)


def first_existing_file(candidates):
    for path in candidates:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Could not find any of:\n  " + "\n  ".join(candidates)
    )


# ============================================================
# POLICY SURFACES
# ============================================================

def build_fixed_surface():
    """Same simplified fixed-rule visualization used in the original Figure 4."""
    _, y_mesh = np.meshgrid(ROUNDS, F1_OVER_TIME)

    return (y_mesh >= FIXED_THRESHOLD).astype(int)


def build_predictor_surface(model):
    """
    Predictor-based policy:
    activate CS iff the predicted F1/time with Selector=ON is higher
    than with Selector=OFF.
    """
    z = np.zeros((len(F1_OVER_TIME), len(ROUNDS)), dtype=int)

    columns = [
        "High-Spec Clients",
        "Low-Spec Clients",
        "IID Data",
        "Selector",
        "FL Round",
        "Previous F1/Time",
    ]

    for row, prev_eff in enumerate(F1_OVER_TIME):
        for col, curr_round in enumerate(ROUNDS):
            x_on = pd.DataFrame(
                [[N_HIGH, N_LOW, IID_VALUE, True, curr_round,
                  prev_eff]],
                columns=columns,
            )
            x_off = pd.DataFrame(
                [[N_HIGH, N_LOW, IID_VALUE, False, curr_round,
                  prev_eff]],
                columns=columns,
            )

            pred_on = float(np.ravel(model.predict(x_on))[0])
            pred_off = float(np.ravel(model.predict(x_off))[0])

            z[row, col] = int(pred_on > pred_off)

    return z


def bo_objective(gpr, scaler, policy_on, curr_round, prev_eff):
    columns = [
        "High-Spec Clients",
        "Low-Spec Clients",
        "IID Data",
        "Selector",
        "FL Round",
        "Previous F1/Time",
    ]

    x = pd.DataFrame(
        [[N_HIGH, N_LOW, IID_VALUE, policy_on, curr_round, prev_eff]],
        columns=columns,
    )

    # Using a DataFrame also avoids scikit-learn warnings when the scaler
    # was originally fitted with feature names.
    x_scaled = scaler.transform(x)

    # gp_minimize minimizes, whereas the CS objective is to maximize F1/time.
    return -float(np.ravel(gpr.predict(x_scaled))[0])


def build_bo_surface(gpr, scaler):
    """
    BO-based policy using the same binary gp_minimize formulation as the
    training notebook, projected onto FL round x previous F1/time.
    """
    z = np.zeros((len(F1_OVER_TIME), len(ROUNDS)), dtype=int)

    for row, prev_eff in enumerate(F1_OVER_TIME):
        for col, curr_round in enumerate(ROUNDS):

            def wrapped_objective(action):
                return bo_objective(
                    gpr,
                    scaler,
                    policy_on=action[0],
                    curr_round=curr_round,
                    prev_eff=prev_eff,
                )

            result = gp_minimize(
                wrapped_objective,
                [Integer(0, 1, name="policy_on")],
                acq_func="EI",
                n_calls=BO_N_CALLS,
                random_state=BO_RANDOM_STATE,
            )

            z[row, col] = int(result.x[0])

    return z


def load_online_data():
    df = pd.read_csv(DATA_PATH)

    df_online = df[
        (df["policy"] == "online")
        & (df["Model Type"] == ONLINE_MODEL_TYPE)
        & (df["Dataset"].isin(ONLINE_DATASETS))
    ].copy()

    df_online = df_online[df_online["FL Round"] > 1]

    if "round_efficiency" not in df_online.columns:
        df_online["round_efficiency"] = (
            df_online["Val F1"] / df_online["Total Time of FL Round"]
        )

    # Preserve the grouping used by the original online visualization script.
    group_columns = ["replication", "Nhigh", "Nlow"]
    missing = [c for c in group_columns if c not in df_online.columns]
    if missing:
        raise KeyError(
            "The online dataset is missing grouping column(s): "
            + ", ".join(missing)
        )

    df_online["prev_eff"] = (
        df_online.groupby(group_columns)["round_efficiency"].shift(1)
    )

    return df_online.dropna(subset=["prev_eff"])


def build_online_surface(df_online):
    """
    Empirical projection of the contextual-bandit decisions, following the
    original generate_heatmaps.py script.
    """
    z = np.full((len(F1_OVER_TIME), len(ROUNDS)), np.nan, dtype=float)

    for row, eff in enumerate(F1_OVER_TIME):
        for col, curr_round in enumerate(ROUNDS):
            subset = df_online[
                (np.abs(df_online["FL Round"] - curr_round)
                 <= ONLINE_ROUND_WINDOW)
                & (np.abs(df_online["prev_eff"] - eff)
                   <= ONLINE_EFFICIENCY_WINDOW)
            ]

            if len(subset) > 0:
                z[row, col] = subset["client_selector"].mean()
            else:
                z[row, col] = ONLINE_EMPTY_CELL_STATE

    return (z >= 0.5).astype(int)


# ============================================================
# HEAT-MAP PLOTTING
# ============================================================

def heatmap_edges(values):
    """
    Convert regularly spaced cell centers into pcolormesh cell boundaries.
    """
    values = np.asarray(values, dtype=float)

    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])

    mids = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (values[1] - values[0]) / 2.0
    last = values[-1] + (values[-1] - values[-2]) / 2.0

    return np.concatenate([[first], mids, [last]])


def draw_heatmap(ax, z, title=None):
    # Two discrete states. Greyscale remains legible in print.
    cmap = plt.get_cmap("Greys", 2)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    x_edges = heatmap_edges(ROUNDS)
    y_edges = heatmap_edges(F1_OVER_TIME)

    ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap=cmap,
        norm=norm,
        shading="flat",
        linewidth=0.25,
        edgecolors="white",
    )

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Previous F1/Time")
    ax.set_xticks(ROUNDS[::2])
    ax.set_ylim(F1_OVER_TIME[0], F1_OVER_TIME[-1])

    y_ticks = np.linspace(F1_OVER_TIME[0], F1_OVER_TIME[-1], 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.4f}" for v in y_ticks])

    if title is not None:
        ax.set_title(title)


def save_individual_heatmap(z, filename):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    draw_heatmap(ax, z)

    # Explicit state labels without a continuous color bar.
    legend = [
        Patch(facecolor=plt.get_cmap("Greys", 2)(0), edgecolor="black",
              label="Pattern OFF"),
        Patch(facecolor=plt.get_cmap("Greys", 2)(1), edgecolor="black",
              label="Pattern ON"),
    ]
    ax.legend(handles=legend, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_combined_heatmap(surfaces, filename):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)

    panels = [
        ("(a) Fixed Policy (FLiP$_{rule}$)", surfaces["fixed"]),
        ("(b) Predictor-based (FLiP$_{pred}$)", surfaces["predictor"]),
        ("(c) Bayesian Optimization-based (FLiP$_{bo}$)", surfaces["bo"]),
        ("(d) Online Learning-based (FLiP$_{online}$)", surfaces["online"]),
    ]

    for ax, (title, z) in zip(axes.ravel(), panels):
        draw_heatmap(ax, z, title=title)

    legend = [
        Patch(facecolor=plt.get_cmap("Greys", 2)(0), edgecolor="black",
              label="Pattern OFF"),
        Patch(facecolor=plt.get_cmap("Greys", 2)(1), edgecolor="black",
              label="Pattern ON"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predictor_model_path = newest_matching_file(PREDICTOR_MODEL_PATTERNS)
    bo_model_path = first_existing_file(BO_MODEL_CANDIDATES)
    bo_scaler_path = first_existing_file(BO_SCALER_CANDIDATES)

    print("Predictor model:", predictor_model_path)
    print("BO model:", bo_model_path)
    print("BO scaler:", bo_scaler_path)
    print("Online data:", DATA_PATH)

    predictor_model = joblib.load(predictor_model_path)
    bo_model = joblib.load(bo_model_path)
    bo_scaler = joblib.load(bo_scaler_path)
    online_data = load_online_data()

    surfaces = {
        "fixed": build_fixed_surface(),
        "predictor": build_predictor_surface(predictor_model),
        "bo": build_bo_surface(bo_model, bo_scaler),
        "online": build_online_surface(online_data),
    }

    save_individual_heatmap(
        surfaces["fixed"],
        os.path.join(OUTPUT_DIR, "fixed_visualization_heatmap.pdf"),
    )
    save_individual_heatmap(
        surfaces["predictor"],
        os.path.join(OUTPUT_DIR, "predictor_visualization_heatmap.pdf"),
    )
    save_individual_heatmap(
        surfaces["bo"],
        os.path.join(OUTPUT_DIR, "bo_visualization_heatmap.pdf"),
    )
    save_individual_heatmap(
        surfaces["online"],
        os.path.join(OUTPUT_DIR, "online_visualization_heatmap.pdf"),
    )

    save_combined_heatmap(
        surfaces,
        os.path.join(OUTPUT_DIR, "policy_heatmaps.pdf"),
    )

    print("Saved heat maps to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
