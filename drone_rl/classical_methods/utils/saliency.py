from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow

_ACTION_VECTORS   = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # must match drone.actions
_OUTPUT_DIR = "saliency_output"


def _ensure_output_dir() -> str:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR


def _get_Q_values(drone, state: tuple) -> np.ndarray:
    """
    Return an action-value vector for *state*.

    Q-based agents expose ``drone.Q`` directly. Value-based agents expose
    ``drone.V`` instead, so we derive per-action preferences from successor
    state values.
    """
    if hasattr(drone, "Q"):
        return drone.Q.get(state, np.zeros(len(drone.actions))).copy()

    if hasattr(drone, "V"):
        values = np.zeros(len(drone.actions), dtype=float)
        for idx, action in enumerate(drone.actions):
            next_state = _infer_next_state(drone, state, action)
            values[idx] = drone.V.get(next_state, 0.0)
        return values

    return np.zeros(len(drone.actions), dtype=float)


def _infer_next_state(drone, state: tuple, action: tuple[int, int]) -> tuple:
    """Best-effort next-state transition for value-based agents."""
    if hasattr(drone, "step"):
        next_state = drone.step(state, action)
    else:
        r, c, has_pkg = state
        dr, dc = action
        nr, nc = r + dr, c + dc
        lane_coords = set(drone.lane_coords)
        if (nr, nc) not in lane_coords:
            nr, nc = r, c
        next_state = (nr, nc, has_pkg)

    if len(next_state) == 3 and not next_state[2] and next_state[:2] == drone.package_pos:
        next_state = (next_state[0], next_state[1], True)
    return next_state


# Visitation Heatmap (environment-specific visualization)
def plot_visitation_heatmap(
    drone,
    path: List[tuple],
    bw_map: np.ndarray,
    title: str = "Cell Visitation Heatmap",
    save: bool = True,
) -> plt.Figure:
    """
    Overlays visit-frequency counts on the grid map using a logarithmic
    colour scale (log1p) so rarely-visited cells remain distinguishable.

    Parameters
    ----------
    drone   : QLearningDrone – used for package/delivery positions.
    path    : list of (row, col, has_pkg) states from policy rollout.
    bw_map  : 2-D binary numpy array (1 = lane, 0 = wall).
    title   : figure title.
    save    : if True, saves PNG to saliency_output/.
    """
    heat = np.zeros_like(bw_map, dtype=float)
    for r, c, _ in path:
        heat[r, c] += 1

    # Mask walls so they render as a dark neutral
    masked_heat = np.where(bw_map == 1, np.log1p(heat), np.nan)

    fig, ax = plt.subplots(figsize=(7, 7))
    # Wall background
    ax.imshow(bw_map, cmap="Greys", vmin=0, vmax=2, alpha=0.55)
    # Heatmap overlay
    im = ax.imshow(masked_heat, cmap="inferno", alpha=0.85,
                   vmin=0, vmax=np.nanmax(masked_heat) or 1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log₁₊(visit count)", fontsize=10)

    # Landmarks
    pr, pc = drone.package_pos
    dr, dc = drone.delivery_pos
    ax.plot(pc, pr, marker="s", color="yellow",   markersize=11, label="Package",  zorder=5)
    ax.plot(dc, dr, marker="x", color="limegreen", markersize=13, mew=2.5, label="Delivery", zorder=5)
    # Start
    sr, sc, _ = path[0]
    ax.plot(sc, sr, marker="^", color="cyan", markersize=10, label="Start", zorder=5)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save:
        path_out = os.path.join(_ensure_output_dir(), "visitation_heatmap.png")
        fig.savefig(path_out, dpi=150)
        print(f"[saliency] Saved → {path_out}")
    return fig


# 3. Action Preference Map 
def plot_action_preference_map(
    drone,
    bw_map: np.ndarray,
    has_pkg: bool = False,
    title: str | None = None,
    save: bool = True,
) -> plt.Figure:
    """
    For every passable cell, looks up the greedy action and draws an arrow.
    Arrow colour encodes Q-value confidence (max Q − second-max Q).

    Parameters
    ----------
    drone   : QLearningDrone – trained agent.
    bw_map  : 2-D binary numpy array.
    has_pkg : which has_package slice to visualise (False = pre-pickup).
    title   : figure title (auto-generated if None).
    save    : if True, saves PNG to saliency_output/.
    """
    if title is None:
        pkg_str = "carrying package" if has_pkg else "searching for package"
        title = f"Greedy Action Map ({pkg_str})"

    rows, cols = bw_map.shape
    lane_set = set(map(tuple, np.argwhere(bw_map == 1)))

    U = np.zeros((rows, cols))
    V = np.zeros((rows, cols))
    confidence = np.zeros((rows, cols))

    for r, c in lane_set:
        state = (r, c, has_pkg)
        q = _get_Q_values(drone, state)
        best = int(np.argmax(q))
        dr, dc = _ACTION_VECTORS[best]
        U[r, c] = dc    # quiver: U = x-component = col-direction
        V[r, c] = -dr   # quiver: V = y-component; flip row→y

        sorted_q = np.sort(q)[::-1]
        confidence[r, c] = float(sorted_q[0] - sorted_q[1]) if sorted_q[0] != sorted_q[1] else 0.0

    # Mask cells with no Q data (all-zero Q = unseen state)
    known_mask = np.zeros((rows, cols), dtype=bool)
    for r, c in lane_set:
        state = (r, c, has_pkg)
        if hasattr(drone, "Q") and state in drone.Q:
            known_mask[r, c] = True
        elif hasattr(drone, "V") and state in drone.V:
            known_mask[r, c] = True

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bw_map, cmap="gray", vmin=0, vmax=1.4)

    # Landmark overlays
    pr, pc = drone.package_pos
    dr_pos, dc_pos = drone.delivery_pos
    ax.plot(pc, pr,     marker="s", color="yellow",    markersize=11, zorder=6, label="Package")
    ax.plot(dc_pos, dr_pos, marker="x", color="limegreen", markersize=13, mew=2.5, zorder=6, label="Delivery")

    # Normalise confidence for colour mapping
    conf_vals = confidence[known_mask]
    norm = mcolors.Normalize(vmin=0, vmax=conf_vals.max() if conf_vals.size else 1)
    cmap = plt.cm.plasma

    for r, c in lane_set:
        if not known_mask[r, c]:
            continue
        col_rgba = cmap(norm(confidence[r, c]))
        ax.annotate(
            "",
            xy=(c + U[r, c] * 0.38, r - V[r, c] * 0.38),   # arrowhead (col, row axes)
            xytext=(c - U[r, c] * 0.18, r + V[r, c] * 0.18),
            arrowprops=dict(arrowstyle="-|>", color=col_rgba,
                            lw=1.5, mutation_scale=12),
            zorder=4,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Action confidence (ΔQ)", fontsize=10)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save:
        tag = "with_pkg" if has_pkg else "no_pkg"
        path_out = os.path.join(_ensure_output_dir(), f"action_preference_{tag}.png")
        fig.savefig(path_out, dpi=150)
        print(f"[saliency] Saved → {path_out}")
    return fig



def run_saliency_suite(
    drone,
    path: List[tuple],
    bw_map: np.ndarray,
    show: bool = True,
) -> None:
    """
    Render all three saliency/visualisation plots and optionally display them.

    Parameters
    ----------
    drone   : trained QLearningDrone.
    path    : policy rollout as list of (row, col, has_pkg) tuples.
    bw_map  : binary grid map.
    show    : call plt.show() after rendering.
    """
    print("[saliency] Running saliency suite…")

    states = path  

    fig2 = plot_visitation_heatmap(drone, path, bw_map)
    fig3a = plot_action_preference_map(drone, bw_map, has_pkg=False)
    fig3b = plot_action_preference_map(drone, bw_map, has_pkg=True)

    print("[saliency] Suite complete. Outputs in ./saliency_output/")

    if show:
        plt.show()
