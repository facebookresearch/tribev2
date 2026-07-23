"""Shared visualization helpers: brain plots and radar charts."""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_brain_surface(
    data: np.ndarray,
    views: list[str] | None = None,
    cmap: str = "hot",
    title: str | None = None,
    colorbar: bool = True,
) -> matplotlib.figure.Figure:
    """Plot brain activation on a cortical surface using nilearn.

    Parameters
    ----------
    data : np.ndarray
        1D array of shape (n_vertices,) on fsaverage5 (20484).
    views : list of str
        View angles, e.g. ["left", "right"]. Defaults to ["left", "right"].
    cmap : str
        Matplotlib colormap name.
    title : str or None
        Optional figure title.
    colorbar : bool
        Whether to show a colorbar.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from tribev2.plotting.cortical import PlotBrainNilearn

    if views is None:
        views = ["left", "right"]

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    fig, axarr = plotter.get_fig_axes(views)
    plotter.plot_surf(
        data,
        views=views,
        axes=axarr,
        cmap=cmap,
        colorbar=colorbar,
    )
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig


def make_radar_chart(
    datasets: dict[str, dict[str, float]],
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Create a radar/spider chart comparing ROI activation profiles.

    Parameters
    ----------
    datasets : dict
        Maps label -> {roi_name: value}. All dicts must have the same keys.
    title : str or None
        Optional chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels = list(next(iter(datasets.values())).keys())
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

    for (name, values), color in zip(datasets.items(), colors):
        vals = [values[label] for label in labels]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    return fig
