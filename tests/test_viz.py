import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

from neurolens.viz import plot_brain_surface, make_radar_chart


def test_plot_brain_surface_returns_figure():
    data = np.random.randn(20484)
    fig = plot_brain_surface(data, views=["left", "right"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_make_radar_chart_single():
    roi_data = {
        "Visual Cortex": 0.8,
        "Auditory Cortex": 0.3,
        "Language Areas": 0.6,
    }
    fig = make_radar_chart({"Stimulus A": roi_data})
    assert fig is not None


def test_make_radar_chart_comparison():
    data_a = {"Visual Cortex": 0.8, "Auditory Cortex": 0.3}
    data_b = {"Visual Cortex": 0.4, "Auditory Cortex": 0.7}
    fig = make_radar_chart({"A": data_a, "B": data_b})
    assert fig is not None
