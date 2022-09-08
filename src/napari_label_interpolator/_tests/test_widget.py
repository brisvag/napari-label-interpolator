import numpy as np

from napari_label_interpolator.interpolator import interpolate


def test_interpolator(make_napari_viewer):
    viewer = make_napari_viewer()
    labels = np.ones((3, 3), dtype=int)
    labels[1] = 0
    viewer.add_labels(labels)
    w = interpolate()
    viewer.window.add_dock_widget(w)
    w()
    assert np.all(viewer.layers[-1].data == 1)
