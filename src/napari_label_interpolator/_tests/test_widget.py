import numpy as np

from napari_label_interpolator.interpolator import interpolate


def test_interpolator(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_labels(np.random.random((100, 100)))
    viewer.window.add_widget(interpolate)
