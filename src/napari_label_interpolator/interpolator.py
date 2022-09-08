from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import edt
import numpy as np
from magicgui import magic_factory

if TYPE_CHECKING:
    import napari


def interpolate_sdf(labels_data, background=0, axis=0):
    """
    Interpolate labels along a given using weighted SDFs
    """
    # compute euclidean distance transform for each z_slice and each label
    dists = defaultdict(dict)
    data_reordered = np.moveaxis(labels_data, axis, 0)
    for z_idx, z_slice in enumerate(data_reordered):
        labels_in_slice = [lb for lb in np.unique(z_slice) if lb != background]
        if not labels_in_slice:
            continue
        for label in labels_in_slice:
            masked = np.ma.masked_not_equal(z_slice, label).filled(0)
            dists[z_idx][label] = edt.sdf(masked)

    if len(dists) < 2:
        # less than 2 slices, stop
        return

    result = data_reordered.copy()

    labeled_slices_ordered = list(dists)
    previous_labels = {}
    next_labels = {}
    for z_idx, z_slice in enumerate(result):
        # set up new interpolations if needed
        if z_idx in dists:
            # ensure we update but keep old ones missing from this slice
            previous_labels.update({k: z_idx for k in dists[z_idx].keys()})
            # ensure we have a next label for each previous
            for label in list(previous_labels):
                next_index = labeled_slices_ordered.index(z_idx) + 1
                for slice_index in labeled_slices_ordered[next_index:]:
                    # look into next slices for the same label
                    if label in dists[slice_index]:
                        next_labels[label] = slice_index
                        break
                else:
                    next_labels[label] = None

                if next_labels[label] is None:
                    previous_labels.pop(label)
                    next_labels.pop(label, None)

        # actually do the interpolation
        for label, prev_idx in previous_labels.items():
            next_idx = next_labels[label]
            prev_data = dists[prev_idx][label]
            next_data = dists[next_idx][label]

            k = (z_idx - prev_idx) / (next_idx - prev_idx)
            weighted_average = prev_data * (1 - k) + next_data * k

            result[z_idx][weighted_average > 0] = label

    return np.moveaxis(result, 0, axis)


@magic_factory
def interpolate(
    labels: napari.layers.Labels, axis: int = 0
) -> napari.types.LayerDataTuple:
    """
    Interpolate nd labels along the 0th dimension.
    """
    return (
        interpolate_sdf(
            labels.data, background=labels._background_label, axis=axis
        ),
        {"name": "{labels} - interpolated", "scale": labels.scale},
        "labels",
    )
