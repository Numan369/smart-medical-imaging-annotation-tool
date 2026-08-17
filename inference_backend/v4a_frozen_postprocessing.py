"""Frozen post-processing for the V4A pneumothorax model.

This configuration was selected using validation data only.
Do not change these constants after test evaluation.
"""

import numpy as np
from scipy.ndimage import label as connected_component_label


IMAGE_SIZE = 512
PROBABILITY_THRESHOLD = 0.15
MINIMUM_COMPONENT_PIXELS = 112

CONNECTIVITY_STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8,
)


def apply_v4a_postprocessing(probability_map):
    """Convert one 512×512 probability map into a filtered mask.

    Returns:
        mask: uint8 array containing values 0 and 1.
        region_count: number of retained connected regions.
    """

    probability_map = np.asarray(probability_map)

    expected_shape = (IMAGE_SIZE, IMAGE_SIZE)

    if probability_map.shape != expected_shape:
        raise ValueError(
            f"Expected probability-map shape {expected_shape}, "
            f"received {probability_map.shape}."
        )

    if not np.isfinite(probability_map).all():
        raise ValueError(
            "Probability map contains a non-finite value."
        )

    binary_mask = (
        probability_map >= PROBABILITY_THRESHOLD
    )

    if not binary_mask.any():
        return (
            np.zeros(expected_shape, dtype=np.uint8),
            0,
        )

    component_map, component_count = (
        connected_component_label(
            binary_mask,
            structure=CONNECTIVITY_STRUCTURE,
        )
    )

    component_sizes = np.bincount(
        component_map.ravel(),
        minlength=component_count + 1,
    )

    component_sizes[0] = 0

    retained_components = (
        component_sizes
        >= MINIMUM_COMPONENT_PIXELS
    )
    retained_components[0] = False

    filtered_mask = retained_components[
        component_map
    ].astype(np.uint8)

    region_count = int(
        retained_components.sum()
    )

    return filtered_mask, region_count
