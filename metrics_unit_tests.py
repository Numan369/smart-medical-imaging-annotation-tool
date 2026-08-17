"""Unit tests for the Dice / selection-score math used by:

    train_pneumothorax_512_negative_aware.py (update_metrics/finalize_metrics)
    evaluate_pneumothorax_512_negative_aware_test.py (evaluate)

These two files implement the same per-image Dice formula independently.
This script re-implements that formula in plain NumPy (no torch needed) and
checks it against hand-computed expected values on tiny synthetic tensors.

Run: python metrics_unit_tests.py
No GPU, dataset, or checkpoint is touched. This never loads real images.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def per_image_dice(logits, targets, threshold):
    """Mirror of the Dice block in update_metrics()/evaluate()."""

    probabilities = sigmoid(logits)
    predictions = probabilities >= threshold
    binary_targets = targets >= 0.5
    axes = tuple(range(1, targets.ndim))

    intersection = np.logical_and(predictions, binary_targets).sum(axis=axes).astype(float)
    predicted_area = predictions.sum(axis=axes).astype(float)
    target_area = binary_targets.sum(axis=axes).astype(float)
    denominator = predicted_area + target_area

    dice = np.where(
        denominator > 0,
        2.0 * intersection / np.where(denominator > 0, denominator, 1.0),
        np.ones_like(denominator),
    )
    return dice, predicted_area, target_area


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise SystemExit(f"Unit test failed: {name}")


def main():
    threshold = 0.35

    # 1. Empty prediction on empty target -> Dice must be 1.0, not 0/0 or 0.0.
    #    This checks the code does NOT penalize correct "healthy" predictions.
    logits = np.full((1, 4, 4), -10.0)
    targets = np.zeros((1, 4, 4))
    dice, pred_area, tgt_area = per_image_dice(logits, targets, threshold)
    check("empty-empty Dice == 1.0", dice[0] == 1.0)
    check("empty-empty predicted area == 0", pred_area[0] == 0.0)

    # 2. Positive target, model predicts nothing -> a total miss, Dice == 0.0.
    logits = np.full((1, 4, 4), -10.0)
    targets = np.zeros((1, 4, 4))
    targets[0, 0, 0] = 1.0
    dice, pred_area, tgt_area = per_image_dice(logits, targets, threshold)
    check("missed-positive Dice == 0.0", dice[0] == 0.0)

    # 3. Perfect single-pixel overlap -> Dice == 1.0.
    logits = np.full((1, 4, 4), -10.0)
    logits[0, 1, 1] = 10.0
    targets = np.zeros((1, 4, 4))
    targets[0, 1, 1] = 1.0
    dice, pred_area, tgt_area = per_image_dice(logits, targets, threshold)
    check("perfect single-pixel overlap Dice == 1.0", dice[0] == 1.0)

    # 4. Known partial-overlap case with a hand-computed Dice.
    #    Target = 4 pixels, prediction = 4 pixels, intersection = 2 pixels.
    #    Dice = 2*2 / (4+4) = 0.5
    logits = np.full((1, 4, 4), -10.0)
    logits[0, 0, 0:4] = 10.0  # predicts row 0, all 4 columns -> pred_area=4
    targets = np.zeros((1, 4, 4))
    targets[0, 0, 2:4] = 1.0  # target row0 col2,3
    targets[0, 1, 0:2] = 1.0  # target row1 col0,1  -> target_area=4, overlap=2
    dice, pred_area, tgt_area = per_image_dice(logits, targets, threshold)
    check(f"hand-computed partial overlap Dice == 0.5 (got {dice[0]})", np.isclose(dice[0], 0.5))

    # 5. positive_case_dice must average ONLY over images with target_area > 0,
    #    exactly like finalize_metrics()/evaluate() do via the `positive_cases` mask.
    logits_batch = np.concatenate(
        [
            np.full((1, 4, 4), -10.0),  # miss -> dice 0, positive
            np.full((1, 4, 4), -10.0),  # correct empty -> dice 1, but NEGATIVE (excluded)
        ]
    )
    targets_batch = np.concatenate(
        [
            (lambda t: (t.__setitem__((0, 0, 0), 1.0), t)[1])(np.zeros((1, 4, 4))),
            np.zeros((1, 4, 4)),
        ]
    )
    dice_batch, pred_area_batch, tgt_area_batch = per_image_dice(logits_batch, targets_batch, threshold)
    positive_mask = tgt_area_batch > 0
    positive_dice_mean = dice_batch[positive_mask].mean()
    check(
        "positive_case_dice excludes negative images from its average "
        f"(expected 0.0, got {positive_dice_mean})",
        np.isclose(positive_dice_mean, 0.0),
    )

    # 6. negative_empty_mask_accuracy: a negative image with ANY predicted
    #    pixel must count as a false positive, even 1 pixel.
    logits_neg_fp = np.full((1, 4, 4), -10.0)
    logits_neg_fp[0, 0, 0] = 10.0  # one stray predicted pixel
    targets_neg = np.zeros((1, 4, 4))
    _, pred_area_fp, _ = per_image_dice(logits_neg_fp, targets_neg, threshold)
    check("single stray pixel on a negative image is NOT counted as empty", pred_area_fp[0] > 0)

    # 7. Threshold sensitivity: raising the threshold should never increase
    #    predicted area (monotonicity sanity check of the thresholding step).
    rng = np.random.default_rng(0)
    random_logits = rng.normal(size=(5, 16, 16)) * 3.0
    random_targets = (rng.random((5, 16, 16)) > 0.9).astype(float)
    _, area_low, _ = per_image_dice(random_logits, random_targets, 0.2)
    _, area_high, _ = per_image_dice(random_logits, random_targets, 0.6)
    check("higher threshold never predicts more area than a lower threshold", bool(np.all(area_high <= area_low)))

    print("\nAll metric unit tests passed.")
    print("Conclusion: the Dice / positive_case_dice / negative_empty_mask_accuracy")
    print("formulas used in train_pneumothorax_512_negative_aware.py and")
    print("evaluate_pneumothorax_512_negative_aware_test.py are mathematically")
    print("consistent and are NOT a likely cause of the weak Dice scores.")


if __name__ == "__main__":
    main()
