from pathlib import Path
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from tn3k_dataset import TN3KDataset
from tn3k_model import TN3KResNet34UNet


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = (
    Path(__file__).resolve().parent.parent
)

BEST_CHECKPOINT = (
    THYROID_ROOT
    / "checkpoints"
    / "tn3k_v1_earlystop"
    / "tn3k_v1_earlystop_best.pth"
)

OUTPUT_DIR = (
    THYROID_ROOT
    / "checkpoints"
    / "tn3k_v1_earlystop"
    / "validation_diagnostics"
    / "threshold_sweep"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_threshold_sweep.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_threshold_sweep_summary.txt"
)

PLOT_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_threshold_sweep.png"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 512

BATCH_SIZE = 2

NUM_WORKERS = 0

EPSILON = 1e-6


# ------------------------------------------------------------
# We are NOT searching the official test set.
#
# These thresholds are evaluated ONLY on Fold-0 validation.
# ------------------------------------------------------------

THRESHOLDS = [

    0.300,
    0.325,
    0.350,
    0.375,
    0.400,
    0.425,
    0.450,
    0.475,

    0.500,

    0.525,
    0.550,
    0.575,
    0.600,
    0.625,
    0.650,
    0.675,
    0.700,
]


# ============================================================
# VALIDATION DATA
# ============================================================

def create_validation_loader():

    dataset = TN3KDataset(

        split="validation",

        image_size=IMAGE_SIZE,

        augmentation=None,
    )


    if len(dataset) != 576:

        raise RuntimeError(
            f"Expected 576 validation images, "
            f"found {len(dataset)}."
        )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=False,

        drop_last=False,
    )


    return dataset, loader


# ============================================================
# LOAD BEST MODEL
# ============================================================

def load_best_model(
    device,
):

    if not BEST_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Best checkpoint not found:\n"
            f"{BEST_CHECKPOINT}"
        )


    print()
    print("Loading best checkpoint:")

    print(
        BEST_CHECKPOINT
    )


    checkpoint = torch.load(

        BEST_CHECKPOINT,

        map_location=device,

        weights_only=True,
    )


    model = TN3KResNet34UNet(
        use_pretrained_encoder=False,
    )


    model.configure_v1_trainable_layers()


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        device
    )


    model.eval()


    print()
    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        )
    )


    print(
        "Saved best validation Dice:",
        f"{checkpoint.get('best_validation_dice', -1):.6f}"
    )


    return (
        model,
        checkpoint,
    )


# ============================================================
# CACHE VALIDATION PROBABILITIES
# ============================================================

@torch.no_grad()
def collect_probabilities(
    model,
    loader,
    device,
):
    """
    Run the neural network only once.

    We cache:
        probability maps
        ground-truth masks
        size-group labels

    The threshold sweep then reuses the same predictions.
    """

    probability_batches = []

    target_batches = []

    size_groups = []


    number_of_batches = len(
        loader
    )


    print()
    print("=" * 70)
    print("GENERATING VALIDATION PROBABILITY MAPS")
    print("=" * 70)


    for batch_index, batch in enumerate(
        loader
    ):

        images = (
            batch["image"]
            .to(device)
        )


        masks = (
            batch["mask"]
            >= 0.5
        )


        masks = masks.to(
            device
        )


        logits = model(
            images
        )


        probabilities = torch.sigmoid(
            logits
        )


        # ----------------------------------------------------
        # Store on CPU while processing batches.
        #
        # Keep float32 so threshold=0.50 reproduces the exact
        # validation metric as closely as possible.
        # ----------------------------------------------------

        probability_batches.append(

            probabilities
            .detach()
            .cpu()
        )


        target_batches.append(

            masks
            .detach()
            .cpu()
        )


        size_groups.extend(
            list(
                batch[
                    "nodule_size_group"
                ]
            )
        )


        if (
            (batch_index + 1) % 50 == 0
            or
            batch_index + 1
            == number_of_batches
        ):

            print(
                f"Processed batch "
                f"{batch_index + 1}"
                f"/{number_of_batches}"
            )


    probabilities = torch.cat(

        probability_batches,

        dim=0,
    )


    targets = torch.cat(

        target_batches,

        dim=0,
    )


    if probabilities.shape[0] != 576:

        raise RuntimeError(
            "Expected 576 probability maps."
        )


    if targets.shape[0] != 576:

        raise RuntimeError(
            "Expected 576 target masks."
        )


    print()
    print(
        "Cached probability maps:",
        tuple(
            probabilities.shape
        )
    )


    print(
        "Cached target masks:",
        tuple(
            targets.shape
        )
    )


    print()
    print(
        "Moving cached tensors to GPU "
        "for fast threshold evaluation..."
    )


    probabilities = probabilities.to(
        device
    )


    targets = targets.to(
        device
    )


    return (
        probabilities,
        targets,
        size_groups,
    )


# ============================================================
# METRICS AT ONE THRESHOLD
# ============================================================

@torch.no_grad()
def evaluate_threshold(
    probabilities,
    targets,
    threshold,
):

    predictions = (
        probabilities
        >= threshold
    )


    # --------------------------------------------------------
    # Per-image TP / FP / FN
    # --------------------------------------------------------

    true_positive = (

        predictions
        & targets

    ).sum(
        dim=(1, 2, 3)
    ).float()


    false_positive = (

        predictions
        & (~targets)

    ).sum(
        dim=(1, 2, 3)
    ).float()


    false_negative = (

        (~predictions)
        & targets

    ).sum(
        dim=(1, 2, 3)
    ).float()


    predicted_pixels = (

        predictions

    ).sum(
        dim=(1, 2, 3)
    )


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice = (

        2.0 * true_positive
        + EPSILON

    ) / (

        2.0 * true_positive
        + false_positive
        + false_negative
        + EPSILON
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = (

        true_positive
        + EPSILON

    ) / (

        true_positive
        + false_positive
        + false_negative
        + EPSILON
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (

        true_positive
        + EPSILON

    ) / (

        true_positive
        + false_positive
        + EPSILON
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (

        true_positive
        + EPSILON

    ) / (

        true_positive
        + false_negative
        + EPSILON
    )


    # --------------------------------------------------------
    # Empty predicted masks
    # --------------------------------------------------------

    empty_predictions = (
        predicted_pixels == 0
    )


    return {

        "threshold":
            float(
                threshold
            ),

        "mean_dice":
            float(
                dice.mean().item()
            ),

        "median_dice":
            float(
                dice.median().item()
            ),

        "mean_iou":
            float(
                iou.mean().item()
            ),

        "mean_precision":
            float(
                precision.mean().item()
            ),

        "mean_recall":
            float(
                recall.mean().item()
            ),

        "dice_below_0_50":
            int(
                (dice < 0.50)
                .sum()
                .item()
            ),

        "dice_at_least_0_80":
            int(
                (dice >= 0.80)
                .sum()
                .item()
            ),

        "dice_at_least_0_90":
            int(
                (dice >= 0.90)
                .sum()
                .item()
            ),

        "empty_predictions":
            int(
                empty_predictions
                .sum()
                .item()
            ),

        # Keep per-image values temporarily for subgroup
        # analysis of the selected threshold.
        "_dice_tensor":
            dice,

        "_iou_tensor":
            iou,

        "_precision_tensor":
            precision,

        "_recall_tensor":
            recall,
    }


# ============================================================
# SIZE-GROUP METRICS FOR SELECTED THRESHOLD
# ============================================================

def calculate_size_group_results(
    threshold_result,
    size_groups,
):

    results = {}


    dice = (
        threshold_result[
            "_dice_tensor"
        ]
    )

    iou = (
        threshold_result[
            "_iou_tensor"
        ]
    )

    precision = (
        threshold_result[
            "_precision_tensor"
        ]
    )

    recall = (
        threshold_result[
            "_recall_tensor"
        ]
    )


    for group in [

        "tiny",
        "small",
        "medium",
        "large",

    ]:

        indices = [

            index

            for index, value
            in enumerate(
                size_groups
            )

            if value == group
        ]


        if not indices:

            continue


        index_tensor = torch.tensor(

            indices,

            device=dice.device,

            dtype=torch.long,
        )


        group_dice = dice[
            index_tensor
        ]


        group_iou = iou[
            index_tensor
        ]


        group_precision = precision[
            index_tensor
        ]


        group_recall = recall[
            index_tensor
        ]


        results[group] = {

            "count":
                len(indices),

            "mean_dice":
                float(
                    group_dice
                    .mean()
                    .item()
                ),

            "mean_iou":
                float(
                    group_iou
                    .mean()
                    .item()
                ),

            "mean_precision":
                float(
                    group_precision
                    .mean()
                    .item()
                ),

            "mean_recall":
                float(
                    group_recall
                    .mean()
                    .item()
                ),
        }


    return results


# ============================================================
# REMOVE TEMPORARY TENSORS
# ============================================================

def public_result(
    result,
):

    return {

        key: value

        for key, value
        in result.items()

        if not key.startswith(
            "_"
        )
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    results,
):

    clean_results = [

        public_result(
            result
        )

        for result in results
    ]


    fieldnames = [

        "threshold",

        "mean_dice",
        "median_dice",

        "mean_iou",

        "mean_precision",
        "mean_recall",

        "dice_below_0_50",

        "dice_at_least_0_80",
        "dice_at_least_0_90",

        "empty_predictions",
    ]


    with open(

        CSV_PATH,

        "w",

        newline="",

        encoding="utf-8",

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames,
        )


        writer.writeheader()


        writer.writerows(
            clean_results
        )


    print()
    print(
        "Saved CSV:"
    )

    print(
        CSV_PATH
    )


# ============================================================
# SAVE PLOT
# ============================================================

def save_plot(
    results,
):

    thresholds = [

        result[
            "threshold"
        ]

        for result in results
    ]


    mean_dice = [

        result[
            "mean_dice"
        ]

        for result in results
    ]


    mean_iou = [

        result[
            "mean_iou"
        ]

        for result in results
    ]


    precision = [

        result[
            "mean_precision"
        ]

        for result in results
    ]


    recall = [

        result[
            "mean_recall"
        ]

        for result in results
    ]


    figure = plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        thresholds,
        mean_dice,
        marker="o",
        label="Mean Dice",
    )


    plt.plot(
        thresholds,
        mean_iou,
        marker="o",
        label="Mean IoU",
    )


    plt.plot(
        thresholds,
        precision,
        marker="o",
        label="Precision",
    )


    plt.plot(
        thresholds,
        recall,
        marker="o",
        label="Recall",
    )


    plt.axvline(
        0.50,
        linestyle="--",
        label="Original threshold 0.50",
    )


    plt.xlabel(
        "Prediction Threshold"
    )


    plt.ylabel(
        "Metric"
    )


    plt.title(
        "TN3K V1 Validation Threshold Sweep"
    )


    plt.legend()


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    figure.savefig(

        PLOT_PATH,

        dpi=160,

        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    print()
    print(
        "Saved plot:"
    )

    print(
        PLOT_PATH
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    checkpoint,
    results,
    best_result,
    baseline_result,
    size_results,
):

    difference = (

        best_result[
            "mean_dice"
        ]

        -

        baseline_result[
            "mean_dice"
        ]
    )


    lines = []


    lines.append(
        "TN3K V1 VALIDATION THRESHOLD SWEEP"
    )

    lines.append(
        "=" * 70
    )


    lines.append("")


    lines.append(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )


    lines.append(
        f"Saved checkpoint validation Dice: "
        f"{checkpoint.get('best_validation_dice', -1):.6f}"
    )


    lines.append(
        "Official TN3K test set used: NO"
    )


    lines.append("")
    lines.append("")


    # ========================================================
    # FULL TABLE
    # ========================================================

    lines.append(
        "THRESHOLD RESULTS"
    )

    lines.append(
        "-" * 70
    )


    lines.append(

        "Threshold   "
        "Dice       "
        "IoU        "
        "Precision  "
        "Recall     "
        "Dice<0.50"
    )


    for result in results:

        lines.append(

            f"{result['threshold']:<11.3f}"

            f"{result['mean_dice']:<11.6f}"

            f"{result['mean_iou']:<11.6f}"

            f"{result['mean_precision']:<11.6f}"

            f"{result['mean_recall']:<11.6f}"

            f"{result['dice_below_0_50']}"
        )


    # ========================================================
    # BASELINE
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "ORIGINAL THRESHOLD"
    )

    lines.append(
        "-" * 70
    )


    lines.append(
        f"Threshold: "
        f"{baseline_result['threshold']:.3f}"
    )


    lines.append(
        f"Mean Dice: "
        f"{baseline_result['mean_dice']:.6f}"
    )


    lines.append(
        f"Mean IoU: "
        f"{baseline_result['mean_iou']:.6f}"
    )


    lines.append(
        f"Precision: "
        f"{baseline_result['mean_precision']:.6f}"
    )


    lines.append(
        f"Recall: "
        f"{baseline_result['mean_recall']:.6f}"
    )


    # ========================================================
    # BEST
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "BEST VALIDATION THRESHOLD"
    )

    lines.append(
        "-" * 70
    )


    lines.append(
        f"Threshold: "
        f"{best_result['threshold']:.3f}"
    )


    lines.append(
        f"Mean Dice: "
        f"{best_result['mean_dice']:.6f}"
    )


    lines.append(
        f"Median Dice: "
        f"{best_result['median_dice']:.6f}"
    )


    lines.append(
        f"Mean IoU: "
        f"{best_result['mean_iou']:.6f}"
    )


    lines.append(
        f"Precision: "
        f"{best_result['mean_precision']:.6f}"
    )


    lines.append(
        f"Recall: "
        f"{best_result['mean_recall']:.6f}"
    )


    lines.append(
        f"Dice < 0.50 cases: "
        f"{best_result['dice_below_0_50']}"
    )


    lines.append(
        f"Dice >= 0.80 cases: "
        f"{best_result['dice_at_least_0_80']}"
    )


    lines.append(
        f"Dice >= 0.90 cases: "
        f"{best_result['dice_at_least_0_90']}"
    )


    lines.append(
        f"Empty predictions: "
        f"{best_result['empty_predictions']}"
    )


    lines.append("")
    lines.append(

        f"Dice improvement over threshold 0.50: "
        f"{difference:+.6f}"
    )


    # ========================================================
    # SIZE GROUPS
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "BEST-THRESHOLD RESULTS BY NODULE SIZE"
    )

    lines.append(
        "-" * 70
    )


    for group in [

        "tiny",
        "small",
        "medium",
        "large",

    ]:

        if group not in size_results:

            continue


        values = size_results[
            group
        ]


        lines.append("")


        lines.append(
            group.upper()
        )


        lines.append(
            f"  Images:     "
            f"{values['count']}"
        )


        lines.append(
            f"  Mean Dice:  "
            f"{values['mean_dice']:.6f}"
        )


        lines.append(
            f"  Mean IoU:   "
            f"{values['mean_iou']:.6f}"
        )


        lines.append(
            f"  Precision:  "
            f"{values['mean_precision']:.6f}"
        )


        lines.append(
            f"  Recall:     "
            f"{values['mean_recall']:.6f}"
        )


    summary_text = "\n".join(
        lines
    )


    SUMMARY_PATH.write_text(

        summary_text,

        encoding="utf-8",
    )


    print()
    print(summary_text)


    print()
    print(
        "Saved summary:"
    )

    print(
        SUMMARY_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TN3K V1 EARLY-STOPPING "
        "VALIDATION THRESHOLD SWEEP"
    )

    print("=" * 70)


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This script uses ONLY the "
        "576 Fold-0 validation images."
    )

    print(
        "Official TN3K test set remains untouched."
    )


    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"
    )


    print()
    print(
        "Device:",
        device
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # DATA
    # ========================================================

    dataset, loader = (
        create_validation_loader()
    )


    print()
    print(
        "Validation images:",
        len(dataset)
    )


    # ========================================================
    # MODEL
    # ========================================================

    model, checkpoint = (
        load_best_model(
            device
        )
    )


    # ========================================================
    # PROBABILITY MAPS
    # ========================================================

    (
        probabilities,
        targets,
        size_groups,
    ) = collect_probabilities(

        model=model,

        loader=loader,

        device=device,
    )


    # ========================================================
    # SWEEP
    # ========================================================

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)


    results = []


    for threshold in THRESHOLDS:

        result = evaluate_threshold(

            probabilities=probabilities,

            targets=targets,

            threshold=threshold,
        )


        results.append(
            result
        )


        print(

            f"Threshold {threshold:.3f} | "

            f"Dice "
            f"{result['mean_dice']:.6f} | "

            f"IoU "
            f"{result['mean_iou']:.6f} | "

            f"Precision "
            f"{result['mean_precision']:.6f} | "

            f"Recall "
            f"{result['mean_recall']:.6f}"
        )


    # ========================================================
    # BASELINE = 0.50
    # ========================================================

    baseline_result = next(

        result

        for result in results

        if abs(
            result["threshold"]
            - 0.50
        ) < 1e-9
    )


    # ========================================================
    # BEST THRESHOLD BY MEAN VALIDATION DICE
    # ========================================================

    best_result = max(

        results,

        key=lambda result:
            result[
                "mean_dice"
            ],
    )


    # ========================================================
    # SUBGROUP ANALYSIS FOR BEST THRESHOLD
    # ========================================================

    size_results = (
        calculate_size_group_results(

            threshold_result=best_result,

            size_groups=size_groups,
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_csv(
        results
    )


    save_plot(
        results
    )


    save_summary(

        checkpoint=checkpoint,

        results=results,

        best_result=best_result,

        baseline_result=baseline_result,

        size_results=size_results,
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VALIDATION THRESHOLD SWEEP COMPLETE"
    )
    print("=" * 70)


    print()
    print(
        "Original threshold:",
        f"{baseline_result['threshold']:.3f}"
    )


    print(
        "Original mean Dice:",
        f"{baseline_result['mean_dice']:.6f}"
    )


    print()
    print(
        "Best validation threshold:",
        f"{best_result['threshold']:.3f}"
    )


    print(
        "Best mean Dice:",
        f"{best_result['mean_dice']:.6f}"
    )


    print()
    print(
        "Official TN3K test set was NOT used."
    )


    print()
    print(
        "DO NOT evaluate the test set "
        "with multiple thresholds."
    )


    print(
        "Choose and lock ONE threshold "
        "from validation first."
    )


if __name__ == "__main__":

    main()