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
    / "official_test_evaluation"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_official_test_per_image_metrics.csv"
)


SUMMARY_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_official_test_summary.txt"
)


COMPLETION_MARKER = (
    OUTPUT_DIR
    / "OFFICIAL_TEST_EVALUATION_COMPLETE.txt"
)


# ============================================================
# LOCKED FINAL CONFIGURATION
# ============================================================

IMAGE_SIZE = 512

BATCH_SIZE = 2

NUM_WORKERS = 0


# ------------------------------------------------------------
# IMPORTANT:
#
# This threshold was selected using ONLY the 576-image
# Fold-0 validation set.
#
# It must NOT be tuned using official test results.
# ------------------------------------------------------------

LOCKED_PREDICTION_THRESHOLD = 0.50


EXPECTED_CHECKPOINT_EPOCH = 15

EXPECTED_VALIDATION_DICE = 0.808592

EXPECTED_TEST_IMAGES = 614


EPSILON = 1e-6

NUMBER_OF_WORST_EXAMPLES = 6

NUMBER_OF_BEST_EXAMPLES = 6


# ============================================================
# TEST DATA
# ============================================================

def create_test_loader():

    print()
    print(
        "Loading official TN3K test dataset..."
    )


    dataset = TN3KDataset(

        split="test",

        image_size=IMAGE_SIZE,

        augmentation=None,
    )


    if len(dataset) != EXPECTED_TEST_IMAGES:

        raise RuntimeError(
            f"Official test count mismatch.\n"
            f"Expected: {EXPECTED_TEST_IMAGES}\n"
            f"Found:    {len(dataset)}"
        )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=False,

        drop_last=False,
    )


    return (
        dataset,
        loader,
    )


# ============================================================
# LOAD LOCKED MODEL
# ============================================================

def load_locked_model(
    device,
):

    if not BEST_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Locked best checkpoint not found:\n"
            f"{BEST_CHECKPOINT}"
        )


    print()
    print(
        "Loading locked checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )


    checkpoint = torch.load(

        BEST_CHECKPOINT,

        map_location=device,

        weights_only=True,
    )


    checkpoint_epoch = int(
        checkpoint.get(
            "epoch",
            -1,
        )
    )


    checkpoint_validation_dice = float(
        checkpoint.get(
            "best_validation_dice",
            -1.0,
        )
    )


    # ========================================================
    # SAFETY: VERIFY THE EXACT MODEL WE LOCKED
    # ========================================================

    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:

        raise RuntimeError(
            "Wrong checkpoint epoch.\n"
            f"Expected epoch: "
            f"{EXPECTED_CHECKPOINT_EPOCH}\n"
            f"Found epoch: "
            f"{checkpoint_epoch}"
        )


    if abs(
        checkpoint_validation_dice
        - EXPECTED_VALIDATION_DICE
    ) > 1e-5:

        raise RuntimeError(
            "Checkpoint validation Dice does not match "
            "the locked development result.\n"
            f"Expected: {EXPECTED_VALIDATION_DICE:.6f}\n"
            f"Found:    {checkpoint_validation_dice:.6f}"
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
        f"Locked checkpoint epoch: "
        f"{checkpoint_epoch}"
    )


    print(
        f"Locked validation Dice: "
        f"{checkpoint_validation_dice:.6f}"
    )


    print(
        f"Locked prediction threshold: "
        f"{LOCKED_PREDICTION_THRESHOLD:.2f}"
    )


    return (
        model,
        checkpoint,
    )


# ============================================================
# PER-IMAGE METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
):

    prediction = (
        prediction >= 0.5
    ).float()


    target = (
        target >= 0.5
    ).float()


    true_positive = (
        prediction
        * target
    ).sum().item()


    false_positive = (
        prediction
        * (1.0 - target)
    ).sum().item()


    false_negative = (
        (1.0 - prediction)
        * target
    ).sum().item()


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

    precision_denominator = (
        true_positive
        + false_positive
    )


    if precision_denominator > 0:

        precision = (
            true_positive
            / precision_denominator
        )

    else:

        precision = 0.0


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        + false_negative
    )


    if recall_denominator > 0:

        recall = (
            true_positive
            / recall_denominator
        )

    else:

        recall = 0.0


    return {

        "dice":
            float(dice),

        "iou":
            float(iou),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "true_positive":
            int(true_positive),

        "false_positive":
            int(false_positive),

        "false_negative":
            int(false_negative),
    }


# ============================================================
# OFFICIAL TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_official_test(
    model,
    loader,
    device,
):

    results = []


    number_of_batches = len(
        loader
    )


    print()
    print("=" * 70)

    print(
        "OFFICIAL TN3K TEST EVALUATION"
    )

    print("=" * 70)


    print()
    print(
        f"Locked threshold: "
        f"{LOCKED_PREDICTION_THRESHOLD:.2f}"
    )


    print(
        "No threshold search will be performed."
    )


    print(
        "No model weights will be changed."
    )


    print()


    for batch_index, batch in enumerate(
        loader
    ):

        images = (
            batch["image"]
            .to(device)
        )


        masks = (
            batch["mask"]
            .to(device)
        )


        # ----------------------------------------------------
        # Forward only
        # ----------------------------------------------------

        logits = model(
            images
        )


        probabilities = torch.sigmoid(
            logits
        )


        predictions = (
            probabilities
            >= LOCKED_PREDICTION_THRESHOLD
        ).float()


        current_batch_size = (
            images.shape[0]
        )


        for item_index in range(
            current_batch_size
        ):

            dataset_index = (

                batch_index
                * BATCH_SIZE
                + item_index
            )


            target = masks[
                item_index,
                0
            ]


            prediction = predictions[
                item_index,
                0
            ]


            probability = probabilities[
                item_index,
                0
            ]


            metrics = calculate_metrics(
                prediction,
                target,
            )


            ground_truth_pixels = int(
                target.sum().item()
            )


            predicted_pixels = int(
                prediction.sum().item()
            )


            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            sample_id = (
                batch[
                    "sample_id"
                ][item_index]
            )


            original_id = (
                batch[
                    "original_id"
                ][item_index]
            )


            size_group = (
                batch[
                    "nodule_size_group"
                ][item_index]
            )


            area_fraction_value = (
                batch[
                    "nodule_area_fraction"
                ][item_index]
            )


            if torch.is_tensor(
                area_fraction_value
            ):

                area_fraction = float(
                    area_fraction_value.item()
                )

            else:

                area_fraction = float(
                    area_fraction_value
                )


            result = {

                "dataset_index":
                    dataset_index,

                "sample_id":
                    sample_id,

                "original_id":
                    original_id,

                "nodule_size_group":
                    size_group,

                "nodule_area_fraction":
                    area_fraction,

                "dice":
                    metrics[
                        "dice"
                    ],

                "iou":
                    metrics[
                        "iou"
                    ],

                "precision":
                    metrics[
                        "precision"
                    ],

                "recall":
                    metrics[
                        "recall"
                    ],

                "true_positive":
                    metrics[
                        "true_positive"
                    ],

                "false_positive":
                    metrics[
                        "false_positive"
                    ],

                "false_negative":
                    metrics[
                        "false_negative"
                    ],

                "ground_truth_pixels":
                    ground_truth_pixels,

                "predicted_pixels":
                    predicted_pixels,

                "empty_prediction":
                    predicted_pixels == 0,

                "mean_probability":
                    float(
                        probability
                        .mean()
                        .item()
                    ),

                "max_probability":
                    float(
                        probability
                        .max()
                        .item()
                    ),
            }


            results.append(
                result
            )


        if (

            (batch_index + 1)
            % 50
            == 0

            or

            batch_index + 1
            == number_of_batches

        ):

            print(
                f"Processed batch "
                f"{batch_index + 1}"
                f"/{number_of_batches}"
            )


    if len(results) != EXPECTED_TEST_IMAGES:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TEST_IMAGES} results, "
            f"found {len(results)}."
        )


    return results


# ============================================================
# SUMMARY
# ============================================================

def summarize_rows(
    rows,
):

    dice_values = np.array(

        [
            row["dice"]
            for row in rows
        ],

        dtype=np.float64,
    )


    iou_values = np.array(

        [
            row["iou"]
            for row in rows
        ],

        dtype=np.float64,
    )


    precision_values = np.array(

        [
            row["precision"]
            for row in rows
        ],

        dtype=np.float64,
    )


    recall_values = np.array(

        [
            row["recall"]
            for row in rows
        ],

        dtype=np.float64,
    )


    empty_predictions = sum(

        1

        for row in rows

        if row[
            "empty_prediction"
        ]
    )


    return {

        "count":
            len(rows),

        "mean_dice":
            float(
                dice_values.mean()
            ),

        "median_dice":
            float(
                np.median(
                    dice_values
                )
            ),

        "std_dice":
            float(
                dice_values.std()
            ),

        "minimum_dice":
            float(
                dice_values.min()
            ),

        "maximum_dice":
            float(
                dice_values.max()
            ),

        "mean_iou":
            float(
                iou_values.mean()
            ),

        "mean_precision":
            float(
                precision_values.mean()
            ),

        "mean_recall":
            float(
                recall_values.mean()
            ),

        "empty_predictions":
            int(
                empty_predictions
            ),

        "empty_prediction_percent":
            float(
                empty_predictions
                / len(rows)
                * 100.0
            ),
    }


# ============================================================
# SIZE-GROUP SUMMARY
# ============================================================

def summarize_by_size(
    results,
):

    summaries = {}


    for group in [

        "tiny",
        "small",
        "medium",
        "large",

    ]:

        group_rows = [

            row

            for row in results

            if (
                row[
                    "nodule_size_group"
                ]
                == group
            )
        ]


        if group_rows:

            summaries[
                group
            ] = summarize_rows(
                group_rows
            )


    return summaries


# ============================================================
# DICE PERFORMANCE BANDS
# ============================================================

def calculate_dice_bands(
    results,
):

    bands = {

        "dice_ge_0_90":
            0,

        "dice_0_80_to_0_90":
            0,

        "dice_0_70_to_0_80":
            0,

        "dice_0_50_to_0_70":
            0,

        "dice_lt_0_50":
            0,
    }


    for row in results:

        dice = row[
            "dice"
        ]


        if dice >= 0.90:

            bands[
                "dice_ge_0_90"
            ] += 1


        elif dice >= 0.80:

            bands[
                "dice_0_80_to_0_90"
            ] += 1


        elif dice >= 0.70:

            bands[
                "dice_0_70_to_0_80"
            ] += 1


        elif dice >= 0.50:

            bands[
                "dice_0_50_to_0_70"
            ] += 1


        else:

            bands[
                "dice_lt_0_50"
            ] += 1


    return bands


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    results,
):

    fieldnames = [

        "dataset_index",

        "sample_id",
        "original_id",

        "nodule_size_group",
        "nodule_area_fraction",

        "dice",
        "iou",

        "precision",
        "recall",

        "true_positive",
        "false_positive",
        "false_negative",

        "ground_truth_pixels",
        "predicted_pixels",

        "empty_prediction",

        "mean_probability",
        "max_probability",
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
            results
        )


    print()
    print(
        "Saved per-image test metrics:"
    )

    print(
        CSV_PATH
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    checkpoint,
    results,
    overall,
    by_size,
):

    bands = calculate_dice_bands(
        results
    )


    sorted_results = sorted(

        results,

        key=lambda row:
            row["dice"],
    )


    lines = []


    lines.append(
        "TN3K V1 EARLY-STOPPING"
    )


    lines.append(
        "OFFICIAL TEST EVALUATION"
    )


    lines.append(
        "=" * 70
    )


    lines.append("")


    # ========================================================
    # LOCKED DEVELOPMENT CONFIGURATION
    # ========================================================

    lines.append(
        "LOCKED DEVELOPMENT CONFIGURATION"
    )


    lines.append(
        "-" * 70
    )


    lines.append(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )


    lines.append(
        f"Validation Dice used for model selection: "
        f"{checkpoint.get('best_validation_dice', -1):.6f}"
    )


    lines.append(
        f"Prediction threshold selected on validation: "
        f"{LOCKED_PREDICTION_THRESHOLD:.2f}"
    )


    lines.append(
        "Threshold tuned on test set: NO"
    )


    lines.append(
        "Model changed after viewing test set: NO"
    )


    lines.append("")
    lines.append("")


    # ========================================================
    # OVERALL TEST RESULT
    # ========================================================

    lines.append(
        "OFFICIAL TEST RESULTS"
    )


    lines.append(
        "-" * 70
    )


    lines.append(
        f"Images:          "
        f"{overall['count']}"
    )


    lines.append(
        f"Mean Dice:       "
        f"{overall['mean_dice']:.6f}"
    )


    lines.append(
        f"Median Dice:     "
        f"{overall['median_dice']:.6f}"
    )


    lines.append(
        f"Dice std:        "
        f"{overall['std_dice']:.6f}"
    )


    lines.append(
        f"Minimum Dice:    "
        f"{overall['minimum_dice']:.6f}"
    )


    lines.append(
        f"Maximum Dice:    "
        f"{overall['maximum_dice']:.6f}"
    )


    lines.append(
        f"Mean IoU:        "
        f"{overall['mean_iou']:.6f}"
    )


    lines.append(
        f"Mean Precision:  "
        f"{overall['mean_precision']:.6f}"
    )


    lines.append(
        f"Mean Recall:     "
        f"{overall['mean_recall']:.6f}"
    )


    lines.append(
        f"Empty predictions: "
        f"{overall['empty_predictions']}"
        f"/{overall['count']} "
        f"({overall['empty_prediction_percent']:.2f}%)"
    )


    # ========================================================
    # DEVELOPMENT VS TEST
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "VALIDATION VS OFFICIAL TEST"
    )


    lines.append(
        "-" * 70
    )


    validation_dice = float(
        checkpoint.get(
            "best_validation_dice",
            -1.0,
        )
    )


    test_difference = (

        overall[
            "mean_dice"
        ]

        - validation_dice
    )


    lines.append(
        f"Validation Dice: "
        f"{validation_dice:.6f}"
    )


    lines.append(
        f"Official test Dice: "
        f"{overall['mean_dice']:.6f}"
    )


    lines.append(
        f"Test - validation difference: "
        f"{test_difference:+.6f}"
    )


    # ========================================================
    # DICE BANDS
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "DICE PERFORMANCE BANDS"
    )


    lines.append(
        "-" * 70
    )


    lines.append(
        f"Dice >= 0.90:      "
        f"{bands['dice_ge_0_90']}"
    )


    lines.append(
        f"Dice 0.80 - 0.90:  "
        f"{bands['dice_0_80_to_0_90']}"
    )


    lines.append(
        f"Dice 0.70 - 0.80:  "
        f"{bands['dice_0_70_to_0_80']}"
    )


    lines.append(
        f"Dice 0.50 - 0.70:  "
        f"{bands['dice_0_50_to_0_70']}"
    )


    lines.append(
        f"Dice < 0.50:       "
        f"{bands['dice_lt_0_50']}"
    )


    # ========================================================
    # SIZE GROUPS
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "OFFICIAL TEST RESULTS BY NODULE SIZE"
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

        if group not in by_size:

            continue


        values = by_size[
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


        lines.append(
            f"  Empty predictions: "
            f"{values['empty_predictions']}"
            f"/{values['count']}"
        )


    # ========================================================
    # WORST TEST CASES
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "10 LOWEST-DICE OFFICIAL TEST CASES"
    )


    lines.append(
        "-" * 70
    )


    for row in sorted_results[:10]:

        lines.append(

            f"{row['sample_id']:18} "
            f"group={row['nodule_size_group']:7} "
            f"Dice={row['dice']:.6f} "
            f"IoU={row['iou']:.6f} "
            f"P={row['precision']:.6f} "
            f"R={row['recall']:.6f} "
            f"GT={row['ground_truth_pixels']} "
            f"Pred={row['predicted_pixels']}"
        )


    # ========================================================
    # BEST TEST CASES
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "10 HIGHEST-DICE OFFICIAL TEST CASES"
    )


    lines.append(
        "-" * 70
    )


    for row in (
        sorted_results[-10:][::-1]
    ):

        lines.append(

            f"{row['sample_id']:18} "
            f"group={row['nodule_size_group']:7} "
            f"Dice={row['dice']:.6f} "
            f"IoU={row['iou']:.6f}"
        )


    summary_text = "\n".join(
        lines
    )


    SUMMARY_PATH.write_text(

        summary_text,

        encoding="utf-8",
    )


    print()
    print(
        summary_text
    )


    print()
    print(
        "Saved official test summary:"
    )

    print(
        SUMMARY_PATH
    )


# ============================================================
# VISUAL EXAMPLE
# ============================================================

@torch.no_grad()
def save_visual(
    model,
    dataset,
    device,
    row,
    prefix,
):

    dataset_index = int(
        row[
            "dataset_index"
        ]
    )


    sample = dataset[
        dataset_index
    ]


    image = sample[
        "image"
    ]


    target = sample[
        "mask"
    ]


    model_input = (

        image
        .unsqueeze(0)
        .to(device)
    )


    logits = model(
        model_input
    )


    probability = (

        torch.sigmoid(
            logits
        )[0, 0]
        .cpu()
        .numpy()
    )


    prediction = (

        probability
        >= LOCKED_PREDICTION_THRESHOLD

    ).astype(
        np.float32
    )


    image_np = (

        image[
            0
        ]
        .cpu()
        .numpy()
    )


    target_np = (

        target[
            0
        ]
        .cpu()
        .numpy()
    )


    # ========================================================
    # FIGURE
    # ========================================================

    figure, axes = plt.subplots(

        1,
        5,

        figsize=(20, 4),
    )


    # --------------------------------------------------------
    # Ultrasound
    # --------------------------------------------------------

    axes[0].imshow(

        image_np,

        cmap="gray",

        vmin=0,

        vmax=1,
    )


    axes[0].set_title(
        "Ultrasound"
    )


    axes[0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(

        target_np,

        cmap="gray",

        vmin=0,

        vmax=1,
    )


    axes[1].set_title(
        "Ground Truth"
    )


    axes[1].axis(
        "off"
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_display = (
        axes[2].imshow(

            probability,

            vmin=0,

            vmax=1,
        )
    )


    axes[2].set_title(
        "Probability"
    )


    axes[2].axis(
        "off"
    )


    figure.colorbar(

        probability_display,

        ax=axes[2],

        fraction=0.046,

        pad=0.04,
    )


    # --------------------------------------------------------
    # Predicted mask
    # --------------------------------------------------------

    axes[3].imshow(

        prediction,

        cmap="gray",

        vmin=0,

        vmax=1,
    )


    axes[3].set_title(
        "Predicted Mask"
    )


    axes[3].axis(
        "off"
    )


    # --------------------------------------------------------
    # Overlay
    #
    # Green = ground truth
    # Red   = AI prediction
    # --------------------------------------------------------

    axes[4].imshow(

        image_np,

        cmap="gray",

        vmin=0,

        vmax=1,
    )


    if target_np.max() > 0:

        axes[4].contour(

            target_np,

            levels=[0.5],

            linewidths=2,

            colors="lime",
        )


    if prediction.max() > 0:

        axes[4].contour(

            prediction,

            levels=[0.5],

            linewidths=2,

            colors="red",
        )


    axes[4].set_title(
        "GT (Green) + Prediction (Red)"
    )


    axes[4].axis(
        "off"
    )


    figure.suptitle(

        f"{row['sample_id']} | "
        f"{row['nodule_size_group']} | "
        f"Dice={row['dice']:.4f} | "
        f"IoU={row['iou']:.4f} | "
        f"P={row['precision']:.4f} | "
        f"R={row['recall']:.4f}"
    )


    plt.tight_layout()


    output_path = (

        OUTPUT_DIR

        / (

            f"{prefix}_"
            f"{row['sample_id']}_"
            f"dice_{row['dice']:.4f}.png"
        )
    )


    figure.savefig(

        output_path,

        dpi=160,

        bbox_inches="tight",
    )


    plt.close(
        figure
    )


    print(
        "Saved:",
        output_path.name
    )


# ============================================================
# BEST / WORST VISUALS
# ============================================================

def save_visual_examples(
    model,
    dataset,
    device,
    results,
):

    sorted_results = sorted(

        results,

        key=lambda row:
            row["dice"],
    )


    worst = sorted_results[
        :NUMBER_OF_WORST_EXAMPLES
    ]


    best = (
        sorted_results[
            -NUMBER_OF_BEST_EXAMPLES:
        ][::-1]
    )


    print()
    print("=" * 70)
    print(
        "SAVING WORST OFFICIAL TEST CASES"
    )
    print("=" * 70)


    for rank, row in enumerate(
        worst,
        start=1,
    ):

        save_visual(

            model=model,

            dataset=dataset,

            device=device,

            row=row,

            prefix=(
                f"worst_{rank:02d}"
            ),
        )


    print()
    print("=" * 70)
    print(
        "SAVING BEST OFFICIAL TEST CASES"
    )
    print("=" * 70)


    for rank, row in enumerate(
        best,
        start=1,
    ):

        save_visual(

            model=model,

            dataset=dataset,

            device=device,

            row=row,

            prefix=(
                f"best_{rank:02d}"
            ),
        )


# ============================================================
# WRITE COMPLETION MARKER
# ============================================================

def write_completion_marker(
    overall,
):

    text = (

        "TN3K OFFICIAL TEST EVALUATION COMPLETE\n"
        "======================================\n\n"

        f"Locked checkpoint epoch: "
        f"{EXPECTED_CHECKPOINT_EPOCH}\n"

        f"Locked validation threshold: "
        f"{LOCKED_PREDICTION_THRESHOLD:.2f}\n"

        f"Official test images: "
        f"{overall['count']}\n"

        f"Official test mean Dice: "
        f"{overall['mean_dice']:.6f}\n\n"

        "IMPORTANT:\n"
        "Do not tune the model or threshold using "
        "these official test results.\n"
    )


    COMPLETION_MARKER.write_text(

        text,

        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "TN3K V1 EARLY-STOPPING"
    )

    print(
        "FINAL OFFICIAL TEST EVALUATION"
    )

    print("=" * 70)


    # ========================================================
    # TEST-SET SAFETY
    # ========================================================

    print()
    print(
        "LOCKED BEFORE TEST:"
    )


    print(
        f"  Checkpoint epoch: "
        f"{EXPECTED_CHECKPOINT_EPOCH}"
    )


    print(
        f"  Validation Dice: "
        f"{EXPECTED_VALIDATION_DICE:.6f}"
    )


    print(
        f"  Prediction threshold: "
        f"{LOCKED_PREDICTION_THRESHOLD:.2f}"
    )


    print()
    print(
        "NO threshold sweep will be performed on test."
    )


    print(
        "NO optimization will be performed on test."
    )


    print(
        "NO training will be performed on test."
    )


    # ========================================================
    # IF ALREADY COMPLETED
    # ========================================================

    if COMPLETION_MARKER.exists():

        print()
        print("=" * 70)

        print(
            "WARNING:"
        )

        print(
            "An official test evaluation has already "
            "completed for this experiment."
        )

        print("=" * 70)


        print()
        print(
            "Completion marker:"
        )


        print(
            COMPLETION_MARKER
        )


        print()
        print(
            COMPLETION_MARKER.read_text(
                encoding="utf-8"
            )
        )


        print(
            "No new evaluation was performed."
        )


        return


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
        create_test_loader()
    )


    print()
    print(
        "Official test images:",
        len(dataset)
    )


    # ========================================================
    # MODEL
    # ========================================================

    model, checkpoint = (
        load_locked_model(
            device
        )
    )


    # ========================================================
    # EVALUATE
    # ========================================================

    results = evaluate_official_test(

        model=model,

        loader=loader,

        device=device,
    )


    # ========================================================
    # SUMMARIZE
    # ========================================================

    overall = summarize_rows(
        results
    )


    by_size = summarize_by_size(
        results
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_csv(
        results
    )


    save_summary(

        checkpoint=checkpoint,

        results=results,

        overall=overall,

        by_size=by_size,
    )


    save_visual_examples(

        model=model,

        dataset=dataset,

        device=device,

        results=results,
    )


    # ========================================================
    # MARK COMPLETE ONLY AFTER EVERYTHING SUCCEEDS
    # ========================================================

    write_completion_marker(
        overall
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "FINAL TN3K OFFICIAL TEST RESULT"
    )

    print("=" * 70)


    print()
    print(
        f"Images: "
        f"{overall['count']}"
    )


    print(
        f"Mean Dice: "
        f"{overall['mean_dice']:.6f}"
    )


    print(
        f"Median Dice: "
        f"{overall['median_dice']:.6f}"
    )


    print(
        f"Mean IoU: "
        f"{overall['mean_iou']:.6f}"
    )


    print(
        f"Precision: "
        f"{overall['mean_precision']:.6f}"
    )


    print(
        f"Recall: "
        f"{overall['mean_recall']:.6f}"
    )


    print(
        f"Empty predictions: "
        f"{overall['empty_predictions']}"
        f"/{overall['count']}"
    )


    print()
    print(
        "Official test evaluation is now LOCKED."
    )


    print(
        "Do not use these test results to "
        "retune the model or threshold."
    )


if __name__ == "__main__":

    main()