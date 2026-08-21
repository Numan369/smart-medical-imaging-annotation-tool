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
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CSV_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_earlystop_validation_per_image_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "tn3k_v1_earlystop_validation_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_WORKERS = 0
PREDICTION_THRESHOLD = 0.50

EPSILON = 1e-6

NUMBER_OF_WORST_EXAMPLES = 6
NUMBER_OF_BEST_EXAMPLES = 6


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
):
    """
    prediction and target:
        binary tensors [H, W]
    """

    prediction = (
        prediction >= 0.5
    ).float()

    target = (
        target >= 0.5
    ).float()


    true_positive = (
        prediction * target
    ).sum().item()


    false_positive = (
        prediction
        * (1.0 - target)
    ).sum().item()


    false_negative = (
        (1.0 - prediction)
        * target
    ).sum().item()


    dice = (
        2.0 * true_positive
        + EPSILON
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + EPSILON
    )


    iou = (
        true_positive
        + EPSILON
    ) / (
        true_positive
        + false_positive
        + false_negative
        + EPSILON
    )


    precision = (
        true_positive
        + EPSILON
    ) / (
        true_positive
        + false_positive
        + EPSILON
    )


    recall = (
        true_positive
        + EPSILON
    ) / (
        true_positive
        + false_negative
        + EPSILON
    )


    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }


# ============================================================
# LOAD VALIDATION DATA
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
    print("Loading checkpoint:")

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
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )


    print(
        f"Saved best validation Dice: "
        f"{checkpoint.get('best_validation_dice', -1):.6f}"
    )


    return model, checkpoint


# ============================================================
# FULL VALIDATION EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_validation(
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
    print("FULL TN3K FOLD-0 VALIDATION EVALUATION")
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
            .to(device)
        )


        logits = model(
            images
        )


        probabilities = torch.sigmoid(
            logits
        )


        predictions = (
            probabilities
            >= PREDICTION_THRESHOLD
        ).float()


        current_batch_size = (
            images.shape[0]
        )


        for item_index in range(
            current_batch_size
        ):

            # Because validation loader has:
            #
            # shuffle=False
            # drop_last=False
            #
            # this gives the exact dataset index.
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


            gt_pixels = int(
                target.sum().item()
            )


            predicted_pixels = int(
                prediction.sum().item()
            )


            # ------------------------------------------------
            # Read metadata
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
                    metrics["dice"],

                "iou":
                    metrics["iou"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "true_positive":
                    metrics["true_positive"],

                "false_positive":
                    metrics["false_positive"],

                "false_negative":
                    metrics["false_negative"],

                "ground_truth_pixels":
                    gt_pixels,

                "predicted_pixels":
                    predicted_pixels,

                "missed_nodule":
                    predicted_pixels == 0,

                "mean_probability":
                    float(
                        probability.mean().item()
                    ),

                "max_probability":
                    float(
                        probability.max().item()
                    ),
            }


            results.append(
                result
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


    if len(results) != 576:

        raise RuntimeError(
            f"Expected 576 results, "
            f"found {len(results)}."
        )


    return results


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_rows(
    rows,
):

    dice = np.array(
        [
            row["dice"]
            for row in rows
        ],
        dtype=np.float64,
    )


    iou = np.array(
        [
            row["iou"]
            for row in rows
        ],
        dtype=np.float64,
    )


    precision = np.array(
        [
            row["precision"]
            for row in rows
        ],
        dtype=np.float64,
    )


    recall = np.array(
        [
            row["recall"]
            for row in rows
        ],
        dtype=np.float64,
    )


    missed = sum(
        row["missed_nodule"]
        for row in rows
    )


    return {

        "count":
            len(rows),

        "mean_dice":
            float(
                dice.mean()
            ),

        "median_dice":
            float(
                np.median(dice)
            ),

        "std_dice":
            float(
                dice.std()
            ),

        "min_dice":
            float(
                dice.min()
            ),

        "max_dice":
            float(
                dice.max()
            ),

        "mean_iou":
            float(
                iou.mean()
            ),

        "mean_precision":
            float(
                precision.mean()
            ),

        "mean_recall":
            float(
                recall.mean()
            ),

        "missed_nodules":
            int(missed),

        "missed_percent":
            float(
                missed
                / len(rows)
                * 100.0
            ),
    }


# ============================================================
# SIZE GROUP SUMMARY
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

            summaries[group] = (
                summarize_rows(
                    group_rows
                )
            )


    return summaries


# ============================================================
# PERFORMANCE-BAND COUNTS
# ============================================================

def calculate_dice_bands(
    results,
):

    bands = {

        "dice_ge_0_90": 0,

        "dice_0_80_to_0_90": 0,

        "dice_0_70_to_0_80": 0,

        "dice_0_50_to_0_70": 0,

        "dice_lt_0_50": 0,
    }


    for row in results:

        value = row[
            "dice"
        ]


        if value >= 0.90:

            bands[
                "dice_ge_0_90"
            ] += 1


        elif value >= 0.80:

            bands[
                "dice_0_80_to_0_90"
            ] += 1


        elif value >= 0.70:

            bands[
                "dice_0_70_to_0_80"
            ] += 1


        elif value >= 0.50:

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

    fields = [

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

        "missed_nodule",

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
            fieldnames=fields,
        )


        writer.writeheader()


        writer.writerows(
            results
        )


    print()
    print(
        "Saved per-image CSV:"
    )

    print(
        CSV_PATH
    )


# ============================================================
# SAVE SUMMARY TEXT
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


    lines = []


    lines.append(
        "TN3K V1 EARLY-STOPPING VALIDATION DIAGNOSTIC"
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
        f"Checkpoint best validation Dice: "
        f"{checkpoint.get('best_validation_dice', -1):.6f}"
    )


    lines.append(
        f"Prediction threshold: "
        f"{PREDICTION_THRESHOLD:.2f}"
    )


    lines.append("")


    # ========================================================
    # OVERALL
    # ========================================================

    lines.append(
        "OVERALL VALIDATION RESULTS"
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
        f"{overall['min_dice']:.6f}"
    )


    lines.append(
        f"Maximum Dice:    "
        f"{overall['max_dice']:.6f}"
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
        f"{overall['missed_nodules']}"
        f"/{overall['count']} "
        f"({overall['missed_percent']:.2f}%)"
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
        "RESULTS BY NODULE SIZE"
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


        values = (
            by_size[group]
        )


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
            f"  Missed:     "
            f"{values['missed_nodules']}"
            f"/{values['count']}"
        )


    # ========================================================
    # WORST CASES
    # ========================================================

    sorted_results = sorted(
        results,
        key=lambda row:
            row["dice"],
    )


    lines.append("")
    lines.append("")


    lines.append(
        "10 LOWEST-DICE VALIDATION CASES"
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
            f"Precision={row['precision']:.6f} "
            f"Recall={row['recall']:.6f} "
            f"GT={row['ground_truth_pixels']} "
            f"Pred={row['predicted_pixels']}"
        )


    # ========================================================
    # BEST CASES
    # ========================================================

    lines.append("")
    lines.append("")


    lines.append(
        "10 HIGHEST-DICE VALIDATION CASES"
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


    text = "\n".join(
        lines
    )


    SUMMARY_PATH.write_text(
        text,
        encoding="utf-8",
    )


    print()
    print(text)


    print()
    print(
        "Saved summary:"
    )

    print(
        SUMMARY_PATH
    )


# ============================================================
# SAVE VISUAL EXAMPLE
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
        >= PREDICTION_THRESHOLD
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
    # DISPLAY
    # ========================================================

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4),
    )


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


    axes[4].imshow(
        image_np,
        cmap="gray",
        vmin=0,
        vmax=1,
    )


    # Ground-truth contour.
    if target_np.max() > 0:

        axes[4].contour(
            target_np,
            levels=[0.5],
            linewidths=2,
        )


    # Prediction contour.
    if prediction.max() > 0:

        axes[4].contour(
            prediction,
            levels=[0.5],
            linewidths=2,
        )


    axes[4].set_title(
        "GT + Prediction"
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
        /
        (
            f"{prefix}_"
            f"{row['sample_id']}_"
            f"dice_{row['dice']:.4f}.png"
        )
    )


    figure.savefig(
        output_path,
        dpi=150,
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
# SAVE BEST/WORST VISUALS
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
    print("SAVING WORST VALIDATION CASES")
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
    print("SAVING BEST VALIDATION CASES")
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TN3K V1 EARLY-STOPPING "
        "BEST-CHECKPOINT VALIDATION DIAGNOSTIC"
    )
    print("=" * 70)


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
        "Validation samples:",
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
    # EVALUATE
    # ========================================================

    results = evaluate_validation(
        model=model,
        loader=loader,
        device=device,
    )


    # ========================================================
    # SUMMARIES
    # ========================================================

    overall = summarize_rows(
        results
    )


    by_size = summarize_by_size(
        results
    )


    # ========================================================
    # SAVE OUTPUTS
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
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "VALIDATION DIAGNOSTIC COMPLETE"
    )
    print("=" * 70)


    print()
    print(
        "Official TN3K test set was NOT used."
    )


    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()