"""Validation-only diagnostics for the locked V1 negative-aware checkpoint.

Implements audit tasks F (error analysis), G (calibration/threshold curves),
H (lesion-size stratification), plus an extra BatchNorm train/eval-gap probe
(one hypothesis for the train/validation Dice gap).

Hard rules enforced by this script:
  * Only split="validation" is ever instantiated. split="test" is never
    imported, created, or touched.
  * The checkpoint is loaded read-only; nothing is fine-tuned, and the file
    on disk is never overwritten.
  * PREDICTION_THRESHOLD below is used only to REPORT how metrics behave
    at other thresholds. The deployed checkpoint's threshold (0.35) is not
    changed anywhere by this script.

Run this in the same Colab environment used for training (needs torch,
pydicom, and the prepared_data / SIIM_TRAIN_TEST directories on Drive).
"""

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pneumothorax_dataset import PneumothoraxDataset
from pneumothorax_model import PneumothoraxResNet34UNet


CHECKPOINT_PATH = Path("checkpoints/pneumothorax_512_negative_aware_best.pth")
OUTPUT_DIRECTORY = Path("diagnostics_negative_aware_validation")
DEPLOYED_THRESHOLD = 0.35
IMAGE_SIZE = 512
BATCH_SIZE = 2

# Thresholds swept ONLY for reporting; never used to change the checkpoint.
THRESHOLD_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

# Lesion-size bins defined up front (ground-truth mask fraction of the
# 512x512=262144-pixel image), before looking at any results.
LESION_SIZE_BINS = [
    ("tiny", 0.0, 0.001),      # < 0.1% of the image
    ("small", 0.001, 0.005),   # 0.1% - 0.5%
    ("medium", 0.005, 0.02),   # 0.5% - 2%
    ("large", 0.02, 1.0),      # > 2%
]


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_locked_model(device):
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}")

    checkpoint = load_torch_checkpoint(CHECKPOINT_PATH, device)
    configuration = checkpoint.get("configuration", {})

    if configuration.get("test_split_used") is not False:
        raise ValueError("Checkpoint metadata does not confirm an untouched test split.")
    if int(configuration.get("image_size", -1)) != IMAGE_SIZE:
        raise ValueError("Unexpected checkpoint image size.")

    model = PneumothoraxResNet34UNet(use_pretrained_encoder=False, freeze_encoder=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint epoch {checkpoint.get('completed_epoch')} "
          f"({checkpoint.get('training_stage')})")
    return model, checkpoint


@torch.inference_mode()
def collect_validation_predictions(model, loader, device):
    """Run the validation set once and cache everything diagnostics need.

    Only per-image scalar summaries (areas, max/mean probability, dice at
    the deployed threshold) plus a handful of full images/masks for the
    example grids are kept in memory; this avoids holding all 1,205
    512x512 probability maps at once.
    """

    model.eval()
    records = []
    example_cache = {}  # keyed by category -> list of (image, mask, prob) tuples
    max_examples_per_category = 6

    use_amp = device.type == "cuda"
    total_batches = len(loader)
    started = time.perf_counter()

    for batch_number, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        image_ids = batch["image_id"]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        probabilities = torch.sigmoid(logits).float()

        binary_targets = targets >= 0.5
        predictions = probabilities >= DEPLOYED_THRESHOLD
        dims = (1, 2, 3)

        intersection = (predictions & binary_targets).sum(dim=dims).float()
        pred_area = predictions.sum(dim=dims).float()
        tgt_area = binary_targets.sum(dim=dims).float()
        denom = pred_area + tgt_area
        dice = torch.where(denom > 0, 2 * intersection / denom, torch.ones_like(denom))

        max_prob = probabilities.flatten(1).amax(dim=1)
        mean_prob = probabilities.flatten(1).mean(dim=1)

        for i in range(images.shape[0]):
            is_positive = tgt_area[i].item() > 0
            record = {
                "image_id": image_ids[i],
                "is_positive": is_positive,
                "target_area_pixels": int(tgt_area[i].item()),
                "target_area_fraction": tgt_area[i].item() / (IMAGE_SIZE * IMAGE_SIZE),
                "predicted_area_pixels": int(pred_area[i].item()),
                "dice_at_deployed_threshold": dice[i].item(),
                "max_probability": max_prob[i].item(),
                "mean_probability": mean_prob[i].item(),
            }
            records.append(record)

            # Cache a few examples per category for the visual grids.
            if is_positive:
                if pred_area[i].item() == 0:
                    category = "positive_empty_miss"
                elif dice[i].item() < 0.3:
                    category = "positive_low_dice"
                elif dice[i].item() < 0.7:
                    category = "positive_medium_dice"
                else:
                    category = "positive_high_dice"
            else:
                if pred_area[i].item() == 0:
                    category = "negative_correct_empty"
                elif pred_area[i].item() > 0.01 * IMAGE_SIZE * IMAGE_SIZE:
                    category = "negative_large_false_positive"
                else:
                    category = "negative_tiny_false_positive"

            bucket = example_cache.setdefault(category, [])
            if len(bucket) < max_examples_per_category:
                bucket.append(
                    (
                        images[i, 0].detach().cpu().numpy(),
                        targets[i, 0].detach().cpu().numpy(),
                        probabilities[i, 0].detach().cpu().numpy(),
                        record,
                    )
                )

        if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_batches:
            elapsed_minutes = (time.perf_counter() - started) / 60.0
            images_done = len(records)
            rate = images_done / max(elapsed_minutes, 1e-9)
            remaining_minutes = (len(loader.dataset) - images_done) / max(rate, 1e-9)
            print(
                f"  Validation pass [{batch_number}/{total_batches} batches, "
                f"{images_done}/{len(loader.dataset)} images] "
                f"elapsed={elapsed_minutes:.1f} min, "
                f"est. remaining={remaining_minutes:.1f} min",
                flush=True,
            )

    return records, example_cache


def save_example_grids(example_cache, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)

    for category, examples in example_cache.items():
        if not examples:
            continue

        rows = len(examples)
        figure, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
        if rows == 1:
            axes = axes[None, :]

        for row_index, (image, mask, prob, record) in enumerate(examples):
            thresholded = (prob >= DEPLOYED_THRESHOLD).astype(float)

            axes[row_index, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 0].set_title(f"{record['image_id'][:18]}\nX-ray")

            axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row_index, 1].set_title("Ground truth")

            axes[row_index, 2].imshow(prob, cmap="inferno", vmin=0, vmax=1)
            axes[row_index, 2].set_title(
                f"Prob heatmap\nmax={record['max_probability']:.2f} "
                f"mean={record['mean_probability']:.3f}"
            )

            axes[row_index, 3].imshow(image, cmap="gray", vmin=0, vmax=1)
            overlay = np.zeros((*image.shape, 4))
            overlay[thresholded > 0] = (1, 0, 0, 0.4)
            overlay[mask > 0.5] = (0, 1, 0, 0.4)
            axes[row_index, 3].imshow(overlay)
            axes[row_index, 3].set_title(
                f"Overlay (green=GT, red=pred)\nDice={record['dice_at_deployed_threshold']:.3f} "
                f"GT area={record['target_area_pixels']}px pred={record['predicted_area_pixels']}px"
            )

            for column in range(4):
                axes[row_index, column].axis("off")

        figure.suptitle(f"Validation-only error category: {category}", fontsize=14)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(output_directory / f"{category}.png", dpi=130, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved example grid: {output_directory / f'{category}.png'} ({len(examples)} cases)")


def summarize_probability_and_area_distributions(records, output_directory):
    positive_max = np.array([r["max_probability"] for r in records if r["is_positive"]])
    negative_max = np.array([r["max_probability"] for r in records if not r["is_positive"]])
    negative_area_fraction = np.array(
        [r["predicted_area_pixels"] / (IMAGE_SIZE * IMAGE_SIZE) for r in records if not r["is_positive"]]
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(positive_max, bins=30, alpha=0.6, label="positive images", color="tab:red")
    axes[0].hist(negative_max, bins=30, alpha=0.6, label="negative images", color="tab:blue")
    axes[0].axvline(DEPLOYED_THRESHOLD, color="black", linestyle="--", label=f"deployed threshold {DEPLOYED_THRESHOLD}")
    axes[0].set_title("Per-image max predicted probability")
    axes[0].set_xlabel("max probability")
    axes[0].legend()

    axes[1].hist(negative_area_fraction, bins=40, color="tab:blue")
    axes[1].set_title("Predicted area fraction on negative images")
    axes[1].set_xlabel("predicted area / image area")

    output_directory.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_directory / "probability_and_area_distributions.png", dpi=130, bbox_inches="tight")
    plt.close(figure)

    print("\nMax-probability summary")
    print(f"  Positive images: median={np.median(positive_max):.3f} mean={positive_max.mean():.3f}")
    print(f"  Negative images: median={np.median(negative_max):.3f} mean={negative_max.mean():.3f}")
    overlap = np.mean(negative_max >= DEPLOYED_THRESHOLD)
    print(f"  Fraction of NEGATIVE images with max prob >= deployed threshold: {overlap:.3f}")
    print("  (this is the source of the ~30% negative false-positive rate)")


def threshold_sweep_report(records):
    """Report-only: how positive Dice / miss rate / FP rate move with threshold.

    This NEVER changes the deployed checkpoint threshold. It only tells you
    whether 0.35 sits in a sensible part of the precision/recall curve, or
    whether the model is simply mis-calibrated around that value.
    """

    positive_records = [r for r in records if r["is_positive"]]
    negative_records = [r for r in records if not r["is_positive"]]

    print("\nThreshold sweep (validation only, report-only, threshold NOT changed)")
    print(f"{'thr':>5} | {'miss_rate':>10} | {'neg_fp_rate':>12} | {'mean_pos_area_frac':>18}")

    rows = []
    for thr in THRESHOLD_SWEEP:
        misses = sum(1 for r in positive_records if r["max_probability"] < thr)
        miss_rate = misses / len(positive_records)
        neg_fp = sum(1 for r in negative_records if r["max_probability"] >= thr)
        neg_fp_rate = neg_fp / len(negative_records)
        rows.append({"threshold": thr, "positive_miss_rate": miss_rate, "negative_false_positive_rate": neg_fp_rate})
        print(f"{thr:>5.2f} | {miss_rate:>10.3f} | {neg_fp_rate:>12.3f} |")

    return rows


def lesion_size_report(records):
    positive_records = [r for r in records if r["is_positive"]]

    print("\nLesion-size stratified validation performance (ground-truth area fraction bins)")
    print(f"{'bin':>8} | {'n':>4} | {'mean_dice':>10} | {'median_dice':>12} | {'empty_pred_rate':>16}")

    results = {}
    for name, low, high in LESION_SIZE_BINS:
        bucket = [r for r in positive_records if low <= r["target_area_fraction"] < high]
        if not bucket:
            print(f"{name:>8} | {'0':>4} | (no cases in this split fall in this bin)")
            continue
        dices = np.array([r["dice_at_deployed_threshold"] for r in bucket])
        empty_rate = np.mean([r["predicted_area_pixels"] == 0 for r in bucket])
        results[name] = {
            "n": len(bucket),
            "mean_dice": float(dices.mean()),
            "median_dice": float(np.median(dices)),
            "empty_prediction_rate": float(empty_rate),
        }
        print(
            f"{name:>8} | {len(bucket):>4} | {dices.mean():>10.3f} | "
            f"{np.median(dices):>12.3f} | {empty_rate:>16.3f}"
        )

    return results


@torch.inference_mode()
def batchnorm_train_eval_gap_probe(model, loader, device, num_batches=40):
    """Isolate whether decoder BatchNorm (batch_size=2) explains part of the gap.

    Runs the SAME validation images twice:
      (a) model.eval()  -> uses BN running statistics (the real deployed path)
      (b) model.train()-with-no-grad -> uses this tiny batch's own statistics
    A large gap between (a) and (b) means the decoder's BatchNorm running
    statistics (estimated from batch_size=2 during training) do not match
    the true population statistics well -- a plausible, checkable
    contributor to the train/validation Dice gap. This never updates any
    weights; gradients stay off throughout (torch.inference_mode()).
    """

    dims = (1, 2, 3)
    eval_dices, batchstat_dices = [], []

    for batch_number, batch in enumerate(loader, start=1):
        if batch_number > num_batches:
            break

        images = batch["image"].to(device)
        targets = batch["mask"].to(device) >= 0.5

        model.eval()
        for m in model.encoder_modules():
            m.eval()
        logits_eval = model(images)

        model.train()
        for m in model.encoder_modules():
            m.eval()  # encoder BN must stay frozen either way, per the training script
        logits_batchstat = model(images)

        for logits, store in ((logits_eval, eval_dices), (logits_batchstat, batchstat_dices)):
            preds = torch.sigmoid(logits) >= DEPLOYED_THRESHOLD
            inter = (preds & targets).sum(dim=dims).float()
            pred_area = preds.sum(dim=dims).float()
            tgt_area = targets.sum(dim=dims).float()
            denom = pred_area + tgt_area
            dice = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
            store.extend(dice.cpu().tolist())

    model.eval()  # restore the correct deployed mode before returning
    eval_mean = float(np.mean(eval_dices))
    batchstat_mean = float(np.mean(batchstat_dices))
    gap = batchstat_mean - eval_mean

    print("\nBatchNorm train/eval-mode probe (first "
          f"{min(num_batches, len(loader))} validation batches, all images)")
    print(f"  Dice using eval-mode running statistics (real deployed path): {eval_mean:.4f}")
    print(f"  Dice using this tiny batch's OWN statistics (train-mode BN):  {batchstat_mean:.4f}")
    print(f"  Gap (batch-stat - eval-stat): {gap:+.4f}")
    if abs(gap) > 0.03:
        print("  -> Non-trivial gap: batch_size=2 BatchNorm running statistics in the")
        print("     decoder are a plausible contributor to the train/validation Dice gap.")
    else:
        print("  -> Small gap: BatchNorm running-statistic mismatch does not look like")
        print("     a major contributor on this sample.")

    return {"eval_mode_dice": eval_mean, "train_batchstat_dice": batchstat_mean, "gap": gap}


def main():
    device = choose_device()
    print(f"Device: {device}")

    model, checkpoint = load_locked_model(device)

    validation_dataset = PneumothoraxDataset(split="validation", image_size=IMAGE_SIZE)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    print(f"Validation images: {len(validation_dataset)} (test split NOT instantiated)")

    records, example_cache = collect_validation_predictions(model, validation_loader, device)

    save_example_grids(example_cache, OUTPUT_DIRECTORY / "example_grids")
    summarize_probability_and_area_distributions(records, OUTPUT_DIRECTORY)
    threshold_rows = threshold_sweep_report(records)
    lesion_rows = lesion_size_report(records)
    bn_probe = batchnorm_train_eval_gap_probe(model, validation_loader, device)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint_epoch": checkpoint.get("completed_epoch"),
        "deployed_threshold": DEPLOYED_THRESHOLD,
        "validation_images": len(validation_dataset),
        "threshold_sweep_report_only": threshold_rows,
        "lesion_size_report": lesion_rows,
        "batchnorm_probe": bn_probe,
        "test_split_used": False,
        "note": "Report-only diagnostics. Deployed checkpoint and threshold were not modified.",
    }
    (OUTPUT_DIRECTORY / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUTPUT_DIRECTORY / 'diagnostic_summary.json'}")
    print("Test split was never created or accessed.")


if __name__ == "__main__":
    main()
