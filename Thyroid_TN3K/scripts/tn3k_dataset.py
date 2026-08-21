from pathlib import Path
import csv

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# PATHS
# ============================================================

THYROID_ROOT = Path(__file__).resolve().parent.parent

INDEX_CSV = (
    THYROID_ROOT
    / "prepared_data"
    / "tn3k_dataset_index.csv"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 512

# Verified from TN3K JPEG masks:
#
# background approximately 0-11
# foreground approximately 243-255
#
MASK_THRESHOLD = 128


# ============================================================
# LETTERBOX RESIZE
# ============================================================

def resize_and_pad(
    image,
    mask,
    target_size=IMAGE_SIZE,
):
    """
    Resize image and mask while preserving aspect ratio,
    then center-pad both to target_size x target_size.

    Image:
        bilinear interpolation

    Mask:
        nearest-neighbor interpolation
    """

    original_width, original_height = image.size

    scale = min(
        target_size / original_width,
        target_size / original_height,
    )

    new_width = int(
        round(original_width * scale)
    )

    new_height = int(
        round(original_height * scale)
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR,
    )

    resized_mask = mask.resize(
        (new_width, new_height),
        Image.Resampling.NEAREST,
    )

    # --------------------------------------------------------
    # Create 512 x 512 canvases
    # --------------------------------------------------------

    image_canvas = Image.new(
        "L",
        (target_size, target_size),
        color=0,
    )

    mask_canvas = Image.new(
        "L",
        (target_size, target_size),
        color=0,
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    left = (
        target_size - new_width
    ) // 2

    top = (
        target_size - new_height
    ) // 2

    image_canvas.paste(
        resized_image,
        (left, top),
    )

    mask_canvas.paste(
        resized_mask,
        (left, top),
    )

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "resized_width": new_width,
        "resized_height": new_height,
        "pad_left": left,
        "pad_top": top,
    }

    return (
        image_canvas,
        mask_canvas,
        metadata,
    )


# ============================================================
# TN3K DATASET
# ============================================================

class TN3KDataset(Dataset):

    def __init__(
        self,
        split,
        image_size=IMAGE_SIZE,
        augmentation=None,
    ):

        """
        Parameters
        ----------
        split:
            "train"
            "validation"
            "test"

        image_size:
            final square tensor size.
        """

        valid_splits = {
            "train",
            "validation",
            "test",
        }

        if split not in valid_splits:

            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of "
                f"{sorted(valid_splits)}"
            )

        self.split = split
        self.image_size = image_size
        self.augmentation = augmentation

        if (
           self.augmentation is not None
            and self.split != "train"
          ):
          raise ValueError(
             "TN3K augmentation is allowed only "
             "for the training split."
          )
        

        self.rows = []

        # ----------------------------------------------------
        # Read index
        # ----------------------------------------------------

        with open(
            INDEX_CSV,
            "r",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row["Split"] == split:
                    self.rows.append(row)

        if len(self.rows) == 0:

            raise RuntimeError(
                f"No samples found for split: "
                f"{split}"
            )

        print(
            f"TN3KDataset("
            f"split='{split}') "
            f"loaded {len(self.rows)} samples"
        )


    def __len__(self):

        return len(self.rows)


    def __getitem__(self, index):

        row = self.rows[index]

        # ----------------------------------------------------
        # Paths
        # ----------------------------------------------------

        image_path = (
            THYROID_ROOT
            / row["ImagePath"]
        )

        mask_path = (
            THYROID_ROOT
            / row["MaskPath"]
        )

        # ----------------------------------------------------
        # Load grayscale ultrasound
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("L")

        # ----------------------------------------------------
        # Load raw JPEG mask
        # ----------------------------------------------------

        raw_mask = Image.open(
            mask_path
        ).convert("L")

        raw_mask_array = np.array(
            raw_mask,
            dtype=np.uint8,
        )

        # ----------------------------------------------------
        # Convert JPEG mask to true binary mask
        # ----------------------------------------------------

        binary_mask_array = (
            raw_mask_array >= MASK_THRESHOLD
        ).astype(np.uint8) * 255

        binary_mask = Image.fromarray(
            binary_mask_array,
            mode="L",
        )

        # ----------------------------------------------------
        # Resize + pad
        # ----------------------------------------------------

        (
            image,
            binary_mask,
            geometry,
        ) = resize_and_pad(
            image,
            binary_mask,
            target_size=self.image_size,
        )

        # ----------------------------------------------------
        # Convert image to NumPy
        # ----------------------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32,
        )

        # Scale grayscale pixels:
        #
        # 0-255 -> 0-1
        #
        image_array = (
            image_array / 255.0
        )

        # ----------------------------------------------------
        # Convert mask to NumPy
        # ----------------------------------------------------

        mask_array = np.array(
            binary_mask,
            dtype=np.uint8,
        )

        mask_array = (
            mask_array >= 128
        ).astype(np.float32)

        # ----------------------------------------------------
        # Add channel dimension
        #
        # H x W
        # ->
        # 1 x H x W
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        mask_array = np.expand_dims(
            mask_array,
            axis=0,
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image_array
        ).float()

        mask_tensor = torch.from_numpy(
            mask_array
        ).float()

        augmentation_metadata = {
    "affine_applied": False,
    "angle": 0.0,
    "translate_x": 0,
    "translate_y": 0,
    "scale": 1.0,

    "brightness_applied": False,
    "brightness_factor": 1.0,

    "contrast_applied": False,
    "contrast_factor": 1.0,
}

        if self.augmentation is not None:

         (
           image_tensor,
           mask_tensor,
           augmentation_metadata,
         ) = self.augmentation(
           image_tensor,
           mask_tensor,
         )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {
            "image": image_tensor,
            "mask": mask_tensor,

            "sample_id": row["SampleId"],
            "original_id": row["OriginalId"],

            "split": row["Split"],

            "nodule_area_fraction": float(
                row["NoduleAreaFraction"]
            ),

            "nodule_size_group": (
                row["NoduleSizeGroup"]
            ),

            "geometry": geometry,

            "image_path": str(
                image_path
            ),

            "mask_path": str(
                mask_path
            ),
            "augmentation": augmentation_metadata,
        }


# ============================================================
# SIMPLE DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TN3K DATASET BASIC TEST")
    print("=" * 70)

    train_dataset = TN3KDataset(
        split="train",
        image_size=512,
    )

    validation_dataset = TN3KDataset(
        split="validation",
        image_size=512,
    )

    print()
    print(
        f"Train length: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation length: "
        f"{len(validation_dataset)}"
    )

    sample = train_dataset[0]

    print()
    print("FIRST TRAINING SAMPLE")
    print("-" * 70)

    print(
        f"Sample ID: "
        f"{sample['sample_id']}"
    )

    print(
        f"Image tensor shape: "
        f"{tuple(sample['image'].shape)}"
    )

    print(
        f"Mask tensor shape: "
        f"{tuple(sample['mask'].shape)}"
    )

    print(
        f"Image minimum: "
        f"{sample['image'].min().item():.4f}"
    )

    print(
        f"Image maximum: "
        f"{sample['image'].max().item():.4f}"
    )

    print(
        f"Mask unique values: "
        f"{torch.unique(sample['mask']).tolist()}"
    )

    print(
        f"Nodule group: "
        f"{sample['nodule_size_group']}"
    )

    print(
        f"Original nodule area: "
        f"{sample['nodule_area_fraction'] * 100:.4f}%"
    )

    print()
    print("Geometry:")

    for key, value in (
        sample["geometry"].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()
    print("=" * 70)
    print("BASIC DATASET TEST COMPLETE")
    print("=" * 70)