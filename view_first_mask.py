from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import pydicom


DICOM_FOLDER = Path("SIIM_TRAIN_TEST") / "dicom-images-train"
CSV_PATH = Path("SIIM_TRAIN_TEST") / "train-rle.csv"


def decode_siim_rle(encoded_pixels, rows, columns):
    """Convert one SIIM relative-RLE annotation into a binary mask."""

    values = np.asarray(
        [int(value) for value in encoded_pixels.split()],
        dtype=np.int64,
    )

    if len(values) % 2 != 0:
        raise ValueError("The RLE must contain gap-length pairs.")

    gaps = values[0::2]
    lengths = values[1::2]

    flat_mask = np.zeros(rows * columns, dtype=np.uint8)
    current_position = 0

    for gap, length in zip(gaps, lengths):
        # Move forward from the end of the previous annotated run
        current_position += gap
        run_end = current_position + length

        if run_end > flat_mask.size:
            raise ValueError("The decoded mask exceeds the image dimensions.")

        flat_mask[current_position:run_end] = 1
        current_position = run_end

    # SIIM stores pixels in column-first order
    return flat_mask.reshape((columns, rows)).T


# Read and group every positive RLE row by image ID
positive_annotations = defaultdict(list)

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip the header

    for row in reader:
        image_id = row[0].strip()
        encoded_pixels = row[1].strip()

        if encoded_pixels != "-1":
            positive_annotations[image_id].append(encoded_pixels)


if not positive_annotations:
    raise ValueError("No positive annotations were found in the CSV.")


# Select the first positive image
image_id = next(iter(positive_annotations))
rle_annotations = positive_annotations[image_id]


# Find its corresponding DICOM file
matching_files = list(DICOM_FOLDER.rglob(f"{image_id}.dcm"))

if len(matching_files) != 1:
    raise FileNotFoundError(
        f"Expected one DICOM for {image_id}, but found {len(matching_files)}"
    )

dicom_path = matching_files[0]
dicom_data = pydicom.dcmread(dicom_path)
image = dicom_data.pixel_array

if dicom_data.PhotometricInterpretation == "MONOCHROME1":
    image = image.max() - image


rows, columns = image.shape
combined_mask = np.zeros((rows, columns), dtype=np.uint8)


# Decode and combine every mask row belonging to this image
for encoded_pixels in rle_annotations:
    decoded_mask = decode_siim_rle(
        encoded_pixels,
        rows,
        columns,
    )
    combined_mask = np.maximum(combined_mask, decoded_mask)


print("Positive annotation loaded successfully")
print(f"Image ID: {image_id}")
print(f"Image dimensions: {image.shape}")
print(f"Mask rows combined: {len(rle_annotations)}")
print(f"Annotated pixels: {int(combined_mask.sum())}")


# Hide zero-valued mask pixels in the overlay
visible_mask = np.ma.masked_where(
    combined_mask == 0,
    combined_mask,
)


figure, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original DICOM")

axes[1].imshow(combined_mask, cmap="gray")
axes[1].set_title("Decoded binary mask")

axes[2].imshow(image, cmap="gray")
axes[2].imshow(visible_mask, cmap="autumn", alpha=0.45)
axes[2].set_title("Mask overlay")

for axis in axes:
    axis.axis("off")

figure.suptitle(image_id, fontsize=9)
plt.tight_layout()
plt.show()