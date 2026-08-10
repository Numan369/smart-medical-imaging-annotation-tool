from collections import defaultdict
from pathlib import Path
import csv

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SPLITS_PATH = Path("prepared_data") / "dataset_splits.csv"
ANNOTATION_PATH = Path("SIIM_TRAIN_TEST") / "train-rle.csv"

IMAGE_SIZE = 256


def decode_siim_rle(encoded_pixels, rows, columns):
    """Decode one SIIM relative RLE annotation into a binary mask."""

    values = np.asarray(
        [int(value) for value in encoded_pixels.split()],
        dtype=np.int64,
    )

    if len(values) % 2 != 0:
        raise ValueError(
            "The RLE must contain offset-length pairs."
        )

    offsets = values[0::2]
    lengths = values[1::2]

    if np.any(offsets < 0):
        raise ValueError(
            "RLE offsets cannot be negative."
        )

    if np.any(lengths <= 0):
        raise ValueError(
            "RLE lengths must be positive."
        )

    flat_mask = np.zeros(
        rows * columns,
        dtype=np.uint8,
    )

    current_position = 0

    for offset, length in zip(offsets, lengths):
        current_position += offset
        run_end = current_position + length

        if run_end > flat_mask.size:
            raise ValueError(
                "The decoded mask exceeds the image dimensions."
            )

        flat_mask[current_position:run_end] = 1
        current_position = run_end

    # Convert SIIM's flattened column-wise mask
    # back into normal image orientation.
    return flat_mask.reshape(
        (columns, rows)
    ).T


def normalise_dicom_image(dicom_data):
    """Convert a DICOM pixel array into values between 0 and 1."""

    image = dicom_data.pixel_array.astype(np.float32)

    slope = float(
        getattr(dicom_data, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(dicom_data, "RescaleIntercept", 0.0)
    )

    image = image * slope + intercept

    photometric = str(
        getattr(dicom_data, "PhotometricInterpretation", "")
    ).upper()

    # MONOCHROME1 stores brighter pixels using smaller values.
    if photometric == "MONOCHROME1":
        image = image.max() + image.min() - image

    minimum = float(image.min())
    maximum = float(image.max())

    if maximum <= minimum:
        raise ValueError(
            "The DICOM image contains no usable intensity range."
        )

    image = (image - minimum) / (maximum - minimum)

    return image.astype(np.float32)


def resize_image_and_mask(image, mask, image_size):
    """Resize the image and mask while preserving binary mask values."""

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    mask_tensor = torch.from_numpy(
        mask.astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    mask_tensor = F.interpolate(
        mask_tensor,
        size=(image_size, image_size),
        mode="nearest",
    )

    return image_tensor.squeeze(0), mask_tensor.squeeze(0)


class PneumothoraxDataset(Dataset):
    """Load SIIM DICOM images and their segmentation masks."""

    def __init__(
        self,
        split,
        image_size=IMAGE_SIZE,
    ):
        valid_splits = {
            "train",
            "validation",
            "test",
        }

        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Choose from {sorted(valid_splits)}."
            )

        if not SPLITS_PATH.exists():
            raise FileNotFoundError(
                f"Dataset splits file was not found: "
                f"{SPLITS_PATH.resolve()}"
            )

        if not ANNOTATION_PATH.exists():
            raise FileNotFoundError(
                f"Annotation CSV was not found: "
                f"{ANNOTATION_PATH.resolve()}"
            )

        self.split = split
        self.image_size = image_size
        self.rows = []
        self.annotations_by_id = defaultdict(list)

        self._load_split_rows()
        self._load_annotations()
        self._validate_rows()

    def _load_split_rows(self):
        with SPLITS_PATH.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as split_file:

            reader = csv.DictReader(split_file)

            expected_columns = {
                "ImageId",
                "DICOMPath",
                "HasPneumothorax",
                "MaskCount",
                "Split",
            }

            if set(reader.fieldnames or []) != expected_columns:
                raise ValueError(
                    f"Unexpected split columns: "
                    f"{reader.fieldnames}"
                )

            for row_number, row in enumerate(reader, start=2):
                if row["Split"] != self.split:
                    continue

                label = int(row["HasPneumothorax"])

                if label not in (0, 1):
                    raise ValueError(
                        f"Invalid label on split row "
                        f"{row_number}: {label}"
                    )

                dicom_path = Path(row["DICOMPath"])

                if not dicom_path.exists():
                    raise FileNotFoundError(
                        f"DICOM file was not found: {dicom_path}"
                    )

                self.rows.append(
                    {
                        "ImageId": row["ImageId"],
                        "DICOMPath": dicom_path,
                        "HasPneumothorax": label,
                    }
                )

    def _load_annotations(self):
        with ANNOTATION_PATH.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as annotation_file:

            reader = csv.DictReader(
                annotation_file,
                skipinitialspace=True,
            )

            expected_columns = {
                "ImageId",
                "EncodedPixels",
            }

            if set(reader.fieldnames or []) != expected_columns:
                raise ValueError(
                    f"Unexpected annotation columns: "
                    f"{reader.fieldnames}"
                )

            for row in reader:
                image_id = row["ImageId"].strip()
                encoded_pixels = row["EncodedPixels"].strip()

                self.annotations_by_id[image_id].append(
                    encoded_pixels
                )

    def _validate_rows(self):
        if not self.rows:
            raise ValueError(
                f"No images were found for split: {self.split}"
            )

        image_ids = [
            row["ImageId"]
            for row in self.rows
        ]

        if len(image_ids) != len(set(image_ids)):
            raise ValueError(
                f"Duplicate image IDs found in {self.split} split."
            )

        for row in self.rows:
            image_id = row["ImageId"]
            label = row["HasPneumothorax"]

            if image_id not in self.annotations_by_id:
                raise ValueError(
                    f"No annotation row found for: {image_id}"
                )

            positive_rles = [
                value
                for value in self.annotations_by_id[image_id]
                if value != "-1"
            ]

            annotation_label = int(bool(positive_rles))

            if annotation_label != label:
                raise ValueError(
                    f"Label mismatch for image: {image_id}"
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image_id = row["ImageId"]
        dicom_path = row["DICOMPath"]
        label = row["HasPneumothorax"]

        dicom_data = pydicom.dcmread(dicom_path)
        image = normalise_dicom_image(dicom_data)

        rows, columns = image.shape

        combined_mask = np.zeros(
            (rows, columns),
            dtype=np.uint8,
        )

        positive_rles = [
            value
            for value in self.annotations_by_id[image_id]
            if value != "-1"
        ]

        for encoded_pixels in positive_rles:
            decoded_mask = decode_siim_rle(
                encoded_pixels,
                rows,
                columns,
            )

            combined_mask = np.maximum(
                combined_mask,
                decoded_mask,
            )

        image_tensor, mask_tensor = resize_image_and_mask(
            image,
            combined_mask,
            self.image_size,
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(
                label,
                dtype=torch.float32,
            ),
            "image_id": image_id,
        }


if __name__ == "__main__":
    dataset = PneumothoraxDataset(
        split="train",
        image_size=IMAGE_SIZE,
    )

    positive_index = next(
        index
        for index, row in enumerate(dataset.rows)
        if row["HasPneumothorax"] == 1
    )

    negative_index = next(
        index
        for index, row in enumerate(dataset.rows)
        if row["HasPneumothorax"] == 0
    )

    print("Dataset loader check")
    print("--------------------")
    print(f"Split: {dataset.split}")
    print(f"Total images: {len(dataset)}")
    print(f"Output image size: {IMAGE_SIZE} x {IMAGE_SIZE}")

    for sample_type, sample_index in (
        ("Positive", positive_index),
        ("Negative", negative_index),
    ):
        sample = dataset[sample_index]

        image = sample["image"]
        mask = sample["mask"]

        print(f"\n{sample_type} sample")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Label: {int(sample['label'].item())}")
        print(f"  Image tensor shape: {tuple(image.shape)}")
        print(f"  Mask tensor shape: {tuple(mask.shape)}")
        print(
            f"  Image value range: "
            f"{image.min().item():.4f} to "
            f"{image.max().item():.4f}"
        )
        print(
            f"  Annotated resized pixels: "
            f"{int(mask.sum().item())}"
        )