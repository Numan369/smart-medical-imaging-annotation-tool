from collections import defaultdict
from pathlib import Path
import csv


DICOM_FOLDER = Path("SIIM_TRAIN_TEST") / "dicom-images-train"
ANNOTATION_PATH = Path("SIIM_TRAIN_TEST") / "train-rle.csv"

OUTPUT_FOLDER = Path("prepared_data")
OUTPUT_PATH = OUTPUT_FOLDER / "dataset_index.csv"


# Check the required dataset locations
if not DICOM_FOLDER.exists():
    raise FileNotFoundError(
        f"DICOM folder was not found: {DICOM_FOLDER.resolve()}"
    )

if not ANNOTATION_PATH.exists():
    raise FileNotFoundError(
        f"Annotation CSV was not found: {ANNOTATION_PATH.resolve()}"
    )


# Build a lookup table:
# ImageId -> DICOM file path
dicom_paths = {}

for dicom_path in DICOM_FOLDER.rglob("*.dcm"):
    image_id = dicom_path.stem

    if image_id in dicom_paths:
        raise ValueError(
            f"Duplicate DICOM ID found: {image_id}"
        )

    dicom_paths[image_id] = dicom_path


# Group all CSV annotations by ImageId
annotations_by_id = defaultdict(list)

with ANNOTATION_PATH.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as csv_file:

    reader = csv.DictReader(
    csv_file,
    skipinitialspace=True,
)

    expected_columns = {"ImageId", "EncodedPixels"}

    if set(reader.fieldnames or []) != expected_columns:
        raise ValueError(
            f"Unexpected CSV columns: {reader.fieldnames}"
        )

    for row_number, row in enumerate(reader, start=2):
        image_id = row["ImageId"].strip()
        encoded_pixels = row["EncodedPixels"].strip()

        if not image_id:
            raise ValueError(
                f"Missing ImageId on CSV row {row_number}"
            )

        if not encoded_pixels:
            raise ValueError(
                f"Missing EncodedPixels on CSV row {row_number}"
            )

        annotations_by_id[image_id].append(encoded_pixels)


# Create the folder for derived project files
OUTPUT_FOLDER.mkdir(exist_ok=True)


positive_count = 0
negative_count = 0
multiple_mask_count = 0


# Write one index row for each labelled image
with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
    newline="",
) as output_file:

    fieldnames = [
        "ImageId",
        "DICOMPath",
        "HasPneumothorax",
        "MaskCount",
    ]

    writer = csv.DictWriter(
        output_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for image_id in sorted(annotations_by_id):
        if image_id not in dicom_paths:
            raise FileNotFoundError(
                f"No DICOM file was found for CSV ImageId: {image_id}"
            )

        annotation_rows = annotations_by_id[image_id]

        positive_rles = [
            value
            for value in annotation_rows
            if value != "-1"
        ]

        contains_negative_marker = "-1" in annotation_rows

        # An image should not contain both -1 and a positive mask
        if positive_rles and contains_negative_marker:
            raise ValueError(
                f"Image {image_id} has both positive and negative labels"
            )

        has_pneumothorax = bool(positive_rles)
        mask_count = len(positive_rles)

        if has_pneumothorax:
            positive_count += 1
        else:
            negative_count += 1

        if mask_count > 1:
            multiple_mask_count += 1

        writer.writerow(
            {
                "ImageId": image_id,
                "DICOMPath": dicom_paths[image_id].as_posix(),
                "HasPneumothorax": int(has_pneumothorax),
                "MaskCount": mask_count,
            }
        )


total_indexed = positive_count + negative_count


print("Dataset index created successfully")
print("----------------------------------")
print(f"Output file: {OUTPUT_PATH.resolve()}")
print(f"Total indexed images: {total_indexed}")
print(f"Positive images: {positive_count}")
print(f"Negative images: {negative_count}")
print(f"Images with multiple masks: {multiple_mask_count}")
print(f"Unlabelled DICOMs excluded: {len(dicom_paths) - total_indexed}")