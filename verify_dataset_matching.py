from collections import Counter
from pathlib import Path
import csv


DICOM_FOLDER = Path("SIIM_TRAIN_TEST") / "dicom-images-train"
CSV_PATH = Path("SIIM_TRAIN_TEST") / "train-rle.csv"


# Confirm that the required dataset locations exist
if not DICOM_FOLDER.exists():
    raise FileNotFoundError(
        f"DICOM folder was not found: {DICOM_FOLDER.resolve()}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Annotation CSV was not found: {CSV_PATH.resolve()}"
    )


# Find every DICOM file inside the nested folders
dicom_files = list(DICOM_FOLDER.rglob("*.dcm"))

# Count filenames without the .dcm extension
dicom_id_counts = Counter(path.stem for path in dicom_files)
dicom_ids = set(dicom_id_counts)

# Detect repeated DICOM filenames, if any
duplicate_dicom_ids = {
    image_id
    for image_id, count in dicom_id_counts.items()
    if count > 1
}


# Read every unique ImageId from the annotation CSV
csv_ids = set()

with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)

    for row_number, row in enumerate(reader, start=2):
        if len(row) != 2:
            raise ValueError(
                f"CSV row {row_number} should contain 2 columns, "
                f"but found {len(row)}"
            )

        csv_ids.add(row[0].strip())


# Compare the two collections
csv_without_dicom = sorted(csv_ids - dicom_ids)
dicom_without_csv = sorted(dicom_ids - csv_ids)


print("Dataset matching summary")
print("------------------------")
print(f"DICOM files found: {len(dicom_files)}")
print(f"Unique DICOM IDs: {len(dicom_ids)}")
print(f"Unique CSV ImageIds: {len(csv_ids)}")
print(f"Duplicate DICOM IDs: {len(duplicate_dicom_ids)}")
print(f"CSV IDs without a DICOM file: {len(csv_without_dicom)}")
print(f"DICOM IDs without a CSV entry: {len(dicom_without_csv)}")


# Display a few unmatched IDs to help investigate them
if csv_without_dicom:
    print("\nFirst CSV IDs without DICOM files:")
    for image_id in csv_without_dicom[:5]:
        print(f"  {image_id}")

if dicom_without_csv:
    print("\nFirst DICOM IDs without CSV entries:")
    for image_id in dicom_without_csv[:5]:
        print(f"  {image_id}")