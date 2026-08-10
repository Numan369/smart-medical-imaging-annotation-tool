from collections import defaultdict
from pathlib import Path
import csv

import pydicom


INDEX_PATH = Path("prepared_data") / "dataset_index.csv"


if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Dataset index was not found: {INDEX_PATH.resolve()}"
    )


# PatientID -> list of image labels belonging to that patient
patient_labels = defaultdict(list)

total_images = 0
positive_images = 0
negative_images = 0


with INDEX_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as index_file:

    reader = csv.DictReader(index_file)

    expected_columns = {
        "ImageId",
        "DICOMPath",
        "HasPneumothorax",
        "MaskCount",
    }

    if set(reader.fieldnames or []) != expected_columns:
        raise ValueError(
            f"Unexpected index columns: {reader.fieldnames}"
        )

    for row_number, row in enumerate(reader, start=2):
        dicom_path = Path(row["DICOMPath"])
        label = int(row["HasPneumothorax"])

        if not dicom_path.exists():
            raise FileNotFoundError(
                f"DICOM file on index row {row_number} was not found: "
                f"{dicom_path.resolve()}"
            )

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label on index row {row_number}: {label}"
            )

        # Read only DICOM metadata, not the large pixel array
        dicom_data = pydicom.dcmread(
            dicom_path,
            stop_before_pixels=True,
        )

        patient_id = str(
            getattr(dicom_data, "PatientID", "")
        ).strip()

        if not patient_id:
            raise ValueError(
                f"PatientID is missing from: {dicom_path}"
            )

        patient_labels[patient_id].append(label)

        total_images += 1

        if label == 1:
            positive_images += 1
        else:
            negative_images += 1


patients_with_multiple_images = sum(
    len(labels) > 1
    for labels in patient_labels.values()
)

patients_with_mixed_labels = sum(
    len(set(labels)) > 1
    for labels in patient_labels.values()
)

positive_patients = sum(
    any(label == 1 for label in labels)
    for labels in patient_labels.values()
)

negative_only_patients = sum(
    all(label == 0 for label in labels)
    for labels in patient_labels.values()
)


print("Patient grouping summary")
print("------------------------")
print(f"Total indexed images: {total_images}")
print(f"Positive images: {positive_images}")
print(f"Negative images: {negative_images}")
print(f"Unique patients: {len(patient_labels)}")
print(
    f"Patients with multiple images: "
    f"{patients_with_multiple_images}"
)
print(
    f"Patients containing positive and negative images: "
    f"{patients_with_mixed_labels}"
)
print(f"Patients with at least one positive image: {positive_patients}")
print(f"Patients with negative images only: {negative_only_patients}")