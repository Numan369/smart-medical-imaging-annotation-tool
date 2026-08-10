from pathlib import Path

import matplotlib.pyplot as plt
import pydicom


# Folder containing the training DICOM images
TRAIN_FOLDER = Path("SIIM_TRAIN_TEST") / "dicom-images-train"


# Search through all nested folders for DICOM files
dicom_files = list(TRAIN_FOLDER.rglob("*.dcm"))

if not dicom_files:
    raise FileNotFoundError(
        f"No DICOM files were found inside: {TRAIN_FOLDER.resolve()}"
    )


# Select the first DICOM file found
dicom_path = dicom_files[0]

# Read the DICOM file
dicom_data = pydicom.dcmread(dicom_path)

# Extract its image pixels
image = dicom_data.pixel_array


# MONOCHROME1 means lower values should appear white.
# We invert such images so they display correctly.
if dicom_data.PhotometricInterpretation == "MONOCHROME1":
    image = image.max() - image


# Display the X-ray
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="gray")
plt.title(f"First training X-ray\n{dicom_path.name}")
plt.axis("off")
plt.tight_layout()
plt.show()


# Print only non-sensitive technical information
print("DICOM loaded successfully")
print(f"File: {dicom_path}")
print(f"Image dimensions: {image.shape}")
print(f"Photometric interpretation: {dicom_data.PhotometricInterpretation}")