# Pneumothorax AI Inference Bundle

Research-prototype inference for the Medical Images Annotation Tool.
Every generated mask requires human review.

## Frozen configuration

- Input size: 512 x 512 grayscale
- Probability threshold: 0.15
- Minimum connected region: 112 pixels
- Connectivity: eight-connected pixels
- Checkpoint: epoch-10 best model
- Output mask: resized to the original image dimensions

Do not change the threshold or component filter using test results.

## Supported inputs

- DICOM: .dcm and .dicom
- Raster: .png, .jpg and .jpeg

The model was trained and formally evaluated using DICOM X-rays.
PNG normally preserves normalized X-ray information closely.
JPEG is lossy and can change or fragment small predicted regions.
Only single-frame two-dimensional DICOM X-rays are supported.

## Research disclaimer

AI-generated suggestions require human review and must not be
treated as a medical diagnosis.

The model is not intended for autonomous clinical diagnosis,
clinical decision-making, or unreviewed annotation.

## Windows CPU setup

Open PowerShell inside the extracted bundle directory.

Create and activate a Python environment:

    py -3.12 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip

Install CPU PyTorch:

    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

Install remaining packages:

    python -m pip install -r requirements.txt

## Run inference

DICOM example:

    python pneumothorax_inference.py image.dcm --checkpoint pneumothorax_512_v4a_fresh_45_55_best.pth --output-directory output --device cpu

PNG example:

    python pneumothorax_inference.py image.png --checkpoint pneumothorax_512_v4a_fresh_45_55_best.pth --output-directory output --device cpu

## Generated files

- Normalized grayscale preview
- Original-size binary mask
- Cyan overlay preview
- JSON result summary

For newly uploaded images, referenceMetrics must remain null.
Dice, IoU, precision and recall require an expert reference mask.

## Finding messages

possible-region-detected:
Possible pneumothorax region detected

no-region-detected:
No pneumothorax region detected by AI

A successful empty result is not a processing failure.

## Final frozen test results

- Test images: 1,205
- Positive Dice: 0.475908
- Sensitivity: 90.26%
- Specificity: 70.47%
- Mean positive IoU: 0.358062
- Mean positive pixel precision: 0.520447
- Mean positive pixel recall: 0.534611

These are dataset-level research results and must not be displayed
as per-image accuracy or diagnostic confidence.
