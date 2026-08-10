from collections import Counter
from pathlib import Path
import csv


CSV_PATH = Path("SIIM_TRAIN_TEST") / "train-rle.csv"


if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Annotation CSV was not found: {CSV_PATH.resolve()}"
    )


positive_images = set()
negative_images = set()
positive_row_counts = Counter()
total_rows = 0


with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
    reader = csv.reader(csv_file)

    # Read and display the column names
    header = next(reader)
    header = [column.strip() for column in header]

    print(f"CSV columns: {header}")

    for row_number, row in enumerate(reader, start=2):
        if len(row) != 2:
            raise ValueError(
                f"Row {row_number} should contain 2 columns, but found {len(row)}"
            )

        image_id = row[0].strip()
        encoded_pixels = row[1].strip()
        total_rows += 1

        if encoded_pixels == "-1":
            negative_images.add(image_id)
        else:
            positive_images.add(image_id)
            positive_row_counts[image_id] += 1


all_images = positive_images | negative_images

images_with_multiple_masks = sum(
    count > 1 for count in positive_row_counts.values()
)


print("\nAnnotation summary")
print("------------------")
print(f"Total CSV rows: {total_rows}")
print(f"Unique images: {len(all_images)}")
print(f"Positive images: {len(positive_images)}")
print(f"Negative images: {len(negative_images)}")
print(f"Images with multiple mask rows: {images_with_multiple_masks}")