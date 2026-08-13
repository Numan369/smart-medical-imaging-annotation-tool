from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from pneumothorax_dataloaders import create_balanced_training_sampler
from pneumothorax_dataset import PneumothoraxDataset


IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_AUGMENTED_EXAMPLES = 5
RANDOM_SEED = 42

OUTPUT_DIRECTORY = Path("augmentation_examples")
OUTPUT_PATH = OUTPUT_DIRECTORY / "training_augmentation_check.png"


def random_uniform(generator, minimum, maximum):
    """Draw one reproducible floating-point value."""

    value = torch.rand(1, generator=generator).item()
    return minimum + value * (maximum - minimum)


class PairedTrainingAugmentation:
    """Apply conservative training changes to an X-ray and its mask."""

    def __init__(
        self,
        horizontal_flip_probability=0.5,
        maximum_rotation_degrees=5.0,
        maximum_translation_fraction=0.02,
        minimum_scale=0.95,
        maximum_scale=1.05,
        minimum_brightness=0.95,
        maximum_brightness=1.05,
        minimum_contrast=0.90,
        maximum_contrast=1.10,
    ):
        self.horizontal_flip_probability = (
            horizontal_flip_probability
        )
        self.maximum_rotation_degrees = maximum_rotation_degrees
        self.maximum_translation_fraction = (
            maximum_translation_fraction
        )
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.minimum_brightness = minimum_brightness
        self.maximum_brightness = maximum_brightness
        self.minimum_contrast = minimum_contrast
        self.maximum_contrast = maximum_contrast

    def __call__(self, image, mask, generator=None):
        if generator is None:
            generator = torch.default_generator

        if image.ndim != 3 or mask.ndim != 3:
            raise ValueError(
                "Expected image and mask shaped "
                "(channels, height, width)."
            )

        if image.shape != mask.shape:
            raise ValueError(
                "The image and mask must have identical shapes."
            )

        if image.shape[0] != 1:
            raise ValueError(
                "Expected a one-channel grayscale X-ray."
            )

        height, width = image.shape[-2:]
        maximum_horizontal_shift = round(
            width * self.maximum_translation_fraction
        )
        maximum_vertical_shift = round(
            height * self.maximum_translation_fraction
        )

        flipped = (
            torch.rand(1, generator=generator).item()
            < self.horizontal_flip_probability
        )
        angle = random_uniform(
            generator,
            -self.maximum_rotation_degrees,
            self.maximum_rotation_degrees,
        )
        horizontal_shift = int(
            torch.randint(
                -maximum_horizontal_shift,
                maximum_horizontal_shift + 1,
                (1,),
                generator=generator,
            ).item()
        )
        vertical_shift = int(
            torch.randint(
                -maximum_vertical_shift,
                maximum_vertical_shift + 1,
                (1,),
                generator=generator,
            ).item()
        )
        scale = random_uniform(
            generator,
            self.minimum_scale,
            self.maximum_scale,
        )
        brightness = random_uniform(
            generator,
            self.minimum_brightness,
            self.maximum_brightness,
        )
        contrast = random_uniform(
            generator,
            self.minimum_contrast,
            self.maximum_contrast,
        )

        if flipped:
            image = transform_functional.hflip(image)
            mask = transform_functional.hflip(mask)

        affine_arguments = {
            "angle": angle,
            "translate": [horizontal_shift, vertical_shift],
            "scale": scale,
            "shear": [0.0, 0.0],
            "fill": 0.0,
        }

        image = transform_functional.affine(
            image,
            interpolation=InterpolationMode.BILINEAR,
            **affine_arguments,
        )
        mask = transform_functional.affine(
            mask,
            interpolation=InterpolationMode.NEAREST,
            **affine_arguments,
        )

        image = transform_functional.adjust_brightness(
            image,
            brightness,
        )
        image = transform_functional.adjust_contrast(
            image,
            contrast,
        )

        image = image.clamp(0.0, 1.0)
        mask = (mask >= 0.5).to(torch.float32)

        parameters = {
            "flipped": flipped,
            "angle": angle,
            "horizontal_shift": horizontal_shift,
            "vertical_shift": vertical_shift,
            "scale": scale,
            "brightness": brightness,
            "contrast": contrast,
        }

        return image, mask, parameters


class AugmentedTrainingDataset(Dataset):
    """Add paired augmentation to training samples only."""

    def __init__(self, base_dataset, augmentation):
        if base_dataset.split != "train":
            raise ValueError(
                "Augmentation may only wrap the training split."
            )

        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.rows = base_dataset.rows
        self.split = base_dataset.split

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        image, mask, _ = self.augmentation(
            sample["image"],
            sample["mask"],
        )

        augmented_sample = dict(sample)
        augmented_sample["image"] = image
        augmented_sample["mask"] = mask
        return augmented_sample


def first_positive_index(dataset):
    """Find one known-positive item without loading every DICOM."""

    for index, row in enumerate(dataset.rows):
        if int(row["HasPneumothorax"]) == 1:
            return index

    raise ValueError(
        f"No positive images were found in the {dataset.split} split."
    )


def validate_image_and_mask(image, mask, description):
    """Check tensor shapes, ranges, and mask values."""

    expected_shape = (1, IMAGE_SIZE, IMAGE_SIZE)

    if tuple(image.shape) != expected_shape:
        raise ValueError(
            f"Unexpected {description} image shape: "
            f"{tuple(image.shape)}"
        )

    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"Unexpected {description} mask shape: "
            f"{tuple(mask.shape)}"
        )

    if not torch.isfinite(image).all():
        raise ValueError(
            f"The {description} image contains non-finite values."
        )

    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError(
            f"The {description} image is outside the 0-1 range."
        )

    mask_values = set(torch.unique(mask).tolist())

    if not mask_values.issubset({0.0, 1.0}):
        raise ValueError(
            f"The {description} mask is not binary: "
            f"{sorted(mask_values)}"
        )


def make_overlay(image, mask):
    """Create a red mask overlay for visual alignment checking."""

    grayscale = image.squeeze(0).numpy()
    binary_mask = mask.squeeze(0).numpy().astype(bool)

    overlay = torch.from_numpy(grayscale).unsqueeze(-1).repeat(
        1,
        1,
        3,
    ).numpy()
    overlay[binary_mask, 0] = 1.0
    overlay[binary_mask, 1] *= 0.25
    overlay[binary_mask, 2] *= 0.25
    return overlay


def save_comparison(original_image, original_mask, variants):
    """Save original and augmented image-mask pairs for inspection."""

    rows = 1 + len(variants)
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(12, 4 * rows),
    )

    examples = [
        ("Original training sample", original_image, original_mask)
    ]
    examples.extend(
        (
            f"Augmented variant {index}",
            image,
            mask,
        )
        for index, (image, mask, _) in enumerate(variants, start=1)
    )

    for row_index, (title, image, mask) in enumerate(examples):
        axes[row_index, 0].imshow(
            image.squeeze(0).numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row_index, 0].set_title(f"{title}: X-ray")

        axes[row_index, 1].imshow(
            mask.squeeze(0).numpy(),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[row_index, 1].set_title("Paired mask")

        axes[row_index, 2].imshow(make_overlay(image, mask))
        axes[row_index, 2].set_title("Alignment overlay")

        for column_index in range(3):
            axes[row_index, column_index].axis("off")

    figure.suptitle(
        "Training-only pneumothorax augmentation check",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)


def validate_training_loader(training_dataset):
    """Confirm augmented samples work with balanced batch loading."""

    sampler = create_balanced_training_sampler(training_dataset)
    loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    if tuple(images.shape) != (
        BATCH_SIZE,
        1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ):
        raise ValueError(
            f"Unexpected augmented batch shape: {tuple(images.shape)}"
        )

    if tuple(masks.shape) != tuple(images.shape):
        raise ValueError(
            "Augmented batch images and masks have different shapes."
        )

    print("\nAugmented training batch")
    print(f"Images: {tuple(images.shape)}")
    print(f"Masks: {tuple(masks.shape)}")
    print(
        "Positive images: "
        f"{int(labels.sum().item())} / {len(labels)}"
    )
    print(f"Annotated pixels: {int(masks.sum().item()):,}")


def main():
    torch.manual_seed(RANDOM_SEED)

    print("Pneumothorax training-augmentation check")
    print("----------------------------------------")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print("Augmented split: training only")
    print("Validation: loaded without augmentation")
    print("Test split: not created or accessed")
    print("No vertical flip, crop, or elastic deformation")

    training_dataset = PneumothoraxDataset(
        split="train",
        image_size=IMAGE_SIZE,
    )
    validation_dataset = PneumothoraxDataset(
        split="validation",
        image_size=IMAGE_SIZE,
    )

    augmentation = PairedTrainingAugmentation()
    augmented_training_dataset = AugmentedTrainingDataset(
        training_dataset,
        augmentation,
    )

    training_index = first_positive_index(training_dataset)
    original_sample = training_dataset[training_index]
    original_image = original_sample["image"]
    original_mask = original_sample["mask"]
    validate_image_and_mask(
        original_image,
        original_mask,
        "original training",
    )

    original_pixels = int(original_mask.sum().item())

    if original_pixels <= 0:
        raise ValueError(
            "The selected positive training mask is unexpectedly empty."
        )

    print("\nPaired training variants")
    print(
        "Variant | Flip | Rotation | Shift (x,y) | Scale | "
        "Brightness | Contrast | Mask pixels"
    )

    variants = []

    for variant_number in range(1, NUM_AUGMENTED_EXAMPLES + 1):
        generator = torch.Generator().manual_seed(
            RANDOM_SEED + variant_number
        )
        image, mask, parameters = augmentation(
            original_image.clone(),
            original_mask.clone(),
            generator=generator,
        )
        validate_image_and_mask(
            image,
            mask,
            f"augmented variant {variant_number}",
        )

        mask_pixels = int(mask.sum().item())

        if mask_pixels <= 0:
            raise ValueError(
                "A positive mask became empty after conservative "
                f"augmentation in variant {variant_number}."
            )

        variants.append((image, mask, parameters))

        print(
            f"{variant_number:7d} | "
            f"{str(parameters['flipped']):4s} | "
            f"{parameters['angle']:+7.2f} deg | "
            f"({parameters['horizontal_shift']:+3d},"
            f"{parameters['vertical_shift']:+3d}) | "
            f"{parameters['scale']:.3f} | "
            f"{parameters['brightness']:.3f} | "
            f"{parameters['contrast']:.3f} | "
            f"{mask_pixels:,}"
        )

    validation_index = first_positive_index(validation_dataset)
    validation_first = validation_dataset[validation_index]
    validation_second = validation_dataset[validation_index]

    validation_unchanged = (
        torch.equal(
            validation_first["image"],
            validation_second["image"],
        )
        and torch.equal(
            validation_first["mask"],
            validation_second["mask"],
        )
    )

    if not validation_unchanged:
        raise ValueError(
            "Validation loading is not deterministic and unchanged."
        )

    validate_training_loader(augmented_training_dataset)
    save_comparison(original_image, original_mask, variants)

    print("\nValidation isolation check")
    print("Same validation image loaded twice: identical")
    print("Validation augmentation applied: False")

    print("\n## Training augmentation check passed")
    print(
        "Geometric changes were paired between each X-ray and mask."
    )
    print("Masks remained binary and positive masks remained non-empty.")
    print(f"Saved comparison: {OUTPUT_PATH.resolve()}")
    print("No training was performed and no checkpoint was changed.")


if __name__ == "__main__":
    main()
