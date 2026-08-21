import random

import torch
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode


# ============================================================
# TN3K V1 TRAINING AUGMENTATION
# ============================================================

class TN3KTrainAugmentation:
    """
    Conservative augmentation for TN3K V1.

    Input:
        image tensor: [1, H, W], values [0, 1]
        mask tensor:  [1, H, W], values {0, 1}

    Output:
        augmented image
        augmented binary mask
        augmentation metadata

    IMPORTANT:
        All geometric transformations use exactly
        the same random parameters for image and mask.

        Image interpolation:
            BILINEAR

        Mask interpolation:
            NEAREST
    """

    def __init__(
        self,

        affine_probability=0.80,

        max_rotation_degrees=7.0,

        max_translation_fraction=0.03,

        min_scale=0.97,
        max_scale=1.03,

        brightness_probability=0.50,
        min_brightness=0.90,
        max_brightness=1.10,

        contrast_probability=0.50,
        min_contrast=0.90,
        max_contrast=1.10,
    ):

        self.affine_probability = (
            affine_probability
        )

        self.max_rotation_degrees = (
            max_rotation_degrees
        )

        self.max_translation_fraction = (
            max_translation_fraction
        )

        self.min_scale = min_scale
        self.max_scale = max_scale

        self.brightness_probability = (
            brightness_probability
        )

        self.min_brightness = (
            min_brightness
        )

        self.max_brightness = (
            max_brightness
        )

        self.contrast_probability = (
            contrast_probability
        )

        self.min_contrast = (
            min_contrast
        )

        self.max_contrast = (
            max_contrast
        )


    def __call__(
        self,
        image,
        mask,
    ):

        # ====================================================
        # INPUT SAFETY CHECKS
        # ====================================================

        if image.ndim != 3:

            raise ValueError(
                "Expected image shape [C,H,W]. "
                f"Received {tuple(image.shape)}"
            )

        if mask.ndim != 3:

            raise ValueError(
                "Expected mask shape [C,H,W]. "
                f"Received {tuple(mask.shape)}"
            )

        if image.shape != mask.shape:

            raise ValueError(
                "Image and mask shapes must match. "
                f"Image={tuple(image.shape)}, "
                f"Mask={tuple(mask.shape)}"
            )

        height = image.shape[-2]
        width = image.shape[-1]


        # ====================================================
        # AUGMENTATION METADATA
        # ====================================================

        metadata = {

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


        # ====================================================
        # 1. RANDOM AFFINE TRANSFORMATION
        #
        # Includes:
        #
        # rotation
        # translation
        # scaling
        #
        # SAME geometry is applied to:
        #
        # image
        # mask
        # ====================================================

        if (
            random.random()
            < self.affine_probability
        ):

            # ------------------------------------------------
            # Rotation
            # ------------------------------------------------

            angle = random.uniform(
                -self.max_rotation_degrees,
                self.max_rotation_degrees,
            )


            # ------------------------------------------------
            # Translation
            # ------------------------------------------------

            max_dx = int(
                round(
                    width
                    * self.max_translation_fraction
                )
            )

            max_dy = int(
                round(
                    height
                    * self.max_translation_fraction
                )
            )

            translate_x = random.randint(
                -max_dx,
                max_dx,
            )

            translate_y = random.randint(
                -max_dy,
                max_dy,
            )


            # ------------------------------------------------
            # Scaling
            # ------------------------------------------------

            scale = random.uniform(
                self.min_scale,
                self.max_scale,
            )


            # ------------------------------------------------
            # Transform ultrasound IMAGE
            #
            # Ultrasound intensity is continuous,
            # therefore use BILINEAR interpolation.
            # ------------------------------------------------

            image = TF.affine(

                image,

                angle=angle,

                translate=[
                    translate_x,
                    translate_y,
                ],

                scale=scale,

                shear=[
                    0.0,
                    0.0,
                ],

                interpolation=(
                    InterpolationMode.BILINEAR
                ),

                fill=0.0,
            )


            # ------------------------------------------------
            # Transform segmentation MASK
            #
            # Mask contains categorical labels {0,1},
            # therefore use NEAREST interpolation.
            # ------------------------------------------------

            mask = TF.affine(

                mask,

                angle=angle,

                translate=[
                    translate_x,
                    translate_y,
                ],

                scale=scale,

                shear=[
                    0.0,
                    0.0,
                ],

                interpolation=(
                    InterpolationMode.NEAREST
                ),

                fill=0.0,
            )


            metadata[
                "affine_applied"
            ] = True

            metadata[
                "angle"
            ] = angle

            metadata[
                "translate_x"
            ] = translate_x

            metadata[
                "translate_y"
            ] = translate_y

            metadata[
                "scale"
            ] = scale


        # ====================================================
        # REMEMBER BLACK REGIONS
        # ====================================================
        #
        # Our 512x512 preprocessing introduces black padding.
        #
        # After rotation/translation there may also be
        # newly created black borders.
        #
        # We remember these pixels so brightness/contrast
        # augmentation cannot turn artificial padding gray.
        # ====================================================

        zero_pixels = (
            image <= 0.0
        )


        # ====================================================
        # 2. RANDOM BRIGHTNESS
        #
        # IMAGE ONLY.
        #
        # Never change mask brightness.
        # ====================================================

        if (
            random.random()
            < self.brightness_probability
        ):

            brightness_factor = (
                random.uniform(
                    self.min_brightness,
                    self.max_brightness,
                )
            )

            image = TF.adjust_brightness(
                image,
                brightness_factor,
            )

            metadata[
                "brightness_applied"
            ] = True

            metadata[
                "brightness_factor"
            ] = brightness_factor


        # ====================================================
        # 3. RANDOM CONTRAST
        #
        # IMAGE ONLY.
        # ====================================================

        if (
            random.random()
            < self.contrast_probability
        ):

            contrast_factor = (
                random.uniform(
                    self.min_contrast,
                    self.max_contrast,
                )
            )

            image = TF.adjust_contrast(
                image,
                contrast_factor,
            )

            metadata[
                "contrast_applied"
            ] = True

            metadata[
                "contrast_factor"
            ] = contrast_factor


        # ====================================================
        # RESTORE ARTIFICIAL BLACK PADDING
        # ====================================================

        image = image.masked_fill(
            zero_pixels,
            0.0,
        )


        # ====================================================
        # SAFETY CLEANUP
        # ====================================================

        # Keep ultrasound intensity inside [0,1].
        image = torch.clamp(
            image,
            min=0.0,
            max=1.0,
        )


        # Even though nearest-neighbor should preserve
        # binary labels, enforce binary mask explicitly.
        mask = (
            mask >= 0.5
        ).float()


        # ====================================================
        # FINAL SAFETY CHECKS
        # ====================================================

        if not torch.isfinite(
            image
        ).all():

            raise ValueError(
                "Augmentation produced "
                "non-finite image values."
            )

        if not torch.isfinite(
            mask
        ).all():

            raise ValueError(
                "Augmentation produced "
                "non-finite mask values."
            )

        unique_mask_values = set(
            torch.unique(
                mask
            ).tolist()
        )

        if not unique_mask_values.issubset(
            {
                0.0,
                1.0,
            }
        ):

            raise ValueError(
                "Augmentation produced "
                "non-binary mask values: "
                f"{sorted(unique_mask_values)}"
            )


        return (
            image,
            mask,
            metadata,
        )


# ============================================================
# BASIC AUGMENTATION SMOKE TEST
# ============================================================

def main():

    print("=" * 70)
    print("TN3K AUGMENTATION BASIC TEST")
    print("=" * 70)


    # ========================================================
    # SYNTHETIC ULTRASOUND
    # ========================================================

    image = torch.zeros(
        (1, 512, 512),
        dtype=torch.float32,
    )

    image[
        :,
        100:400,
        80:430,
    ] = 0.60


    # ========================================================
    # SYNTHETIC MASK
    # ========================================================

    mask = torch.zeros(
        (1, 512, 512),
        dtype=torch.float32,
    )

    mask[
        :,
        190:300,
        210:330,
    ] = 1.0


    # ========================================================
    # AUGMENTER
    # ========================================================

    augmenter = (
        TN3KTrainAugmentation()
    )


    print()
    print(
        "Running 10 random augmentations..."
    )


    for index in range(10):

        (
            augmented_image,
            augmented_mask,
            info,
        ) = augmenter(

            image.clone(),
            mask.clone(),
        )


        # ====================================================
        # SHAPE CHECK
        # ====================================================

        assert (
            augmented_image.shape
            == image.shape
        )

        assert (
            augmented_mask.shape
            == mask.shape
        )


        # ====================================================
        # FINITE VALUES CHECK
        # ====================================================

        assert torch.isfinite(
            augmented_image
        ).all()

        assert torch.isfinite(
            augmented_mask
        ).all()


        # ====================================================
        # IMAGE RANGE CHECK
        # ====================================================

        assert (
            augmented_image.min()
            >= 0.0
        )

        assert (
            augmented_image.max()
            <= 1.0
        )


        # ====================================================
        # BINARY MASK CHECK
        # ====================================================

        unique_mask_values = set(
            torch.unique(
                augmented_mask
            ).tolist()
        )

        assert (
            unique_mask_values
            .issubset(
                {
                    0.0,
                    1.0,
                }
            )
        )


        # ====================================================
        # MASK MUST SURVIVE
        # ====================================================

        assert (
            augmented_mask.sum()
            > 0
        )


        # ====================================================
        # PRINT
        # ====================================================

        print()

        print(
            f"Augmentation "
            f"{index + 1}"
        )

        print(
            f"  Affine: "
            f"{info['affine_applied']}"
        )

        print(
            f"  Angle: "
            f"{info['angle']:.2f}°"
        )

        print(
            f"  Translation: "
            f"("
            f"{info['translate_x']}, "
            f"{info['translate_y']}"
            f")"
        )

        print(
            f"  Scale: "
            f"{info['scale']:.4f}"
        )

        print(
            f"  Brightness: "
            f"{info['brightness_factor']:.4f}"
        )

        print(
            f"  Contrast: "
            f"{info['contrast_factor']:.4f}"
        )

        print(
            f"  Mask values: "
            f"{sorted(unique_mask_values)}"
        )


    print()

    print("=" * 70)
    print(
        "TN3K AUGMENTATION BASIC TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()