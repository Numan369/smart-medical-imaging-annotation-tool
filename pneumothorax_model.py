import torch
import torch.nn as nn

from torchvision.models import (
    ResNet34_Weights,
    resnet34,
)

from pneumothorax_dataloaders import create_dataloaders


class DoubleConvolution(nn.Module):
    """Apply two convolutions to refine image features."""

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class DecoderBlock(nn.Module):
    """Enlarge features and combine them with encoder features."""

    def __init__(
        self,
        input_channels,
        skip_channels,
        output_channels,
    ):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            input_channels,
            output_channels,
            kernel_size=2,
            stride=2,
        )

        self.convolutions = DoubleConvolution(
            output_channels + skip_channels,
            output_channels,
        )

    def forward(self, inputs, skip_features=None):
        outputs = self.upsample(inputs)

        if skip_features is not None:
            if outputs.shape[-2:] != skip_features.shape[-2:]:
                raise ValueError(
                    "Decoder and skip-connection dimensions "
                    "do not match."
                )

            outputs = torch.cat(
                (outputs, skip_features),
                dim=1,
            )

        return self.convolutions(outputs)


class PneumothoraxResNet34UNet(nn.Module):
    """U-Net using an ImageNet-pretrained ResNet34 encoder."""

    def __init__(
        self,
        use_pretrained_encoder=True,
        freeze_encoder=True,
    ):
        super().__init__()

        if use_pretrained_encoder:
            weights = ResNet34_Weights.DEFAULT
        else:
            weights = None

        backbone = resnet34(weights=weights)

        # Pretrained ResNet34 encoder.
        self.encoder_stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )

        self.encoder_pool = backbone.maxpool
        self.encoder_1 = backbone.layer1
        self.encoder_2 = backbone.layer2
        self.encoder_3 = backbone.layer3
        self.encoder_4 = backbone.layer4

        # New segmentation decoder.
        self.decoder_4 = DecoderBlock(
            input_channels=512,
            skip_channels=256,
            output_channels=256,
        )

        self.decoder_3 = DecoderBlock(
            input_channels=256,
            skip_channels=128,
            output_channels=128,
        )

        self.decoder_2 = DecoderBlock(
            input_channels=128,
            skip_channels=64,
            output_channels=64,
        )

        self.decoder_1 = DecoderBlock(
            input_channels=64,
            skip_channels=64,
            output_channels=64,
        )

        self.decoder_0 = DecoderBlock(
            input_channels=64,
            skip_channels=0,
            output_channels=32,
        )

        self.output_layer = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
        )

        # ImageNet normalization values expected by ResNet34.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "imagenet_standard_deviation",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def encoder_modules(self):
        """Return all modules belonging to the encoder."""

        return (
            self.encoder_stem,
            self.encoder_pool,
            self.encoder_1,
            self.encoder_2,
            self.encoder_3,
            self.encoder_4,
        )

    def encoder_parameters(self):
        """Yield all encoder parameters."""

        for module in self.encoder_modules():
            yield from module.parameters()

    def freeze_encoder(self):
        """Prevent pretrained encoder parameters from changing."""

        for parameter in self.encoder_parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):
        """Allow encoder parameters to be fine-tuned."""

        for parameter in self.encoder_parameters():
            parameter.requires_grad = True

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(
                "Expected images shaped "
                "(batch, channels, height, width)."
            )

        if images.shape[1] != 1:
            raise ValueError(
                "Expected one-channel grayscale X-rays."
            )

        height, width = images.shape[-2:]

        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                "Image height and width must be divisible by 32."
            )

        # ResNet34 was pretrained using three-channel images.
        rgb_images = images.repeat(
            1,
            3,
            1,
            1,
        )

        normalized_images = (
            rgb_images - self.imagenet_mean
        ) / self.imagenet_standard_deviation

        # Encoder
        stem_features = self.encoder_stem(
            normalized_images
        )

        encoder_1 = self.encoder_1(
            self.encoder_pool(stem_features)
        )
        encoder_2 = self.encoder_2(encoder_1)
        encoder_3 = self.encoder_3(encoder_2)
        encoder_4 = self.encoder_4(encoder_3)

        # Decoder with skip connections.
        decoder_4 = self.decoder_4(
            encoder_4,
            encoder_3,
        )
        decoder_3 = self.decoder_3(
            decoder_4,
            encoder_2,
        )
        decoder_2 = self.decoder_2(
            decoder_3,
            encoder_1,
        )
        decoder_1 = self.decoder_1(
            decoder_2,
            stem_features,
        )
        decoder_0 = self.decoder_0(
            decoder_1,
        )

        return self.output_layer(decoder_0)


def count_parameters(model, trainable_only=False):
    parameters = model.parameters()

    if trainable_only:
        parameters = (
            parameter
            for parameter in parameters
            if parameter.requires_grad
        )

    return sum(
        parameter.numel()
        for parameter in parameters
    )


def main():
    torch.manual_seed(42)

    data_loaders = create_dataloaders()

    training_batch = next(
        iter(data_loaders["train"])
    )

    images = training_batch["image"]
    target_masks = training_batch["mask"]

    print("Loading pretrained ResNet34 encoder...")

    model = PneumothoraxResNet34UNet(
        use_pretrained_encoder=True,
        freeze_encoder=True,
    )

    model.eval()

    with torch.no_grad():
        output_logits = model(images)
        output_probabilities = torch.sigmoid(
            output_logits
        )

    if output_logits.shape != target_masks.shape:
        raise ValueError(
            "Model output and target mask shapes do not match."
        )

    if not torch.isfinite(output_logits).all():
        raise ValueError(
            "The model produced non-finite values."
        )

    encoder_is_frozen = not any(
        parameter.requires_grad
        for parameter in model.encoder_parameters()
    )

    print("\nPretrained U-Net check")
    print("----------------------")
    print("Encoder: ImageNet-pretrained ResNet34")
    print(f"Encoder frozen: {encoder_is_frozen}")
    print(f"Input shape: {tuple(images.shape)}")
    print(
        f"Target-mask shape: "
        f"{tuple(target_masks.shape)}"
    )
    print(
        f"Model-output shape: "
        f"{tuple(output_logits.shape)}"
    )
    print(
        f"Total parameters: "
        f"{count_parameters(model):,}"
    )
    print(
        f"Initially trainable parameters: "
        f"{count_parameters(model, trainable_only=True):,}"
    )
    print(
        f"Probability range: "
        f"{output_probabilities.min().item():.4f} to "
        f"{output_probabilities.max().item():.4f}"
    )

    print(
        "\nThe pretrained encoder and new decoder "
        "passed the model check."
    )


if __name__ == "__main__":
    main()