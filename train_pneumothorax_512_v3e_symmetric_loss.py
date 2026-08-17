from pathlib import Path

base_path = Path("/content/train_pneumothorax_512_negative_aware.py")
v3e_path = Path("/content/train_pneumothorax_512_v3e_symmetric_loss.py")
drive_v3e_path = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab/"
    "train_pneumothorax_512_v3e_symmetric_loss.py"
)

text = base_path.read_text(encoding="utf-8")


def replace_once(old, new):
    global text
    count = text.count(old)
    assert count == 1, (
        f"Expected exactly one occurrence but found {count}:\n{old}"
    )
    text = text.replace(old, new, 1)


# Allow smoke/full mode to be selected without editing the script.
replace_once(
    "import math\n",
    "import math\nimport os\n",
)

replace_once(
    "SMOKE_TEST = True",
    'SMOKE_TEST = os.environ.get("V3E_SMOKE_TEST", "1") == "1"',
)

# V3E controlled loss change.
replace_once("BCE_WEIGHT = 0.45", "BCE_WEIGHT = 0.65")
replace_once(
    "NEGATIVE_BCE_WEIGHT = 0.20",
    "NEGATIVE_BCE_WEIGHT = 0.00",
)

# Use the locked V3C epoch-5 checkpoint as the source.
replace_once(
    '''CHECKPOINT_DIRECTORY = Path("checkpoints")
SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / "pneumothorax_512_best.pth"
)''',
    '''CHECKPOINT_DIRECTORY = Path(
    "/content/drive/MyDrive/SmartMedicalImagingColab/checkpoints"
)
SOURCE_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY
    / "pneumothorax_512_v3c_batchnorm_stabilized_best.pth"
)''',
)

# Give V3E its own checkpoint files.
replacements = {
    "pneumothorax_512_negative_aware_smoke_last.pth":
        "pneumothorax_512_v3e_symmetric_loss_smoke_last.pth",
    "pneumothorax_512_negative_aware_last.pth":
        "pneumothorax_512_v3e_symmetric_loss_last.pth",
    "pneumothorax_512_negative_aware_smoke_best.pth":
        "pneumothorax_512_v3e_symmetric_loss_smoke_best.pth",
    "pneumothorax_512_negative_aware_best.pth":
        "pneumothorax_512_v3e_symmetric_loss_best.pth",
    "pneumothorax_512_negative_aware_smoke_test":
        "pneumothorax_512_v3e_symmetric_loss_smoke_test",
    "pneumothorax_512_negative_aware_finetune":
        "pneumothorax_512_v3e_symmetric_loss_finetune",
}

for old, new in replacements.items():
    assert old in text, f"Missing expected text: {old}"
    text = text.replace(old, new)

# Correct the checkpoint's loss description.
replace_once(
    '''"0.45 weighted BCE + 0.35 symmetric positive "
                    "focal Tversky + 0.20 negative-only BCE"''',
    '''"0.65 weighted BCE + 0.35 symmetric positive "
                    "focal Tversky; no negative-only BCE"''',
)

# Preserve saved statistics in every BatchNorm layer, including the decoder.
replace_once(
    '''    is_training = optimizer is not None
    model.train(is_training)

    # Preserve pretrained encoder BatchNorm running statistics.
    for encoder_module in model.encoder_modules():
        encoder_module.eval()
''',
    '''    is_training = optimizer is not None

    batchnorm_snapshot = {}
    if is_training:
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                and module.running_mean is not None
            ):
                batchnorm_snapshot[name] = (
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                    module.num_batches_tracked.detach().clone(),
                )

    model.train(is_training)

    # Preserve the encoder's evaluation behavior.
    for encoder_module in model.encoder_modules():
        encoder_module.eval()

    # V3C BatchNorm stabilization: all BatchNorm layers use their
    # saved running statistics while trainable affine parameters remain
    # controlled by requires_grad.
    if is_training:
        batchnorm_layers = [
            module
            for module in model.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        for module in batchnorm_layers:
            module.eval()

        assert all(
            not module.training for module in batchnorm_layers
        ), "A BatchNorm layer incorrectly remained in training mode."
''',
)

# Verify that no BatchNorm running statistics changed during training.
replace_once(
    '''    return finalize_metrics(
        metrics,
        time.perf_counter() - start_time,
    )
''',
    '''    results = finalize_metrics(
        metrics,
        time.perf_counter() - start_time,
    )

    if is_training:
        changed_layers = []

        for name, module in model.named_modules():
            if name not in batchnorm_snapshot:
                continue

            before = batchnorm_snapshot[name]
            after = (
                module.running_mean.detach(),
                module.running_var.detach(),
                module.num_batches_tracked.detach(),
            )

            if any(
                not torch.equal(old_value, new_value)
                for old_value, new_value in zip(before, after)
            ):
                changed_layers.append(name)

        if changed_layers:
            raise RuntimeError(
                "BatchNorm running statistics changed in: "
                + ", ".join(changed_layers)
            )

        print(
            "  BatchNorm verification: "
            f"{len(batchnorm_snapshot)} layers kept saved "
            "running statistics"
        )

    return results
''',
)

# Improve the console description.
replace_once(
    'print("Negative-aware 512 x 512 pneumothorax fine-tuning")',
    'print("V3E symmetric-loss 512 x 512 pneumothorax fine-tuning")',
)

replace_once(
    '    print("Training augmentation: enabled")',
    '''    print("Controlled experiment: V3E loss change only")
    print("Loss: 0.65 weighted BCE + 0.35 positive focal Tversky")
    print("Negative-only BCE term: disabled")
    print("BatchNorm: all layers use saved running statistics")
    print("Training augmentation: enabled")''',
)

replace_once(
    'print("\\nNegative-aware fine-tuning finished.")',
    'print("\\nV3E symmetric-loss fine-tuning finished.")',
)

replace_once(
    '"  New best negative-aware checkpoint: "',
    '"  New best V3E checkpoint: "',
)

# Save in /content and persist a copy in Google Drive.
v3e_path.write_text(text, encoding="utf-8")
drive_v3e_path.write_text(text, encoding="utf-8")

print("V3E script created:")
print(v3e_path)
print("\nPersistent Drive copy:")
print(drive_v3e_path)

# Final safety checks.
created = v3e_path.read_text(encoding="utf-8")

assert "BCE_WEIGHT = 0.65" in created
assert "NEGATIVE_BCE_WEIGHT = 0.00" in created
assert "TRAINING_POSITIVE_FRACTION = 0.35" in created
assert "V3E_SMOKE_TEST" in created
assert "v3c_batchnorm_stabilized_best.pth" in created
assert "split=\"test\"" not in created

print("\nV3E CONFIGURATION CHECK PASSED")
print("Sampling: 35% positive / 65% negative")
print("Loss: 0.65 BCE + 0.35 positive focal Tversky")
print("Source: locked V3C epoch-5 checkpoint")
print("Test split: not instantiated")