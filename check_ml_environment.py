import platform

import numpy as np
import pydicom


print("Machine-learning environment")
print("----------------------------")
print(f"Python version: {platform.python_version()}")
print(f"NumPy version: {np.__version__}")
print(f"pydicom version: {pydicom.__version__}")


try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_properties = torch.cuda.get_device_properties(
            device_index
        )

        total_memory_gb = (
            device_properties.total_memory / (1024 ** 3)
        )

        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {device_properties.name}")
        print(f"GPU memory: {total_memory_gb:.2f} GB")
        print("Training device: GPU")
    else:
        print("Training device: CPU")
        print(
            "Note: training a segmentation model on CPU "
            "will be very slow."
        )

except ImportError:
    print("PyTorch version: NOT INSTALLED")
    print("CUDA available: unavailable")
    print("Training device: unavailable")


try:
    import torchvision

    print(f"torchvision version: {torchvision.__version__}")

except ImportError:
    print("torchvision version: NOT INSTALLED")