import sys
import platform

import torch
import transformers
import datasets
import sklearn
import pandas
import numpy
import matplotlib
import seaborn
import accelerate


print("=" * 60)
print("TOXICITY DETECTION - ENVIRONMENT CHECK")
print("=" * 60)

print("\nSYSTEM")
print("-" * 60)
print("Python       :", sys.version.split()[0])
print("Platform     :", platform.platform())
print("Architecture :", platform.machine())

print("\nLIBRARY VERSIONS")
print("-" * 60)
print("PyTorch      :", torch.__version__)
print("Transformers :", transformers.__version__)
print("Datasets     :", datasets.__version__)
print("Scikit-learn :", sklearn.__version__)
print("Pandas       :", pandas.__version__)
print("NumPy        :", numpy.__version__)
print("Matplotlib   :", matplotlib.__version__)
print("Seaborn      :", seaborn.__version__)
print("Accelerate   :", accelerate.__version__)

print("\nHARDWARE")
print("-" * 60)

if torch.cuda.is_available():
    print("CUDA available : YES")
    print("GPU            :", torch.cuda.get_device_name(0))
    print("CUDA version   :", torch.version.cuda)

elif torch.backends.mps.is_available():
    print("Apple MPS      : YES")
    print("Device         : Apple Silicon GPU")

else:
    print("GPU            : NOT AVAILABLE")
    print("Device         : CPU")

print("\n" + "=" * 60)
print("ENVIRONMENT CHECK COMPLETED")
print("=" * 60)