from benchy import benchmark_model
import torch.nn as nn
import torchvision.models as models

# Passing num_classes=10 automatically overrides the default 1000-class output
convnext_blank = models.convnext_tiny(weights=None, num_classes=10)

# 2. Pass it to your benchmark function
print("Benchmarking ConvNeXt-Tiny (From Scratch)...")
trained_convnext = benchmark_model(convnext_blank, input_size=224, epochs=20)
