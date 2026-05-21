from benchy import benchmark_model
import torch.nn as nn
import torchvision.models as models

# Passing num_classes=10 automatically overrides the default 1000-class output
effnet_scratch = models.efficientnet_b0(weights=None, num_classes=10)

# 2. Pass it to your benchmark function
print("Benchmarking EfficientNet-B0 (From Scratch)...")
trained_effnet_scratch = benchmark_model(effnet_scratch, epochs=20)
