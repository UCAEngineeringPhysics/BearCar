from benchy import benchmark
import torch.nn as nn
import torchvision.models as models

effnet_scratch = models.efficientnet_b0(weights=None, num_classes=10)
print("Benchmarking EfficientNet-B0 (From Scratch)...")
trained_effnet_scratch = benchmark(effnet_scratch, epochs=20)
