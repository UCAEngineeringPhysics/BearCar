from benchy import benchmark
import torch.nn as nn
import torchvision.models as models

effnet_scratch = models.efficientnet_v2_s(weights=None, num_classes=10)
print("Benchmarking EfficientNet-V2-s (From Scratch)...")
effnet_trained = benchmark(effnet_scratch, epochs=20, batch_size=32)
