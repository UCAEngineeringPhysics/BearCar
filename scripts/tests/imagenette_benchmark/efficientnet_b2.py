from benchy import benchmark
import torch.nn as nn
import torchvision.models as models

effnet_scratch = models.efficientnet_b2(weights=None, num_classes=10)
print("Benchmarking EfficientNet-B2 (From Scratch)...")
trained_effnet_scratch = benchmark(effnet_scratch, input_size=(260, 260), epochs=20, batch_size=32)
