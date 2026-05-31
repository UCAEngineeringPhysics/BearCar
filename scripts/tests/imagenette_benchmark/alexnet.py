from benchy import benchmark
import torch.nn as nn
import torchvision.models as models

alexnet_scratch = models.alexnet(weights=None, num_classes=10)
print("Benchmarking (the infamous) AlexNet...")
alexnet_trained = benchmark(alexnet_scratch, epochs=20)
