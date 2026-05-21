"""
Gemini 3.1 Generated this
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def benchmark_model(model, input_size=(224, 224), epochs=20, batch_size=64, lr=1e-4):
    """
    Trains and evaluates a PyTorch model on Imagenette.
    Accepts input_size as an int (square) or a tuple (height, width).
    """
    # 1. Parse input_size to handle both ints and tuples
    if isinstance(input_size, int):
        crop_h, crop_w = input_size, input_size
    else:
        crop_h, crop_w = input_size

    # 2. Set up Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    print(f"--- Starting Benchmark ---")
    print(f"Device: {device} | Input Size: {crop_h}x{crop_w} | Epochs: {epochs}")

    # 3. Dynamic Transform Scaling (maintaining the zoom-to-crop ratio)
    resize_h = int(crop_h * (256 / 224))
    resize_w = int(crop_w * (256 / 224))

    transform = transforms.Compose(
        [
            transforms.Resize((resize_h, resize_w)),
            transforms.CenterCrop((crop_h, crop_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 4. Load Data
    trainset = torchvision.datasets.Imagenette(
        root="./data", split="train", size="320px", download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    valset = torchvision.datasets.Imagenette(
        root="./data", split="val", size="320px", download=True, transform=transform
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 5. Setup Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training Loop
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)

        # --- Validation Phase ---
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
        )

    print("Benchmark Complete!\n")
    return model
