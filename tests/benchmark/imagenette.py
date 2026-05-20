import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ImageBearNet(nn.Module):
    def __init__(self):
        super(ImageBearNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3)

        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):  # 224
        x = self.relu(self.conv1(x))  # (224 - 7 + 2 * 3) / 2 + 1 = 112.5
        x = self.max_pool(x)  # (112 - 3 + 2 * 1) / 2 + 1 = 56.5
        x = self.relu(self.conv2(x))  # (56 - 3) + 1 = 54
        x = self.relu(self.conv3(x))  # (54 - 3) / 2 + 1 = 26.5
        x = self.relu(self.conv4(x))  # (26 - 3) + 1 = 24
        x = self.relu(self.conv5(x))  # (24 - 3) / 2 + 1 = 11.5
        x = self.relu(self.conv6(x))  # (11 - 3) + 1 = 9
        x = self.relu(self.conv7(x))  # (9 - 3) + 1 = 7

        x = x.view(x.size(0), -1)  # flatten

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        y = self.fc3(x)
        return y


# Use GPU or Apple Silicon if available, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Set up transformations for the raw data
transform = transforms.Compose(
    [
        transforms.Resize(256),  # Scale image up slightly
        transforms.CenterCrop(224),  # Crop the exact 224x224 center
        transforms.ToTensor(),  # Convert to PyTorch Tensor
        transforms.Normalize(  # Standard ImageNet normalization
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Download and Load Imagenette
print("Loading Imagenette datasets...")
trainset = torchvision.datasets.Imagenette(
    root="./data", split="train", size="320px", download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

valset = torchvision.datasets.Imagenette(
    root="./data", split="val", size="320px", download=True, transform=transform
)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

# Initialize Model, Loss, and Optimizer
model = ImageBearNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training & Validation Loop
epochs = 20
print(f"Starting training for {epochs} epochs...")

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

    # torch.no_grad() disables tracking history to save memory and speed up computations
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Get the index of the max log-probability (the predicted class)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    # Print the epoch summary
    print(
        f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%"
    )

print("Finished 20 Epochs!")
