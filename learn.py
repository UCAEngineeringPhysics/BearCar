import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2
from torchvision.io import read_image
import matplotlib.pyplot as plt
from pilot_models import BearNet

# SETUP
# Pass in data directory name as CLI argument
# e.g. python train.py 2025-12-13-14-15
if len(sys.argv) != 2:
    print("Training script needs data! Please provide date and time!")
    sys.exit(1)  # exit with an error code
else:
    data_datetime = sys.argv[1]
# Designate processing unit for CNN training
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


# Define custom dataset
class BearCartDataset(Dataset):
    """
    Customized dataset
    """

    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.trans_in = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
        self.trans_out = v2.Compose([v2.ToDtype(torch.float32, scale=False)])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = self.trans_in(read_image(img_path))
        steering = torch.as_tensor(self.img_labels.iloc[idx, 1], dtype=torch.float32)
        throttle = torch.as_tensor(self.img_labels.iloc[idx, 2], dtype=torch.float32)
        return image, steering, throttle


# Define training process
def train(dataloader, model, loss_fn, optimizer):
    """
    training function for 1 epoch
    """
    model.train()
    num_used_samples = 0
    ep_loss = 0.0
    for i, (img, st, th) in enumerate(dataloader):
        target = torch.stack((st, th), dim=-1)
        feature, target = img.to(DEVICE), target.to(DEVICE)
        pred = model(feature)
        batch_loss = loss_fn(pred, target)
        optimizer.zero_grad()  # zero previous gradient
        batch_loss.backward()  # back propagation
        optimizer.step()  # update params
        num_used_samples += target.shape[0]
        print(
            f"batch loss: {batch_loss.item()} [{num_used_samples}/{len(dataloader.dataset)}]"
        )
        ep_loss = (ep_loss * i + batch_loss.item()) / (i + 1)
    return ep_loss


# Define validation process
def validate(dataloader, model, loss_fn):
    model.eval()
    ep_loss = 0.0
    with torch.no_grad():
        for i, (im, st, th) in enumerate(dataloader):
            target = torch.stack((st, th), dim=-1)
            feature, target = im.to(DEVICE), target.to(DEVICE)
            pred = model(feature)
            batch_loss = loss_fn(pred, target)
            ep_loss = (ep_loss * i + batch_loss.item()) / (i + 1)
    return ep_loss


# LOOP
# Instantiate dataset
data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", data_datetime
)
annotations_file = os.path.join(data_dir, "labels.csv")  # the name of the csv file
img_dir = os.path.join(
    data_dir, "images"
)  # the name of the folder with all the images in it
# Split train/val (9:1)
df = pd.read_csv(annotations_file, header=None, names=['image_id', 'steering_value', 'throttle_value'])
# val_ids = np.arange(int(len(df) * 0.1)) * 8
val_ids = np.arange(
    int(len(df) / 2) - int(len(df) * 0.1), int(len(df) / 2) + int(len(df) * 0.1)
)
val_annotates = df.iloc[val_ids]  # TODO: val annotates somehow kept 1st row in df
train_annotates = df.drop(val_ids)
val_annotates = val_annotates.reset_index(drop=True)
train_annotates = train_annotates.reset_index(drop=True)
print(f"train size: {len(train_annotates)}, validation size: {len(val_annotates)}")
train_annotates.to_csv(os.path.join(data_dir, "labels_train.csv"), index=False, header=False)
val_annotates.to_csv(os.path.join(data_dir, "labels_val.csv"), index=False, header=False)
train_label_file_path = os.path.join(data_dir, "labels_train.csv")  # the name of the csv file
val_label_file_path = os.path.join(
    data_dir, "labels_val.csv"
)  # the name of the csv file
train_set = BearCartDataset(train_label_file_path, img_dir)
val_set = BearCartDataset(val_label_file_path, img_dir)
train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=128)

# Instantiate model
model = BearNet().to(DEVICE)  # choose the architecture class from cnn_network.py

# Hyper-parameters
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
epochs = 64
patience = 7
best_loss = float("inf")  # best loss on test data
best_counter = 0
train_losses = []
val_losses = []
# Optimize the model
for ep in range(epochs):
    print(f"Epoch {ep + 1}\n-------------------------------")
    ep_train_loss = train(train_dataloader, model, loss_fn, optimizer)
    ep_test_loss = validate(val_dataloader, model, loss_fn)
    print(
        f"Epoch {ep + 1} training loss: {ep_train_loss}, testing loss: {ep_test_loss}"
    )
    train_losses.append(ep_train_loss)
    val_losses.append(ep_test_loss)
    # Early stopping
    if ep_test_loss < best_loss:
        best_loss = ep_test_loss
        best_counter = 0  # Reset counter if validation loss improved
        try:
            os.remove(os.path.join(data_dir, f"{model_name}.pth"))
            print("Last best model file has been deleted successfully.")
        except FileNotFoundError:
            print(f"File '{os.path.join(data_dir, f'{model_name}.pth')}' not found.")
        except Exception as e:
            print(f"Error occurred while deleting the file: {e}")
        model_name = (
            f"{model._get_name()}-{ep + 1}ep-{learning_rate}lr-{ep_test_loss:.4f}mse"
        )
        torch.save(model.state_dict(), os.path.join(data_dir, f"{model_name}.pth"))
        print(f"Best model saved as '{os.path.join(data_dir, f'{model_name}.pth')}'")
    else:
        best_counter += 1
        print(f"{best_counter} epochs since best model")
        if best_counter >= patience:
            print("Early stopping triggered!")
            break

print("Optimize Done!")

# Graph training process
plt.plot(range(len(train_losses)), train_losses, "b--", label="Training")
plt.plot(range(len(val_losses)), val_losses, "orange", label="Validation")
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title(model_name)
plt.savefig(os.path.join(data_dir, f"{model_name}.png"))
