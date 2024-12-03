import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64*8*13, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               #   300x300                     #  120x160
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148     |     (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72      |     (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34     |     (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32         |     12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30         |     10 - 3 + 1  = 8    15 - 3 + 1  = 13

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EfficientBearNet(nn.Module):
    """
    Following Perplexity's suggestion from prompt:
        I would like to use EfficientNet-B3 as the backbone of my model, 
        and using transfer learning to make it predict am autonomous vehicles throttle and steering values based on an image input.
        Can you show me how to realize this using pytorch?
    """
    def __init__(self, num_ouputs=2):
        super(EfficientBearNet, self).__init__()
        
        # Load pre-trained EfficientNet-B3
        # self.efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.efficientnet = efficientnet_b3(weights=None)
        
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        return self.efficientnet(x)


if __name__ == '__main__':
    # Create the custom model
    model = DonkeyNet()  # Adjust num_classes as needed
    # model = CustomEfficientNetB3(num_ouputs=2)  # Adjust num_classes as needed
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")

    # Example usage
    # batch_size = 4
    # image_input = torch.randn(batch_size, 3, 300, 300)  # Assuming 300x300 input images
    # additional_input = torch.randn(batch_size, 1800)
    #
    # output = model(image_input, additional_input)
    # print(output.shape)  # Should print torch.Size([4, 10]) for batch_size=4 and num_classes=10

