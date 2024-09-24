# From TinyTracker

import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchinfo import summary


class GTKImageModel(nn.Module):
    # Used for both eyes(with shared weights) and the face (with unique weights)
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=7,
                      stride=4, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 256, kernel_size=5, stride=1,
                      padding='same', groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding='valid'),
            nn.ReLU(),
        )

    def forward(self, inputs):
        return self.features(inputs)


class TinyTracker(nn.Module):
    def __init__(self, in_channels=3, backbone="mobilenetv3"):
        super().__init__()

        self.conv_g = nn.Conv2d(3, in_channels,
                                kernel_size=1, stride=1, padding='valid')
        if backbone == 'mobilenetv3':
            pretrained_weights = MobileNet_V3_Small_Weights.DEFAULT
            self.faceModel = mobilenet_v3_small(
                weights=pretrained_weights).features
            self.conv_fm = nn.Conv2d(576, 2, kernel_size=1,
                                     stride=1, padding='valid')
            n_input = 32
        else:
            self.faceModel = GTKImageModel(in_channels)
            self.conv_fm = nn.Conv2d(64, 2, kernel_size=1,
                                     stride=1, padding='valid')
            n_input = 72

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(n_input, 128),  # assume input size = 112 by 112
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Softmax()
        )  # Coordinate (x, y)

    def forward(self, faces):
        x = self.conv_g(faces)
        print(x.shape)
        x = self.faceModel(x)
        print(x.shape)
        x = self.conv_fm(x)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    dummy_data = torch.randn(2, 3, 112, 112)

    model = TinyTracker(3, None)
    output = model(dummy_data)
    summary(model, dummy_data.size())

    model = TinyTracker(3)
    output = model(dummy_data)
    summary(model, dummy_data.size())
