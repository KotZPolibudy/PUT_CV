import torch.nn as nn
import torch.nn.functional as F

class CifarClassifier(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(CifarClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        conv_output_size = input_size // 4  # bo 2x pool (2x2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * conv_output_size * conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

